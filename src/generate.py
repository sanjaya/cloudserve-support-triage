"""Write the answer from the help articles (FR-06, backlog item B-08).

Only tickets that decide (src/route.py) sent to an automatic answer come here.

How it works:
1. The language model gets prompt PR-03 with the ticket and the passages found by search.
   Each passage starts with its id in square brackets, e.g. [DOC-AUTH-004#resolution].
   It must use only those passages, cite one after every fact, and say when the passages
   do not answer the question. It replies in a fixed JSON shape:
   {"status": "answered" | "partial" | "insufficient", "answer": ..., "citations": [...],
    "unanswered_points": [...]}
2. This code then checks the reply. The answer may go to the customer ONLY if all hold:
   - the model said "answered" (not "partial" or "insufficient")
   - the answer is not empty
   - it cites at least one passage
   - every passage it cites (in the list and in the text) is one search actually found
3. Otherwise the ticket goes to a person, with the draft attached so they can use it.
4. The line saying the reply was drafted automatically is added by code if the model left it out.

Safety checks on the answer itself (private data, unsupported claims, promises) come next, in
B-09. generate() never crashes: any failure means the ticket goes to a person.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src import config
from src.classify import Classification
from src.ingest import Ticket
from src.llm import LLMClient, LLMError, get_client, render
from src.retrieve import Passage

# Prompt versions for writing the answer. Which one is used is set by ANSWER_PROMPT_VERSION
# (.env); scripts/evaluate_answers.py compares them on development tickets.
PROMPTS = {"1.0": "build/PR-03_grounded_answer_v1.0.md",
           "1.1": "build/PR-03_grounded_answer_v1.1.md"}
DISCLOSURE = ("This reply was drafted automatically from CloudServe's documentation. "
              "Reply to this message and a member of the support team will follow up.")
STATUSES = ("answered", "partial", "insufficient")
# A citation in the text looks like [DOC-AUTH-004#resolution]
_CITATION = re.compile(r"\[(DOC-[A-Za-z0-9_-]+#[A-Za-z0-9_-]+)\]")


@dataclass
class Draft:
    status: str                         # answered / partial / insufficient, or "failed"
    answer: str = ""
    citations: list[str] = field(default_factory=list)          # passage ids the answer uses
    unanswered_points: list[str] = field(default_factory=list)
    send: bool = False                  # True only if every check below passed
    check: str = ""                     # which check decided: see CHECKS
    reason: str = ""                    # in words a support manager can read
    prompt_version: str = ""            # e.g. "PR-03 v1.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Short names for the check that decided, used in the metrics report and the decision log.
CHECKS = ("ok", "model_failed", "not_answered", "empty_answer", "no_citation", "unknown_citation")


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in passages)


def check_reply(reply: dict[str, Any], passages: list[Passage], prompt_version: str = "") -> Draft:
    """Turn the model's JSON into a Draft and decide whether it may be sent."""
    status = reply.get("status")
    status = status if status in STATUSES else "failed"
    answer = str(reply.get("answer") or "").strip()
    listed = [str(c).strip("[] ") for c in (reply.get("citations") or []) if str(c).strip()]
    in_text = _CITATION.findall(answer)
    cited = list(dict.fromkeys(listed + in_text))
    unanswered = [str(u) for u in (reply.get("unanswered_points") or []) if str(u).strip()]
    found = {p.passage_id for p in passages}
    unknown = [c for c in cited if c not in found]

    def draft(send: bool, check: str, reason: str) -> Draft:
        return Draft(status, answer, cited, unanswered, send, check, reason, prompt_version)

    if status == "failed":
        return draft(False, "model_failed", "The answer could not be read from the model's reply, "
                                            "so a person will handle it.")
    if status != "answered":
        gap = f" Not covered: {'; '.join(unanswered)}." if unanswered else ""
        what = "only partly answer" if status == "partial" else "do not answer"
        return draft(False, "not_answered", f"The help articles {what} this question, so a person will "
                                            f"handle it, with the draft attached.{gap}")
    if not answer:
        return draft(False, "empty_answer", "The model said it answered but gave no text, so a person will handle it.")
    if not cited:
        return draft(False, "no_citation", "The answer does not cite any help article, so it cannot be "
                                           "checked and a person will handle it.")
    if unknown:
        return draft(False, "unknown_citation", f"The answer cites {', '.join(unknown)}, which search did not "
                                                "find, so it cannot be trusted and a person will handle it.")
    if DISCLOSURE not in answer:
        answer = f"{answer.rstrip()}\n\n{DISCLOSURE}"
    return Draft(status, answer, cited, unanswered, True, "ok",
                 f"Answered from {len(set(c.split('#')[0] for c in cited))} help article(s), "
                 f"citing {', '.join(cited)}.", prompt_version)


def generate(ticket: Ticket, label: Classification, passages: list[Passage],
             client: LLMClient | None = None, version: str | None = None) -> Draft:
    """Draft and check the answer with PR-03 (version from ANSWER_PROMPT_VERSION). Never raises."""
    version = version or config.ANSWER_PROMPT_VERSION
    name = f"PR-03 v{version}"
    try:
        if version not in PROMPTS:
            raise LLMError(f"unknown ANSWER_PROMPT_VERSION {version!r}; use one of {list(PROMPTS)}")
        system, user = render(PROMPTS[version], {
            "channel": ticket.channel, "intent": label.intent, "urgency": label.urgency,
            "subject": ticket.subject, "body": ticket.body, "passages": format_passages(passages)})
        reply = (client or get_client()).complete_json(system, user, max_tokens=700)
    except LLMError as exc:
        return Draft("failed", send=False, check="model_failed", prompt_version=name,
                     reason=f"The answer could not be written ({exc}), so a person will handle it.")
    except Exception as exc:   # last line of defence
        return Draft("failed", send=False, check="model_failed", prompt_version=name,
                     reason=f"The answer could not be written ({type(exc).__name__}: {exc}), "
                            "so a person will handle it.")
    return check_reply(reply, passages, name)
