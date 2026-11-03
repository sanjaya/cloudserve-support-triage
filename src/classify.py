"""Classify each ticket (FR-02, backlog item B-06): what is it about, how urgent, how sure are we.

Two independent methods look at every ticket:

1. The LANGUAGE MODEL (prompt PR-02, through OpenRouter) DECIDES the intent.
   It understands new wording best: 76% right on the development tickets, with no training.
2. A small TRAINED classifier (embeddings + logistic regression, trained on the 500
   development tickets, saved in models/classifier.joblib) CHECKS that answer.

How sure we are depends on whether the two agree:
- "agree"     : the language model's intent is the trained classifier's first choice
- "in_top3"   : it is the trained classifier's 2nd or 3rd choice
- "disagree"  : the trained classifier did not rank it in its top 3

The confidence for each of these cases is the accuracy that case ACTUALLY achieved on the
development tickets (measured by scripts/evaluate_classifier.py and saved in
models/agreement_calibration.json). So when we say "90% sure", tickets in that case were
right about 90% of the time. Neither method's own number is used, because measurement showed
both were badly calibrated (see Stage 5 revision log, change 2).

If something is missing:
- language model unavailable -> trained classifier's answer, with the confidence measured for
                                the trained classifier alone ("trained_only")
- no trained model           -> language model's answer, with the confidence measured for the
                                language model alone ("llm_only")
- neither                    -> "unclear_request", confidence 0 (a person handles it)

classify() never crashes.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.ingest import Ticket
from src.llm import LLMClient, LLMError, fill, get_client, load_prompt

PROMPT_FILE = "build/PR-02_classify_intent_urgency_v1.0.md"
PROMPT_VERSION = "PR-02 v1.0"
LEVELS = ("agree", "in_top3", "disagree", "trained_only", "llm_only")

# Used ONLY if models/agreement_calibration.json is missing. Deliberately low, so that without
# measured numbers nothing is trusted enough to be answered automatically.
UNCALIBRATED = {level: 0.3 for level in LEVELS}


@dataclass
class Classification:
    intent: str
    urgency: str
    confidence: float                          # measured accuracy of this agreement case, 0 to 1
    alternatives: list[dict[str, Any]] = field(default_factory=list)   # trained classifier's other choices
    method: str = "agree"                      # one of LEVELS, or "fallback"
    llm_intent: str | None = None              # what the language model said
    trained_intent: str | None = None          # the trained classifier's first choice
    note: str = ""                             # plain-language note
    prompt_version: str | None = None          # set when the language model answered

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fallback(note: str) -> Classification:
    return Classification("unclear_request", "medium", 0.0, [], "fallback", None, None, note)


def agreement_level(llm_intent: str, ranked: list[tuple[str, float]]) -> str:
    """How the language model's answer relates to the trained classifier's ranking."""
    names = [name for name, _ in ranked]
    if names and llm_intent == names[0]:
        return "agree"
    if llm_intent in names[1:3]:
        return "in_top3"
    return "disagree"


# ------------------------------------------------------------------ near-duplicate groups


def near_duplicate_groups(texts: list[str], threshold: float = 0.5) -> list[int]:
    """Give each text a group number. Texts that share at least half their words
    (Jaccard similarity >= threshold) are in the same group.

    Why: the development tickets are mostly reworded copies of about 80 originals.
    When we test the classifier, a whole group must be either in training or in testing,
    never split. Otherwise the test only checks whether it memorised the copy."""
    words = [set(re.findall(r"[a-z]+", t.lower())) for t in texts]
    parent = list(range(len(texts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = words[i], words[j]
            if a and b and len(a & b) / len(a | b) >= threshold:
                parent[root(i)] = root(j)
    roots = {r: n for n, r in enumerate(dict.fromkeys(root(i) for i in range(len(texts))))}
    return [roots[root(i)] for i in range(len(texts))]


# ------------------------------------------------------------------ trained classifier


class TrainedClassifier:
    """Embedding + logistic regression. Used to CHECK the language model's answer."""

    def __init__(self, intent_model, urgency_model, embedder_name: str, info: dict[str, Any]):
        self.intent_model = intent_model
        self.urgency_model = urgency_model
        self.embedder_name = embedder_name
        self.info = info
        self._embedder = None

    @staticmethod
    def fit(texts: list[str], intents: list[str], urgencies: list[str], embedder=None,
            groups: list[int] | None = None) -> "TrainedClassifier":
        from sklearn.linear_model import LogisticRegression

        from src.retrieve import get_embedder

        embedder = embedder or get_embedder()
        X = embedder.embed(texts)
        groups = groups if groups is not None else near_duplicate_groups(texts)
        info = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tickets": len(texts), "groups": len(set(groups)), "embedder": embedder.name}
        intent_model = LogisticRegression(C=4.0, max_iter=3000).fit(X, intents)
        urgency_model = LogisticRegression(C=4.0, max_iter=3000).fit(X, urgencies)
        clf = TrainedClassifier(intent_model, urgency_model, embedder.name, info)
        clf._embedder = embedder
        return clf

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"intent": self.intent_model, "urgency": self.urgency_model,
                     "embedder": self.embedder_name, "info": self.info}, path)

    @staticmethod
    def load(path: str | Path) -> "TrainedClassifier":
        import joblib
        data = joblib.load(path)
        return TrainedClassifier(data["intent"], data["urgency"], data["embedder"], data["info"])

    def _embed(self, text: str):
        if self._embedder is None:
            from src.retrieve import get_embedder
            self._embedder = get_embedder(self.embedder_name)
        return self._embedder.embed([text])

    def predict(self, text: str) -> tuple[list[tuple[str, float]], str]:
        """Return (intents ranked, best first, with scores) and the most likely urgency.
        The scores are only used for ranking; they are not trusted as confidence."""
        x = self._embed(text)
        probs = self.intent_model.predict_proba(x)[0]
        ranked = sorted(zip(self.intent_model.classes_, probs), key=lambda p: -p[1])
        urg_probs = self.urgency_model.predict_proba(x)[0]
        urgency = self.urgency_model.classes_[int(urg_probs.argmax())]
        return [(str(i), round(float(p), 4)) for i, p in ranked], str(urgency)


# ------------------------------------------------------------------ language model (PR-02)


def ask_llm(ticket: Ticket, client: LLMClient | None = None) -> dict[str, Any]:
    """Ask the language model to classify the ticket with prompt PR-02. Raises LLMError
    if it fails or answers with a label that does not exist."""
    system, template = load_prompt(PROMPT_FILE)
    user = fill(template, {"channel": ticket.channel, "subject": ticket.subject, "body": ticket.body})
    reply = (client or get_client()).complete_json(system, user, max_tokens=200)
    intent, urgency = reply.get("intent"), reply.get("urgency")
    if intent not in config.INTENTS:
        raise LLMError(f"model gave an unknown intent: {intent!r}")
    if urgency not in config.URGENCIES:
        urgency = None
    try:
        confidence = float(reply.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"intent": intent, "urgency": urgency, "confidence": max(0.0, min(1.0, confidence))}


# ------------------------------------------------------------------ measured confidence


def load_calibration(path: str | Path | None = None) -> tuple[dict[str, float], str, str]:
    """Read the measured accuracy of each agreement case. Returns (accuracy per case,
    which method's urgency to use, note). Falls back to low uncalibrated values."""
    path = Path(path or config.CLASSIFIER_CALIBRATION_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        acc = {level: float(data["levels"][level]["accuracy"]) for level in LEVELS if level in data["levels"]}
        return {**UNCALIBRATED, **acc}, data.get("urgency_source", "llm"), ""
    except FileNotFoundError:
        return dict(UNCALIBRATED), "llm", f"no measured confidence at {path}; run scripts/evaluate_classifier.py --llm-sample 160"
    except Exception as exc:
        return dict(UNCALIBRATED), "llm", f"measured confidence could not be read ({type(exc).__name__})"


# ------------------------------------------------------------------ combined classifier


class Classifier:
    def __init__(self, trained: TrainedClassifier | None = None, client: LLMClient | None = None,
                 model_path: str | None = None, calibration_path: str | None = None):
        self.client = client
        self.trained = trained
        self.load_note = ""
        if self.trained is None:
            path = Path(model_path or config.CLASSIFIER_MODEL_PATH)
            try:
                loaded = TrainedClassifier.load(path)
                if loaded.embedder_name != config.EMBEDDING_MODEL:
                    self.load_note = (f"trained model uses embedder {loaded.embedder_name!r} but "
                                      f"{config.EMBEDDING_MODEL!r} is configured; not used")
                else:
                    self.trained = loaded
            except FileNotFoundError:
                self.load_note = f"no trained model at {path}; run scripts/train_classifier.py"
            except Exception as exc:   # damaged file, version mismatch, ...
                self.load_note = f"trained model could not be loaded: {type(exc).__name__}"
        self.accuracy, self.urgency_source, self.calibration_note = load_calibration(calibration_path)

    def classify(self, ticket: Ticket) -> Classification:
        """Never crashes. See the module notes for how the two methods are combined."""
        if not ticket.text:
            return fallback("The ticket has no text to classify.")
        try:
            return self._classify(ticket)
        except Exception as exc:   # last line of defence: a person handles the ticket
            return fallback(f"Classification failed ({type(exc).__name__}); sent to a person.")

    def _classify(self, ticket: Ticket) -> Classification:
        ranked, trained_urgency = self.trained.predict(ticket.text) if self.trained else ([], None)
        try:
            llm = ask_llm(ticket, self.client)
            llm_error = None
        except LLMError as exc:
            llm, llm_error = None, str(exc)

        note = self.calibration_note
        if llm and ranked:
            level = agreement_level(llm["intent"], ranked)
            intent = llm["intent"]
            urgency = (llm["urgency"] if self.urgency_source == "llm" else trained_urgency) or trained_urgency or "medium"
        elif ranked:                                   # language model unavailable
            level, intent, urgency = "trained_only", ranked[0][0], trained_urgency
            note = f"Language model not available ({llm_error}); used the trained classifier. {note}".strip()
        elif llm:                                      # no trained model
            level, intent, urgency = "llm_only", llm["intent"], llm["urgency"] or "medium"
            note = f"No trained model ({self.load_note}); used the language model only. {note}".strip()
        else:
            return fallback(f"No trained model ({self.load_note}) and language model not available ({llm_error}).")

        alternatives = [{"intent": i, "score": p} for i, p in ranked if i != intent][:2]
        return Classification(intent, urgency, round(self.accuracy[level], 4), alternatives, level,
                              llm["intent"] if llm else None, ranked[0][0] if ranked else None,
                              note, PROMPT_VERSION if llm else None)
