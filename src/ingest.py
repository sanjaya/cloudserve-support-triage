"""Ingest (FR-01): read a ticket from any channel and put it in one standard shape.

How it works:
- Every ticket, from email, chat, docs comments or the forum, becomes the same `Ticket`.
  Later steps then do not need special code for each channel.
- We keep the original ticket and its channel, because later steps need both.
- `normalise()` never crashes. If a field is missing, the text is empty or the input is
  not even a ticket, we still return a Ticket and write the problem in `issues`.
- We do NOT copy the answer labels into the Ticket. The system must not see the
  answers it is tested against. The test harness reads the labels on its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CHANNELS = ("email", "chat", "docs_comment", "forum")

# Invisible control characters (except tab and new line). They can break prompts and logs.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class Ticket:
    ticket_id: str
    channel: str
    subject: str
    body: str
    received_at: str | None
    customer_id: str
    customer_tier: str
    customer_region: str
    language_fluency: str
    issues: list[str] = field(default_factory=list)
    original: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        """Subject and body joined together. Used to classify and to search the documents."""
        return f"{self.subject}\n{self.body}".strip()

    def to_dict(self, include_original: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_original:
            d.pop("original")
        return d


def clean_text(value: Any) -> str:
    """Tidy a piece of text.

    - None becomes an empty string.
    - Invisible control characters are removed.
    - Repeated spaces and blank lines are reduced.
    - Accents and other languages are kept as they are.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _parse_time(value: Any) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def normalise(raw: Any) -> Ticket:
    """Turn one raw ticket from any channel into a Ticket. This function never crashes."""
    issues: list[str] = []
    original = raw
    if not isinstance(raw, dict):
        issues.append(f"not_a_record:{type(raw).__name__}")
        raw = {}

    ticket_id = clean_text(raw.get("ticket_id"))
    if not ticket_id:
        digest = hashlib.sha1(json.dumps(original, sort_keys=True, default=str).encode()).hexdigest()[:8]
        ticket_id = f"UNKNOWN-{digest}"
        issues.append("missing_ticket_id")

    channel = clean_text(raw.get("channel")).lower()
    if channel not in CHANNELS:
        issues.append(f"unknown_channel:{channel or 'missing'}")
        channel = "unknown"

    subject = clean_text(raw.get("subject"))  # chat tickets never have a subject; that is normal
    body = clean_text(raw.get("body"))
    if not body:
        issues.append("empty_body")

    received_raw = raw.get("received_at")
    received_at = _parse_time(received_raw)
    if received_raw and received_at is None:
        issues.append("unparseable_received_at")
    elif not received_raw:
        issues.append("missing_received_at")

    def field_or(name: str, default: str = "unknown") -> str:
        value = clean_text(raw.get(name))
        if not value:
            issues.append(f"missing_{name}")
            return default
        return value

    return Ticket(
        ticket_id=ticket_id,
        channel=channel,
        subject=subject,
        body=body,
        received_at=received_at,
        customer_id=field_or("customer_id"),
        customer_tier=field_or("customer_tier"),
        customer_region=field_or("customer_region"),
        language_fluency=field_or("language_fluency"),
        issues=issues,
        original=raw,
    )


def load_raw(path: str | Path) -> list[Any]:
    """Read a ticket file. Three formats are accepted:
    a JSON list, a JSON object with a "tickets" list, or a JSON Lines file (.jsonl).
    If the file cannot be read at all, we stop with an error, because there is nothing to process."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        records = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_unparseable_line": n})
        return records
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("tickets"), list):
        return data["tickets"]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected a list of tickets")


def main() -> None:
    """Checkpoint 1: print one cleaned-up ticket for each channel, so we can verify"""
    parser = argparse.ArgumentParser(description="Show one normalised ticket per channel.")
    parser.add_argument("--input", required=True, help="path to a ticket file")
    args = parser.parse_args()
    shown = set()
    for raw in load_raw(args.input):
        ticket = normalise(raw)
        if ticket.channel not in shown:
            shown.add(ticket.channel)
            print(json.dumps(ticket.to_dict(), indent=2, ensure_ascii=False))
    missing = set(CHANNELS) - shown
    if missing:
        print(f"No tickets found for: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
