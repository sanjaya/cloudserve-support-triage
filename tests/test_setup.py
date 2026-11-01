"""Setup checks that run without network access or an API key."""
import json

from src import config


def test_corpus_present():
    docs = json.loads((config.DATA_DIR / "documentation.json").read_text())
    assert len(docs) == 29
    assert all("doc_id" in d for d in docs)


def test_validation_tickets_schema():
    tickets = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())
    assert len(tickets) == 80
    assert {t["channel"] for t in tickets} <= {"email", "chat", "docs_comment", "forum"}


def test_blocked_intents():
    assert "security_incident" in config.BLOCKED_INTENTS
