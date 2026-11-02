"""Tests for reading tickets (FR-01). Test ids T-01a, T-01b and T-01c come from the Stage 3 specification."""
import json

import pytest

from src import config
from src.ingest import CHANNELS, Ticket, load_raw, normalise


@pytest.fixture(scope="module")
def validation():
    return load_raw(config.DATA_DIR / "validation_tickets.json")


def test_T01a_one_ticket_per_channel_normalises(validation):
    seen = {}
    for raw in validation:
        t = normalise(raw)
        seen.setdefault(t.channel, t)
    assert set(seen) == set(CHANNELS)
    for channel, t in seen.items():
        assert isinstance(t, Ticket)
        assert t.channel == channel
        assert t.body
        assert t.original["ticket_id"] == t.ticket_id


def test_chat_tickets_have_empty_subject_but_usable_text(validation):
    chat = next(normalise(r) for r in validation if r["channel"] == "chat")
    assert chat.subject == ""
    assert chat.text == chat.body


def test_labels_are_not_copied_into_the_ticket(validation):
    t = normalise(validation[0])
    d = t.to_dict()
    assert "labels" not in d and "history" not in d


def test_T01b_empty_body_and_missing_fields_are_processed_not_dropped():
    t = normalise({"ticket_id": "X-1", "channel": "email", "body": "   "})
    assert t.ticket_id == "X-1"
    assert t.body == ""
    assert "empty_body" in t.issues
    assert "missing_customer_tier" in t.issues
    assert t.customer_tier == "unknown"


@pytest.mark.parametrize("raw", [None, 42, "text", [], {}, {"channel": "fax"}, {"received_at": "yesterday"}])
def test_T01b_never_raises_on_bad_input(raw):
    t = normalise(raw)
    assert t.ticket_id
    assert t.channel in CHANNELS or t.channel == "unknown"


def test_T01c_non_ascii_preserved_and_control_characters_removed():
    t = normalise({"ticket_id": "X-2", "channel": "forum", "body": "Configuração não funciona \u0007 — 東京 region\r\n\r\n\r\n\r\nthanks"})
    assert "Configuração não funciona" in t.body
    assert "東京" in t.body
    assert "\u0007" not in t.body
    assert "\n\n\n" not in t.body


def test_unknown_channel_is_flagged():
    t = normalise({"ticket_id": "X-3", "channel": "Fax", "body": "hello"})
    assert t.channel == "unknown"
    assert "unknown_channel:fax" in t.issues


def test_load_raw_accepts_wrapped_list_and_jsonl(tmp_path):
    wrapped = tmp_path / "t.json"
    wrapped.write_text(json.dumps({"tickets": [{"ticket_id": "A"}]}))
    assert load_raw(wrapped) == [{"ticket_id": "A"}]
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text('{"ticket_id": "A"}\nnot json\n')
    assert load_raw(jsonl) == [{"ticket_id": "A"}, {"_unparseable_line": 2}]
