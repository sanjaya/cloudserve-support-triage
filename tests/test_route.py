"""Tests for deciding: answer automatically or send to a person (FR-04, FR-05, B-07).
Test ids T-04a, T-04c and T-05a come from the Stage 3 specification (T-04b, the threshold
sweep table, is written by scripts/sweep_threshold.py on the development tickets)."""
import json

import pytest

from evaluation.metrics import compute
from src import config
from src.classify import Classification
from src.retrieve import Passage
from src.route import AUTO, ESCALATE, RULES, route

PASSAGE = Passage("DOC-DEPLOY-002#resolution", "DOC-DEPLOY-002", "Rolling back", "resolution", "text", 0.6)
RAW = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())


def label(intent="rate_limit", confidence=0.97, method="agree"):
    return Classification(intent, "medium", confidence, [], method)


def test_T04a_same_input_gives_the_same_decision():
    for args in [(label(), [PASSAGE]), (label(confidence=0.5), [PASSAGE]), (label(), [])]:
        assert route(*args) == route(*args)


def test_all_conditions_met_answers_automatically():
    d = route(label(), [PASSAGE], threshold=0.80)
    assert (d.route, d.rule, d.threshold) == (AUTO, "all_conditions_met", 0.80)
    assert "97% sure" in d.reason and "1 help passage matched" in d.reason


def test_confidence_below_threshold_goes_to_a_person():
    d = route(label(confidence=0.6667, method="in_top3"), [PASSAGE], threshold=0.80)
    assert (d.route, d.rule) == (ESCALATE, "low_confidence")
    assert "only 67% sure" in d.reason and "needs 80%" in d.reason


def test_confidence_equal_to_threshold_is_enough():
    assert route(label(confidence=0.80), [PASSAGE], threshold=0.80).route == AUTO


def test_no_matching_passage_goes_to_a_person():
    d = route(label(), [], threshold=0.80)
    assert (d.route, d.rule) == (ESCALATE, "no_passage")


def test_unclassified_ticket_goes_to_a_person():
    d = route(Classification("unclear_request", "medium", 0.0, [], "fallback"), [PASSAGE])
    assert (d.route, d.rule) == (ESCALATE, "not_classified")


@pytest.mark.parametrize("intent", sorted(config.BLOCKED_INTENTS))
def test_never_automate_topics_go_to_a_person_however_sure(intent):
    d = route(label(intent, confidence=1.0), [PASSAGE], threshold=0.0)
    assert (d.route, d.rule) == (ESCALATE, "never_automate")
    assert "FR-05" in d.reason


def test_T05a_every_must_not_auto_respond_ticket_escalates():
    """Even if the classifier were 100% sure of the right intent and passages were found,
    none of the never-automate tickets in the validation set may be routed to an automatic answer."""
    must_not = [r for r in RAW if r["labels"]["must_not_auto_respond"]]
    assert must_not
    for r in must_not:
        assert route(label(r["labels"]["intent"], confidence=1.0), [PASSAGE], threshold=0.0).route == ESCALATE


def test_rule_order_never_automate_comes_before_confidence():
    d = route(label("security_incident", confidence=0.1), [], threshold=0.80)
    assert d.rule == "never_automate"


def test_every_rule_name_is_known():
    seen = {route(label(), [PASSAGE]).rule, route(label(confidence=0.1), [PASSAGE]).rule,
            route(label(), []).rule, route(label("feature_request"), [PASSAGE]).rule,
            route(Classification("unclear_request", "medium", 0.0, [], "fallback"), []).rule}
    assert seen <= set(RULES)


def _record(routing, expected, must_not=False, intent="rate_limit", true_intent="rate_limit"):
    return {"latency_s": 0.1,
            "result": {"action": "escalate", "reason": "r", "sources": [], "intent": intent,
                       "routing_decision": routing, "routing_rule": "x"},
            "labels": {"expected_route": expected, "must_not_auto_respond": must_not, "intent": true_intent}}


def test_T04c_routing_accuracy_uses_the_routing_decision():
    m = compute([_record("auto_respond", "auto_respond"), _record("escalate", "escalate"),
                 _record("auto_respond", "escalate"), _record("escalate", "auto_respond")])
    assert m["technical"]["routing_accuracy_pct"] == 50.0
    assert m["technical"]["routed_to_automatic_answer_pct"] == 50.0
    assert m["business"]["first_contact_resolution_pct"] == 0.0     # nothing is actually answered yet


def test_never_automate_tickets_routed_automatic_are_counted():
    m = compute([_record("auto_respond", "escalate", must_not=True), _record("escalate", "escalate", must_not=True)])
    assert m["governance"]["routed_automatic_on_never_automate_tickets"] == 1
