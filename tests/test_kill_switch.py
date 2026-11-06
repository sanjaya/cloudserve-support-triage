"""Tests for the kill switch (B-21, Governance Framework section 5): it stops automatic answers
at once, including an answer that is already being written, and every change is logged."""
import pytest

from src import kill_switch, pipeline
from tests.test_graph import TwoPromptLLM, steps, sure  # noqa: F401  (steps and sure are fixtures)


@pytest.fixture(autouse=True)
def switch_off():
    kill_switch.set_switch(False, "test")
    yield
    kill_switch.set_switch(False, "test")


def test_switch_on_sends_every_ticket_to_a_person(sure, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pipeline, "_llm_client", TwoPromptLLM("answered"))
    assert pipeline.process(sure).action == "auto_respond"          # normally answered
    kill_switch.set_switch(True, "test", "drill")
    out = pipeline.process(sure)
    assert out.action == "escalate" and out.routing_rule == "kill_switch"
    assert "kill switch" in out.reason and out.handoff["summary"]   # the person still gets a full note


def test_switch_turned_on_while_an_answer_is_being_written_stops_it(sure, monkeypatch):  # noqa: F811
    class TurnsSwitchOn(TwoPromptLLM):
        def complete_json(self, system, user, max_tokens=400):
            if "strict fact-checker" in system:           # during the safety checks
                kill_switch.set_switch(True, "test", "in flight")
            return super().complete_json(system, user, max_tokens)
    monkeypatch.setattr(pipeline, "_llm_client", TurnsSwitchOn("answered"))
    out = pipeline.process(sure)
    assert out.action == "escalate" and out.answer is None
    assert "before it was sent" in out.reason and out.draft["answer"]


def test_setting_in_the_environment_also_works(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "1")
    assert kill_switch.is_on() and kill_switch.why() == "KILL_SWITCH setting"


def test_command_line_and_log(capsys):
    kill_switch.main(["on", "--by", "Marcus Adeyemi", "--reason", "wrong answers reported"])
    kill_switch.main(["status"])
    assert "ON" in capsys.readouterr().out
    kill_switch.main(["off", "--by", "Marcus Adeyemi"])
    assert not kill_switch.is_on()
    log = kill_switch._log().read_text()
    assert "ON by Marcus Adeyemi: wrong answers reported" in log and "OFF by Marcus Adeyemi" in log
