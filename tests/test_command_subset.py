"""The command interpreter over a subset of the catalog: the agent's moves (2026-11-13)."""
import json

import pytest

import agent_common as ag
import command_common as cc


def test_the_prompt_lists_only_the_moves_offered():
    sub = {n: cc.G1_SKILLS[n] for n in ("arm_action", "stop")}
    text = cc.build_system_prompt("g1", sub)
    assert "- arm_action:" in text and "- stop:" in text and "- unknown:" in text
    assert "- walk:" not in text and "- damp:" not in text
    assert "- walk:" in cc.build_system_prompt("g1")              # the drive's voice: all of it


def test_a_move_outside_the_subset_is_unknown():
    sub = {"stop": cc.G1_SKILLS["stop"]}
    assert cc.normalize_intent({"skill": "damp"}, "g1", sub)["skill"] == "unknown"
    assert cc.normalize_intent({"skill": "stop"}, "g1", sub)["skill"] == "stop"
    assert cc.normalize_intent({"skill": "damp"}, "g1")["skill"] == "damp"


@pytest.mark.real_interpreter
def test_the_agent_asks_the_interpreter_over_its_moves_with_their_notes(monkeypatch):
    """Over the agent's moves only, each with the catalog's notes and examples (catalog() has
    none: without them "bajá los brazos" was "down"); a value of its own is asked once more."""
    asked = []
    answers = [{"skill": "arm_action", "params": {"action": "kiss"}},
               {"skill": "arm_action", "params": {"action": "left_kiss"}}]

    def stream_chat(payload, url, timeout):
        asked.append((payload["messages"][0]["content"], payload["messages"][1]["content"]))
        return json.dumps(answers[len(asked) - 1]), "", "stop", 1, 1
    monkeypatch.setattr(cc, "stream_chat", stream_chat)
    monkeypatch.setattr(cc, "model_supports_thinking", lambda model, url: False)
    got = ag._interpret("mandame un beso", ag.command_skills())
    assert got == {"skill": "arm_action", "params": {"action": "left_kiss"}}
    system, first = asked[0]
    assert first == "Command: mandame un beso"
    assert '"kiss" is not one of action\'s values' in asked[1][1]
    assert "- damp:" not in system and "- arm_action:" in system
    assert "release_arm" in system and "note: Map 'put/lower your arms down'" in system


@pytest.mark.real_interpreter
def test_a_good_answer_is_asked_once(monkeypatch):
    asked = []

    def stream_chat(payload, url, timeout):
        asked.append(1)
        return json.dumps({"skill": "arm_action", "params": {"action": "clap"}}), "", "stop", 1, 1
    monkeypatch.setattr(cc, "stream_chat", stream_chat)
    monkeypatch.setattr(cc, "model_supports_thinking", lambda model, url: False)
    assert ag._interpret("aplaudí", ag.command_skills())["params"] == {"action": "clap"}
    assert len(asked) == 1
