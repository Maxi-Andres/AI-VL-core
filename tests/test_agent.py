"""agent_common — the /agent tab's loop: understood, plan, tools, reply, all emitted live.

Every test names the defect it catches. No model and no robot: the LLM is a script of answers,
the tools are fakes. In Spanish AND English, like every text of the assistant.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_common as ag


def scripted(*answers):
    """An llm(system, user) that gives these answers in order and records what it was told."""
    seen = []

    def llm(system, user):
        seen.append(user)
        return answers[len(seen) - 1] if len(seen) <= len(answers) else {}
    llm.seen = seen
    return llm


def collect():
    events = []
    return events, events.append


def fake_pick(result):
    def tool(args, emit, stop):
        emit({"type": "tool_event", "tool": "pick", "step": "looking",
              "detail": f"finding '{args['object']}'"})
        return result
    return tool


@pytest.mark.parametrize("instruction, understood, reply", [
    ("agarrá la botella", "levantar la botella", "Listo, la levanté."),
    ("pick up the bottle", "pick up the bottle", "Done, I lifted it."),
])
def test_an_instruction_is_understood_planned_done_and_answered(instruction, understood, reply):
    events, emit = collect()
    llm = scripted({"understood": understood, "plan": ["ubicar", "agarrar"], "needs_robot": True},
                   {"thought": "pick it", "tool": "pick", "args": {"object": "the bottle"}},
                   {"thought": "it worked", "reply": reply})
    out = ag.run(instruction, emit, llm=llm,
                 tools={"pick": fake_pick({"ok": True, "state": "success", "held": True})})
    assert out == reply
    kinds = [e["type"] for e in events]
    assert kinds == ["heard", "understood", "plan", "thought", "tool", "tool_event",
                     "tool_result", "thought", "reply"]
    assert events[1]["text"] == understood


def test_a_failed_tool_is_told_to_the_model_not_hidden():
    """The defect: the robot saying "done" after a grasp on air."""
    _events, emit = collect()
    llm = scripted({"understood": "x", "plan": []},
                   {"tool": "pick", "args": {"object": "the bottle"}},
                   {"reply": "No pude agarrarla."})
    ag.run("agarrá la botella", emit, llm=llm,
           tools={"pick": fake_pick({"ok": False, "state": "failure", "held": False})})
    assert '"held": false' in llm.seen[2], "the model saw that nothing was held"


def test_a_question_needs_no_robot():
    events, emit = collect()
    llm = scripted({"understood": "what is on the table", "plan": [], "needs_robot": False},
                   {"tool": "look", "args": {"question": "what is on the table?"}},
                   {"reply": "A bottle and a cup."})
    ag.run("what's on the table?", emit, llm=llm,
           tools={"look": lambda a, e, s: {"ok": True, "answer": "a bottle and a cup"}})
    assert [e["type"] for e in events if e["type"] in ("tool", "reply")] == ["tool", "reply"]


def test_an_unknown_tool_is_an_error_and_the_loop_goes_on():
    """The defect: a made-up tool name crashing the loop (or silently ignored)."""
    events, emit = collect()
    llm = scripted({"understood": "x", "plan": []}, {"tool": "fly", "args": {}},
                   {"reply": "I can't fly."})
    assert ag.run("fly", emit, llm=llm, tools={}) == "I can't fly."
    assert any(e["type"] == "error" and "fly" in e["text"] for e in events)


def test_stop_ends_the_run_and_the_model_says_so():
    _events, emit = collect()
    stop = threading.Event()

    def pick(args, emit_, stop_):
        stop_.set()                                   # STOP pressed while it ran
        return {"ok": False, "state": "cancelled"}
    llm = scripted({"understood": "x", "plan": []}, {"tool": "pick", "args": {"object": "b"}},
                   {"reply": "Paré, como pediste."})
    out = ag.run("agarrá la botella", emit, stop=stop, llm=llm, tools={"pick": pick})
    assert out == "Paré, como pediste."
    assert "STOP" in llm.seen[-1]


def test_a_model_that_says_nothing_still_gets_a_bilingual_last_word():
    _events, emit = collect()
    out = ag.run("hola", emit, llm=lambda s, u: {}, tools={}, max_steps=2)
    assert out == "No pude terminarlo. / I could not finish it."


class FakeAction:
    def __init__(self):
        self.events = [{"step": "looking", "detail": "a"}, {"step": "found", "detail": "b"},
                       {"step": "done", "detail": "success"}]
        self.state, self.cancelled, self.calls = "running", False, 0

        class T:
            @staticmethod
            def is_alive():
                return self.state == "running"
        self.thread = T()

    def snapshot(self, since=0):
        self.calls += 1
        if self.calls >= 2:
            self.state = "success"
        n = min(len(self.events), self.calls + 1)
        return {"state": self.state, "result": {"held": True}, "events": self.events[since:n]}

    def cancel(self):
        self.cancelled = True


def test_pick_streams_every_step_of_the_action_once():
    """The defect: the trace missing the action's steps, or repeating them on every poll."""
    events, emit = collect()
    act = FakeAction()
    out = ag.tool_pick({"object": "the bottle"}, emit, threading.Event(),
                       start=lambda o: act, poll_s=0)
    assert out["ok"] and out["held"]
    assert [e["step"] for e in events] == ["looking", "found", "done"]


def test_place_asks_for_the_target_and_reports_whether_it_was_placed():
    """The defect: a place with nowhere named starting a pick, or its result losing "placed"."""
    events, emit = collect()
    assert not ag.tool_place({"object": "the bottle"}, emit, threading.Event(),
                             start=lambda o, **kw: None)["ok"]
    act, asked = FakeAction(), {}

    def start(obj, **kw):
        asked.update(kw, object=obj)
        return act
    real = act.snapshot
    act.snapshot = lambda since=0: {**real(since), "result": {"held": True, "placed": True}}
    out = ag.tool_place({"object": "the bottle", "into": "the box"}, emit, threading.Event(),
                        start=start, poll_s=0)
    assert asked == {"object": "the bottle", "into": "the box"}
    assert out["ok"] and out["placed"] and events[0]["tool"] == "place"


def test_a_model_that_stops_answering_ends_the_run_cleanly():
    """The defect (2026-11-03): llama-server went down mid-run and the exception escaped the
    loop — the stream would end with no reply and the tab would wait forever."""
    events, emit = collect()

    def down(system, user):
        raise ConnectionError("port 8081 refused")
    out = ag.run("agarrá la botella", emit, llm=down, tools={})
    assert out.startswith("El modelo de lenguaje no responde")
    assert [e["type"] for e in events][-2:] == ["error", "reply"]


# --------------------------------------------------------------------------- #
# walk — the robot walks to a point; the loop is on the robot, this only follows it
# --------------------------------------------------------------------------- #
def fake_walk_executor(states, refuse=None):
    """A /walk/control that answers goto, then `states` to successive status calls."""
    sent = []

    def call(body):
        sent.append(body)
        if body["op"] == "goto":
            return {"ok": False, "error": refuse} if refuse else {"ok": True, "walk": states[0]}
        if body["op"] == "cancel":
            return {"ok": True}
        k = sum(1 for b in sent if b["op"] == "status") - 1
        return {"ok": True, "walk": states[min(k, len(states) - 1)]}
    call.sent = sent
    return call


def test_walk_sends_metres_and_radians_and_reports_how_far_off_it_stopped():
    call = fake_walk_executor([
        {"state": "running", "phase": "translate"},
        {"state": "running", "phase": "turn"},
        {"state": "success", "phase": "done", "error": {"pos_m": 0.04, "yaw_deg": -2.0}},
    ])
    events, emit = collect()
    out = ag.tool_walk({"forward_m": 2}, emit, threading.Event(), call=call, poll_s=0)
    assert call.sent[0] == {"op": "goto", "dx": 2.0, "dy": 0.0, "dyaw": 0.0}
    assert out == {"ok": True, "state": "success", "off_cm": 4, "off_deg": 2.0}
    assert [e["step"] for e in events] == ["translate", "turn", "success"]


def test_turn_is_degrees_left_sent_as_radians():
    import math
    call = fake_walk_executor([{"state": "success", "error": {"pos_m": 0.0, "yaw_deg": 1.0}}])
    out = ag.tool_turn({"degrees": -90}, collect()[1], threading.Event(), call=call, poll_s=0)
    assert call.sent[0] == {"op": "goto", "dx": 0.0, "dy": 0.0, "dyaw": math.radians(-90)}
    assert out["ok"] is True


def test_walk_stop_cancels_it_on_the_robot():
    call = fake_walk_executor([{"state": "running", "phase": "translate"},
                               {"state": "cancelled", "reason": "cancel from the agent"}])
    stop = threading.Event()
    stop.set()
    out = ag.tool_walk({"forward_m": 1}, collect()[1], stop, call=call, poll_s=0)
    assert {"op": "cancel"} in call.sent
    assert out["ok"] is False and out["state"] == "cancelled"


@pytest.mark.parametrize("args, why", [
    ({"forward_m": "lejos"}, "numbers"),
    ({"forward_m": 9}, "at most 3 m per request"),
])
def test_walk_refusals_reach_the_model(args, why):
    call = fake_walk_executor([], refuse="at most 3 m per request")
    out = ag.tool_walk(args, collect()[1], threading.Event(), call=call, poll_s=0)
    assert out["ok"] is False and why in out["error"]
