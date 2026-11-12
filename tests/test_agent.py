"""agent_common — the /agent tab's loop: understood, plan, tools, reply, all emitted live.

Every test names the defect it catches. No model and no robot: the LLM is a script of answers,
the tools are fakes. In Spanish AND English, like every text of the assistant.
"""
from __future__ import annotations

import math
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
# navigate — the robot walks to a point; the loop is on the robot, this only follows it
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


def test_navigate_by_metres_sends_metres_and_radians_and_reports_how_far_off_it_stopped():
    call = fake_walk_executor([
        {"state": "running", "phase": "translate"},
        {"state": "running", "phase": "turn"},
        {"state": "success", "phase": "done", "error": {"pos_m": 0.04, "yaw_deg": -2.0}},
    ])
    events, emit = collect()
    out = ag.tool_navigate({"forward_m": 2}, emit, threading.Event(), call=call, poll_s=0)
    assert call.sent[0] == {"op": "goto", "dx": 2.0, "dy": 0.0, "dyaw": 0.0}
    assert out == {"ok": True, "state": "success", "off_cm": 4, "off_deg": 2.0}
    assert [e["step"] for e in events] == ["translate", "turn", "success"]


def test_navigate_turn_only_is_degrees_left_sent_as_radians():
    call = fake_walk_executor([{"state": "success", "error": {"pos_m": 0.0, "yaw_deg": 1.0}}])
    out = ag.tool_navigate({"turn_deg": -90}, collect()[1], threading.Event(), call=call,
                           poll_s=0)
    assert call.sent[0] == {"op": "goto", "dx": 0.0, "dy": 0.0,
                            "dyaw": round(math.radians(-90), 4)}
    assert out["ok"] is True


def test_navigate_says_when_it_is_blocked_and_when_it_goes_on():
    call = fake_walk_executor([
        {"state": "running", "phase": "translate"},
        {"state": "running", "phase": "translate",
         "blocked": {"reason": "something is in the way: waiting for it to clear"}},
        {"state": "running", "phase": "translate", "blocked": None},
        {"state": "success", "error": {"pos_m": 0.2, "yaw_deg": 3.0}},
    ])
    events, emit = collect()
    ag.tool_navigate({"forward_m": 3}, emit, threading.Event(), call=call, poll_s=0)
    assert [e["step"] for e in events] == ["translate", "blocked", "going on", "success"]


def test_navigate_stop_cancels_it_on_the_robot():
    call = fake_walk_executor([{"state": "running", "phase": "translate"},
                               {"state": "cancelled", "reason": "cancel from the agent"}])
    stop = threading.Event()
    stop.set()
    out = ag.tool_navigate({"forward_m": 1}, collect()[1], stop, call=call, poll_s=0)
    assert {"op": "cancel"} in call.sent
    assert out["ok"] is False and out["state"] == "cancelled"


@pytest.mark.parametrize("args, why", [
    ({"forward_m": "lejos"}, "numbers"),
    ({}, "nowhere to go"),
    ({"forward_m": 12}, "at most 10 m"),
])
def test_navigate_refusals_reach_the_model(args, why):
    call = fake_walk_executor([], refuse="at most 10 m per walk")
    out = ag.tool_navigate(args, collect()[1], threading.Event(), call=call, poll_s=0)
    assert out["ok"] is False and why in out["error"]


# navigate to something it sees: found and measured (locate_seen), then walked to
def test_goal_toward_stops_short_of_the_thing_facing_it():
    fwd, left, turn = ag.goal_toward(2.6, 0.0)
    assert (fwd, left, turn) == pytest.approx((2.0, 0.0, 0.0))
    fwd, left, turn = ag.goal_toward(2.0, 2.0)                  # 45 deg to the left
    assert math.hypot(fwd, left) == pytest.approx(math.hypot(2, 2) - ag.NAV_STANDOFF_M)
    assert turn == pytest.approx(45.0) and fwd > 0 and left > 0
    assert ag.goal_toward(0.7, -0.1)[:2] == (0.0, 0.0)          # close already: only turn


def test_navigate_to_a_target_walks_to_where_it_was_seen():
    call = fake_walk_executor([{"state": "success", "error": {"pos_m": 0.1, "yaw_deg": 2.0}}])
    events, emit = collect()
    out = ag.tool_navigate({"target": "la silla"}, emit, threading.Event(), call=call, poll_s=0,
                           locate=lambda phrase: ((3.0, 0.0), "yolo chair, chosen by the vlm"))
    assert call.sent[0] == {"op": "goto", "dx": 2.4, "dy": 0.0, "dyaw": 0.0}
    assert out["ok"] is True
    assert [e["step"] for e in events][:2] == ["looking", "seen"]
    assert "3.0 m away" in events[1]["detail"]


def test_navigate_to_a_target_not_in_sight_tells_the_model_and_does_not_move():
    call = fake_walk_executor([])
    out = ag.tool_navigate({"target": "la puerta"}, collect()[1], threading.Event(), call=call,
                           poll_s=0, locate=lambda phrase: (None, "I do not see la puerta"))
    assert out == {"ok": False, "error": "I do not see la puerta"} and not call.sent


def test_navigate_to_a_target_right_in_front_does_not_walk():
    call = fake_walk_executor([])
    out = ag.tool_navigate({"target": "vos"}, collect()[1], threading.Event(), call=call,
                           poll_s=0, locate=lambda phrase: ((0.65, 0.02), "yolo person"))
    assert out["ok"] is True and out["state"] == "already there" and not call.sent


# command — one of the robot's built-in moves, checked like the command interpreter's
def test_command_offers_the_safe_moves_and_not_walking_or_dangerous_ones():
    names = ag.command_skills()
    assert {"wave_hand", "shake_hand", "arm_action", "sit", "stop"} <= set(names)
    assert not {"walk", "turn", "damp", "zero_torque", "dance", "set_fsm_id", "unknown",
                "run", "start"} & set(names)                       # the mode is the operator's
    assert "clap" in ag._tools_text()                            # the arm actions are listed


def test_command_sends_a_checked_skill_with_safe_mode():
    sent = []

    def call(body):
        sent.append(body)
        return {"ok": True}
    events, emit = collect()
    out = ag.tool_command({"skill": "arm_action", "params": {"action": "clap", "bogus": 1}},
                          emit, threading.Event(), call=call)
    assert out == {"ok": True, "skill": "arm_action"}
    assert sent == [{"skill": "arm_action", "params": {"action": "clap"}}]
    assert events[0]["step"] == "arm_action"


@pytest.mark.parametrize("skill", ["damp", "walk", "volar", ""])
def test_command_refuses_what_it_does_not_offer_without_calling_the_robot(skill):
    sent = []
    out = ag.tool_command({"skill": skill}, collect()[1], threading.Event(),
                          call=lambda b: sent.append(b) or {"ok": True})
    assert out["ok"] is False and "no move" in out["error"] and not sent


def test_command_failures_reach_the_model():
    out = ag.tool_command({"skill": "wave_hand"}, collect()[1], threading.Event(),
                          call=lambda b: {"ok": False, "error": "the robot is not standing"})
    assert out == {"ok": False, "skill": "wave_hand", "error": "the robot is not standing"}


# --------------------------------------------------------------------------- #
# The conversation so far (2026-11-12): "ahora dejala en la mesa" after "agarrá la botella"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("earlier, now", [
    ({"question": "agarrá la botella", "answer": "Listo, la tengo.",
      "done": ["pick(object=botella) -> ok"]}, "ahora dejala en la mesa"),
    ({"question": "pick up the bottle", "answer": "Done, I have it.",
      "done": ["pick(object=bottle) -> ok"]}, "now put it on the table"),
])
def test_the_conversation_so_far_goes_with_every_call_to_the_model(earlier, now):
    """Defect: each instruction ran alone, so "it" after "pick up the bottle" meant nothing."""
    _events, emit = collect()
    llm = scripted({"understood": "place it", "plan": ["place"], "needs_robot": True},
                   {"thought": "place", "tool": "place", "args": {"target": "table"}},
                   {"reply": "ok"})
    ag.run(now, emit, llm=llm, tools={"place": lambda a, e, s: {"ok": True}}, history=[earlier])
    for told in llm.seen:                    # the plan, and every step
        assert earlier["question"] in told and earlier["answer"] in told
        assert earlier["done"][0] in told
        assert "do NOT do any of it again" in told
    assert llm.seen[0].endswith(f"Instruction now: {now}")


def test_without_a_conversation_no_earlier_turns_are_told():
    _events, emit = collect()
    llm = scripted({"understood": "hi", "plan": []}, {"reply": "hola"})
    ag.run("hola", emit, llm=llm, tools={}, history=[])
    assert "Earlier in this conversation" not in llm.seen[0]
    assert llm.seen[0].endswith("Instruction now: hola")
    assert "Instruction: hola" in llm.seen[1]


def test_only_the_newest_turns_that_fit_go_and_empty_ones_are_skipped():
    """Defect it prevents: a long conversation overflowing the model's 4096-token window."""
    turns = [{"question": f"q{i} " + "x" * 300, "answer": f"a{i}"} for i in range(40)]
    turns.append({"question": "stopped before anything", "answer": "", "done": []})
    text = ag.history_text(turns)
    assert len(text) <= ag.HISTORY_CHARS + 200
    assert "a39" in text and "a0\n" not in text
    assert "stopped before anything" not in text
    assert text.index("q38") < text.index("q39")             # oldest first
    assert ag.history_text([]) == ""


# --------------------------------------------------------------------------- #
# What the vision chat has, the agent has (2026-11-12)
# --------------------------------------------------------------------------- #
def test_the_agent_is_told_the_time_the_robots_notes_and_its_state():
    _events, emit = collect()
    llm = scripted({"understood": "x", "plan": []}, {"reply": "En el cajón."})
    ag.run("¿dónde está la llave?", emit, llm=llm, tools={},
           notes=["la llave está en el cajón"], state={"battery": {"percent": 41, "charging": False}})
    for told in llm.seen:
        assert "Now it is" in told
        assert "la llave está en el cajón" in told
        assert "41 %" in told


@pytest.mark.parametrize("instruction, lang", [("aplaudí", "Spanish"), ("clap your hands", "English")])
def test_every_step_is_told_the_language_to_reply_in(instruction, lang):
    """Seen on the robot: "Aplaudí" answered "I can't clap right now"."""
    _events, emit = collect()
    llm = scripted({"understood": "x", "plan": []}, {"reply": "ok"})
    ag.run(instruction, emit, llm=llm, tools={})
    assert llm.seen[1].endswith(f"(Any reply must be in {lang}.)")


def test_a_move_named_by_its_parameter_is_explained_not_guessed():
    """Seen on the robot: skill "clap" was refused with only the list of moves, and the model
    gave up. Nothing is mapped by hand: the error says where "clap" is, the model calls it."""
    out = ag.tool_command({"skill": "clap"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"))
    assert not out["ok"]
    assert '{"skill": "arm_action", "params": {"action": "clap"}}' in out["error"]
    assert "no move 'fly'" in ag.tool_command({"skill": "fly"}, lambda e: None, threading.Event(),
                                              call=lambda body: pytest.fail("sent"))["error"]


def test_moves_are_offered_as_calls():
    assert 'skill "arm_action" (Arm action) with params {"action": one of' in ag._skills_text()


def test_search_tells_the_model_the_results_with_their_sites_and_today():
    events, emit = collect()
    found = [{"title": "Clima en Buenos Aires", "url": "https://www.smn.gob.ar/x",
              "content": "Hoy 24 grados, soleado", "date": "2026-11-12"}]
    out = ag.tool_search({"query": "clima buenos aires"}, emit, threading.Event(),
                         find=lambda q: found)
    assert out["ok"] and "24 grados" in out["results"] and "smn.gob.ar" in out["results"]
    assert out["today"]
    assert [e["step"] for e in events] == ["searching", "found"]


def test_a_failed_or_empty_search_says_so_and_not_to_guess():
    def down(q):
        raise ConnectionError("searxng down")
    out = ag.tool_search({"query": "dólar hoy"}, lambda e: None, threading.Event(), find=down)
    assert not out["ok"] and "do not guess" in out["error"]
    out = ag.tool_search({"query": "zzz"}, lambda e: None, threading.Event(), find=lambda q: [])
    assert "nothing found" in out["results"]
    assert not ag.tool_search({}, lambda e: None, threading.Event(), find=lambda q: [])["ok"]


def test_remember_hands_the_note_and_picture_to_the_gateway():
    events, emit = collect()
    out = ag.tool_remember({"note": "la llave está en el cajón", "picture": True}, emit,
                           threading.Event(), frame=lambda: "QUJD")
    assert out == {"ok": True, "kept": "a note and a picture"}
    assert events == [{"type": "keep", "note": "la llave está en el cajón", "image": "QUJD"}]
    events.clear()
    assert ag.tool_remember({"note": "el código es 4455"}, emit, threading.Event())["ok"]
    assert events == [{"type": "keep", "note": "el código es 4455"}]
    assert not ag.tool_remember({}, emit, threading.Event())["ok"]
    assert not ag.tool_remember({"picture": True}, emit, threading.Event(), frame=lambda: "")["ok"]


def test_the_model_is_offered_search_and_remember():
    text = ag._tools_text()
    assert "- search(" in text and "- remember(" in text
