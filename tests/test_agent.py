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
def test_command_offers_every_drive_move_but_safe_mode_ones():
    """2026-11-13: "todas las opciones del drive, menos las del safe mode". The modes too."""
    names = set(ag.command_skills())
    assert {"wave_hand", "shake_hand", "arm_action", "sit", "stop", "stand_up", "squat",
            "start", "run", "run_waist", "walk_waist", "climb", "lie_up"} <= names
    assert not {"damp", "zero_torque", "dance", "squat_sdk", "set_fsm_id", "set_speed_mode",
                "switch_mode"} & names                          # safe mode's and the raw ones
    assert not {"walk", "turn", "unknown"} & names              # navigate walks, with Nav2
    assert "clap" in ag._tools_text()                            # the arm actions are listed


def test_the_moves_are_the_catalog_s_nothing_listed_by_hand(monkeypatch):
    """A skill added to the catalog reaches the agent with no change here, and one the
    catalog calls dangerous does not."""
    import command_common as cc
    real = cc.catalog

    def catalog(robot):
        cat = real(robot)
        skills = {**cat["skills"], "bow": {"label": "Bow", "params": {}},
                  "backflip": {"label": "Backflip", "params": {}}}
        return {**cat, "skills": skills, "dangerous": {*cat["dangerous"], "backflip"}}
    monkeypatch.setattr(cc, "catalog", catalog)
    assert "bow" in ag.command_skills() and "backflip" not in ag.command_skills()
    assert 'skill "bow" (Bow)' in ag._skills_text()


RELAY = {"transports": {"g1": {
    "mode": "relay", "allowed_skills": ["stop", "stand_up", "walk_waist", "run_waist", "arm_action",
                                        "damp", "wave_hand"],
    "allowed_values": {"arm_action": {"action": ["clap", "hug", "release_arm"]}}}}}


def test_the_moves_are_what_the_robot_s_link_delivers(monkeypatch):
    """The same answer the drive pad strikes its buttons out with: over the relay one walk
    mode of the waist pair, no sit, no high five."""
    monkeypatch.setitem(ag._SENDABLE, "value", ag._fetch_sendable(get=lambda: RELAY))
    moves = ag.command_skills()
    assert set(moves) == {"stop", "stand_up", "walk_waist", "run_waist", "arm_action", "wave_hand"}
    action = moves["arm_action"]["params"]["action"]
    assert action["values"] == ["release_arm", "hug", "clap"] or set(action["values"]) == {
        "clap", "hug", "release_arm"}
    assert "high_five" not in ag._skills_text() and "sit" not in ag._skills_text()
    assert all(set(vs) <= {"clap", "hug", "release_arm"} for vs in action["groups"].values())


def test_what_the_link_delivers_is_read_from_the_executor_and_kept_a_while(monkeypatch):
    assert ag._fetch_sendable(get=lambda: RELAY)["skills"] >= {"stop", "walk_waist"}
    dds = {"transports": {"g1": {"mode": "dds"}}}
    assert ag._fetch_sendable(get=lambda: dds) is None            # DDS: everything gets through

    def down():
        raise ConnectionError("executor down")
    assert ag._fetch_sendable(get=down) is None                   # the executor refuses, then
    asked = []
    monkeypatch.setattr(ag, "_fetch_sendable", lambda: asked.append(1) or {"skills": set(),
                                                                           "values": {}})
    monkeypatch.setitem(ag._SENDABLE, "at", None)
    ag.sendable(now=100.0)
    ag.sendable(now=100.0 + ag.SENDABLE_TTL_S - 1)
    assert len(asked) == 1
    ag.sendable(now=100.0 + ag.SENDABLE_TTL_S + 1)
    assert len(asked) == 2


def test_the_model_reads_each_value_s_label_and_the_mode_it_needs():
    """A label that says more than the name goes with it; the groups say the mode each needs.
    The labels and the groups are the catalog's, nothing written here."""
    text = ag._skills_text()
    assert "heart (Arm heart)" in text
    assert "hug (Hug)" not in text and "x_ray (X-ray)" not in text   # a label that adds nothing
    assert "needing Run mode: hug|high_five_clap|" in text
    assert "needing Walk or Run: release_arm|clap|" in text
    assert "(true = offer the hand (start), false = end the handshake)" in text


def test_a_value_that_cannot_be_sent_is_explained_never_swapped_for_the_default():
    """The interpreter puts the default in place of a bad value: arm_action "chocar" would
    have released the arms."""
    out = ag.tool_command({"skill": "arm_action", "params": {"action": "chocar"}}, lambda e: None,
                          threading.Event(), call=lambda body: pytest.fail("sent"))
    assert not out["ok"] and "'chocar' is not a value" in out["error"] and "clap" in out["error"]


def test_a_safe_mode_move_is_refused_with_why(monkeypatch):
    out = ag.tool_command({"skill": "damp"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"))
    assert not out["ok"] and "safe mode" in out["error"]
    monkeypatch.setitem(ag._SENDABLE, "value", ag._fetch_sendable(get=lambda: RELAY))
    out = ag.tool_command({"skill": "high_five"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"), interpret=fake_interpreter({}))
    assert not out["ok"] and "cannot be done now" in out["error"]
    out = ag.tool_command({"skill": "arm_action", "params": {"action": "high_five"}},
                          lambda e: None, threading.Event(), call=lambda body: pytest.fail("sent"))
    assert not out["ok"] and "can be sent now" in out["error"]


def test_command_sends_a_checked_skill_with_safe_mode():
    sent = []

    def call(body):
        sent.append(body)
        return {"ok": True}
    events, emit = collect()
    out = ag.tool_command({"skill": "arm_action", "params": {"action": "clap", "bogus": 1}},
                          emit, threading.Event(), call=call)
    assert out == {"ok": True, "skill": "arm_action", "action": "clap"}   # what it did
    assert sent == [{"skill": "arm_action", "params": {"action": "clap"}}]
    assert events[0]["step"] == "arm_action"


@pytest.mark.parametrize("skill, why", [("damp", "safe mode"), ("zero_torque", "safe mode"),
                                        ("dance", "safe mode"), ("walk", "navigate"),
                                        ("turn", "navigate"), ("", "which move")])
def test_command_refuses_what_it_does_not_offer_without_calling_the_robot(skill, why):
    """A move of the catalog it may not do is refused with why, and never asked about."""
    sent = []
    out = ag.tool_command({"skill": skill}, collect()[1], threading.Event(),
                          call=lambda b: sent.append(b) or {"ok": True},
                          interpret=lambda w, m: pytest.fail("asked the interpreter"))
    assert out["ok"] is False and why in out["error"] and not sent


def fake_interpreter(answers):
    """The command interpreter, by the words it gets; records the moves it was offered."""
    seen = []

    def interpret(words, moves):
        seen.append((words, set(moves)))
        return answers.get(words, {"skill": "unknown", "params": {}})
    interpret.seen = seen
    return interpret


def test_a_move_in_words_is_the_interpreter_s_to_understand():
    """"Bajá los brazos" was a squat and "chocá los cinco" a look when the agent picked the
    move alone (2026-11-13): the words go to the drive's own interpreter, over the moves that
    can be done now only, and what it picked is what is sent."""
    sent = []
    interpret = fake_interpreter({"bajá los brazos": {"skill": "arm_action",
                                                     "params": {"action": "release_arm"}}})
    events, emit = collect()
    out = ag.tool_command({"move": "bajá los brazos"}, emit, threading.Event(),
                          call=lambda b: sent.append(b) or {"ok": True}, interpret=interpret)
    assert out == {"ok": True, "skill": "arm_action", "action": "release_arm"}
    assert sent == [{"skill": "arm_action", "params": {"action": "release_arm"}}]
    assert [e["step"] for e in events] == ["understanding", "arm_action"]
    offered = interpret.seen[0][1]
    assert "damp" not in offered and "walk" not in offered and "clap" not in offered
    assert offered == set(ag.command_skills())


def test_a_move_the_interpreter_does_not_find_is_said_with_the_moves_there_are():
    out = ag.tool_command({"move": "volá"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"), interpret=fake_interpreter({}))
    assert not out["ok"] and "none of the moves that can be done now" in out["error"]
    assert "arm_action" in out["error"]


def test_a_bad_value_from_the_interpreter_is_explained_not_swapped():
    """"Mandame un beso" -> arm_action "kiss": the interpreter would send release_arm."""
    interpret = fake_interpreter({"mandame un beso": {"skill": "arm_action",
                                                     "params": {"action": "kiss"}}})
    out = ag.tool_command({"move": "mandame un beso"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"), interpret=interpret)
    assert not out["ok"] and "'kiss' is not a value" in out["error"]
    assert "left_kiss" in out["error"] and "two_hand_kiss" in out["error"]


def test_the_interpreter_down_is_said_never_guessed():
    def down(words, moves):
        raise ConnectionError("llama-server down")
    out = ag.tool_command({"move": "aplaudí"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"), interpret=down)
    assert not out["ok"] and "could not understand the move" in out["error"]


def test_command_failures_reach_the_model():
    out = ag.tool_command({"skill": "wave_hand"}, collect()[1], threading.Event(),
                          call=lambda b: {"ok": False, "error": "the robot is not standing"})
    assert out == {"ok": False, "skill": "wave_hand", "turn": False,
                   "error": "the robot is not standing"}


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
        assert "none of it is to be carried out now" in told
        assert "asks for the same thing again is done again" in told
    assert f"Instruction now (not done yet): {now}\n(Write understood and the plan in " in llm.seen[0]


def test_without_a_conversation_no_earlier_turns_are_told():
    _events, emit = collect()
    llm = scripted({"understood": "hi", "plan": []}, {"reply": "hola"})
    ag.run("hola", emit, llm=llm, tools={}, history=[])
    assert "Earlier in this conversation" not in llm.seen[0]
    assert "Instruction now (not done yet): hola\n(Write understood and the plan in Spanish" in llm.seen[0]
    assert "Instruction (now, not done yet): hola" in llm.seen[1]


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


def test_a_move_named_by_its_parameter_goes_to_the_interpreter():
    """Seen on the robot: skill "clap" was refused with only the list of moves, and the model
    gave up. A name that is no move is words for the interpreter, which knows clap is an arm
    action; when it finds nothing, the error says where the value lives."""
    sent = []
    interpret = fake_interpreter({"clap": {"skill": "arm_action", "params": {"action": "clap"}}})
    out = ag.tool_command({"skill": "clap"}, lambda e: None, threading.Event(),
                          call=lambda b: sent.append(b) or {"ok": True}, interpret=interpret)
    assert out["ok"] and sent == [{"skill": "arm_action", "params": {"action": "clap"}}]
    out = ag.tool_command({"skill": "hug"}, lambda e: None, threading.Event(),
                          call=lambda body: pytest.fail("sent"), interpret=fake_interpreter({}))
    assert not out["ok"] and "'arm_action hug'" in out["error"]


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


# --------------------------------------------------------------------------- #
# The drive's command interpreter reads the instruction first (2026-11-13: 29 -> 45 of 48)
# --------------------------------------------------------------------------- #
def reads(answer):
    return lambda words, moves: answer


@pytest.mark.parametrize("answer, says", [
    ({"skill": "arm_action", "params": {"action": "release_arm"}},
     "the move 'arm_action release_arm'"),
    ({"skill": "arm_action", "params": {"action": "high_five_clap"}},
     "the move 'arm_action high_five_clap'"),
    ({"skill": "arm_action", "params": {"action": "heart"}}, "heart (Arm heart)"),
    ({"skill": "lie_up", "params": {}}, "the move 'lie_up'"),
    ({"skill": "damp", "params": {}}, "Damping: safe mode holds it back, so say you cannot"),
    ({"skill": "dance", "params": {"name": "Spin_discs"}}, "Dance: safe mode holds it back"),
    ({"skill": "walk", "params": {}}, "walking somewhere is the navigate tool's"),
    ({"skill": "turn", "params": {}}, "turning in place is the navigate tool's"),
])
def test_the_interpreter_s_reading_is_told_to_the_agent(answer, says):
    assert says in ag.move_hint("x", interpret=reads(answer))


def test_no_reading_no_hint():
    assert ag.move_hint("¿qué ves?", interpret=reads({"skill": "unknown", "params": {}})) == ""
    assert ag.move_hint("x", interpret=reads({})) == ""
    assert ag.move_hint("x", interpret=reads({"skill": "arm_action",
                                               "params": {"action": "kiss"}})) == ""

    def down(words, moves):
        raise ConnectionError("llama-server down")
    assert ag.move_hint("aplaudí", interpret=down) == ""          # a hint, never a failure


def test_a_reading_the_link_cannot_send_is_said(monkeypatch):
    monkeypatch.setitem(ag._SENDABLE, "value", ag._fetch_sendable(get=lambda: RELAY))
    hint = ag.move_hint("x", interpret=reads({"skill": "arm_action",
                                              "params": {"action": "high_five"}}))
    assert "high_five: the robot's link cannot send it" in hint
    assert "link cannot send it" in ag.move_hint("x", interpret=reads({"skill": "sit"}))


def test_the_hint_is_asked_over_the_whole_catalog():
    """Safe mode's moves included: that is how the agent learns "relajate" is held back."""
    seen = []
    ag.move_hint("relajate", interpret=lambda w, m: seen.append(set(m)) or {})
    assert {"damp", "dance", "walk", "arm_action"} <= seen[0]
    assert not {"set_fsm_id", "unknown"} & seen[0]


def test_the_hint_goes_right_under_the_instruction(monkeypatch):
    monkeypatch.setattr(ag, "move_hint", lambda instruction: "READS: the move 'lie_up'.\n\n")
    llm = scripted({"understood": "u", "plan": []}, {"reply": "Listo."})
    ag.run("levantate del piso", lambda e: None, threading.Event(), llm=llm, tools={})
    assert "Instruction now (not done yet): levantate del piso\nREADS: the move 'lie_up'.\n(Write" in llm.seen[0]
    assert "Instruction (now, not done yet): levantate del piso\nREADS: the move 'lie_up'." in llm.seen[1]


# --------------------------------------------------------------------------- #
# Which way, in words (2026-11-13: left came out as negative numbers, three times out of three)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("args, sent", [
    ({"turn": "left", "degrees": 90}, (0.0, 0.0, round(math.pi / 2, 4))),
    ({"turn": "right"}, (0.0, 0.0, -round(math.pi / 2, 4))),          # a quarter turn
    ({"go": "back"}, (-1.0, 0.0, 0.0)),                                # a metre
    ({"metres": 0.5}, (0.5, 0.0, 0.0)),                                # forward
    ({"turn": "right", "degrees": 45}, (0.0, 0.0, -round(math.pi / 4, 4))),
    ({"go": "left", "metres": 1}, (0.0, 1.0, 0.0)),
    ({"go": "right", "metres": 0.5}, (0.0, -0.5, 0.0)),
    ({"go": "back", "metres": 1}, (-1.0, 0.0, 0.0)),
    ({"go": "forward", "metres": 2, "turn": "right", "degrees": 90},
     (2.0, 0.0, -round(math.pi / 2, 4))),
    ({"go": "left", "metres": -1}, (0.0, 1.0, 0.0)),               # the word decides, not a sign
    ({"turn": "left", "degrees": -90}, (0.0, 0.0, round(math.pi / 2, 4))),
    ({"forward_m": 1, "turn_deg": -30}, (1.0, 0.0, -round(math.pi / 6, 4))),   # the old way
])
def test_the_way_is_a_word_and_the_sign_is_the_code_s(args, sent):
    call = fake_walk_executor([{"state": "success", "error": {"pos_m": 0.0, "yaw_deg": 0.0}}])
    assert ag.tool_navigate(args, collect()[1], threading.Event(), call=call, poll_s=0)["ok"]
    body = call.sent[0]
    assert (body["dx"], body["dy"], body["dyaw"]) == sent


@pytest.mark.parametrize("args", [{"turn": "around"}, {"turn": "around", "degrees": 180},
                                  {"turn": "left", "degrees": 180},
                                  {"turn": "right", "degrees": 200}])
def test_half_a_turn_is_never_rounded_past_what_the_relay_takes(args):
    """"Date media vuelta": 180 deg rounded to 3.1416 rad, over pi, and the relay refused it."""
    call = fake_walk_executor([{"state": "success", "error": {"pos_m": 0.0, "yaw_deg": 0.0}}])
    ag.tool_navigate(args, collect()[1], threading.Event(), call=call, poll_s=0)
    assert 3.14 < abs(call.sent[0]["dyaw"]) <= math.pi


@pytest.mark.parametrize("args, why", [
    ({"go": "izquierda", "metres": 1}, "go must be one of"),
    ({"turn": "up", "degrees": 90}, "turn must be one of"),
    ({}, "nowhere to go"),
    ({"target": "adelante", "metres": 0.5}, "not both"),
])
def test_a_way_it_cannot_take_is_said_without_moving(args, why):
    call = fake_walk_executor([])
    out = ag.tool_navigate(args, collect()[1], threading.Event(), call=call, poll_s=0)
    assert not out["ok"] and why in out["error"] and not call.sent


def test_a_failed_walk_says_how_far_it_really_went_and_turned():
    """The 45 deg turn failed 9 deg off and the agent turned 45 again: the result now says it
    did turn 36."""
    start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    pose = {"x": 0.2, "y": 0.0, "yaw": -math.radians(36)}
    call = fake_walk_executor([{"state": "failure", "start": start, "pose": pose,
                                "error": {"pos_m": 0.2, "yaw_deg": 9.0},
                                "reason": "not there after 43 s"}])
    out = ag.tool_navigate({"turn": "right", "degrees": 45}, collect()[1], threading.Event(),
                           call=call, poll_s=0)
    assert not out["ok"] and out["turned"] == "36 deg right" and out["went_m"] == 0.2
    assert "never do it again whole" in ag._tools_text()


def test_a_reading_the_link_cannot_send_is_read_again_among_what_it_can(monkeypatch):
    """"Chocá los cinco" read as the high five, which the relay leaves out; among what it sends,
    it is the app's high five clap."""
    monkeypatch.setitem(ag._SENDABLE, "value", {
        "skills": {"arm_action", "stop"},
        "values": {"arm_action": {"action": ["clap", "high_five_clap", "hug"]}}})
    asked = []

    def interpret(words, moves):
        asked.append(set(moves["arm_action"]["params"]["action"]["values"]))
        return {"skill": "arm_action",
                "params": {"action": "high_five" if len(asked) == 1 else "high_five_clap"}}
    hint = ag.move_hint("chocá los cinco", interpret=interpret)
    assert "the move 'arm_action high_five_clap'" in hint
    assert "high_five" in asked[0] and "high_five" not in asked[1]


def test_safe_mode_s_moves_are_never_read_again():
    asked = []
    hint = ag.move_hint("relajate", interpret=lambda w, m: asked.append(1) or {"skill": "damp"})
    assert "safe mode holds it back" in hint and len(asked) == 1


def test_the_same_words_again_are_said_to_be_asked_again():
    """A second "Aplaudí" copied the first one's answer and clapped no more (2026-11-13, 6/6)."""
    clap = {"question": "Aplaudí", "answer": "¡Listo!", "done": ["command(...) -> ok"]}
    assert ag.asked_before("aplaudi!", [clap]) and ag.asked_before("  APLAUDÍ ", [clap])
    assert not ag.asked_before("aplaudí de nuevo", [clap]) and not ag.asked_before("x", [])
    llm = scripted({"understood": "u", "plan": []}, {"reply": "Listo."})
    ag.run("Aplaudí", lambda e: None, threading.Event(), llm=llm, tools={}, history=[clap])
    assert "asked again now: do it again, it is not done yet" in llm.seen[0]
    assert "asked again now: do it again" in llm.seen[1]
    llm = scripted({"understood": "u", "plan": []}, {"reply": "Listo."})
    ag.run("Saludá", lambda e: None, threading.Event(), llm=llm, tools={}, history=[clap])
    assert "asked again" not in llm.seen[0]
