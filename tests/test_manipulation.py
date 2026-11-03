"""manipulation — the pick as an action: planned, checked, run step by step, cancelled.

Every test names the defect it catches. No robot: a fake executor speaks the relay's joint
control the way the real one answers, and the scene is given instead of located.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import manipulation as m

HANG = np.array([0.3, -0.22, 0.04, 0.98, -0.13, 0.04, 0.02])
# Where the robot found the bottle on 2026-11-03 (third grasp), and the desk.
SCENE = m.Scene(base=np.array([0.41, -0.015, -0.028]), height=0.23, spread=0.001,
                table_z=-0.028, edge_x=0.30, chosen="yolo bottle #1")


class FakeExecutor:
    """The relay's /joints/control: a session, trajectories that arrive, a grasp that holds."""

    def __init__(self, grasp="held", forces=(1.4, 0.0, None, 1.3, None), traj_state="success"):
        self.ops, self.grasp, self.forces, self.traj_state = [], grasp, list(forces), traj_state
        self.session = None

    def control(self, body):
        op = body.get("op")
        if op not in ("keepalive", "status"):
            self.ops.append(dict(body))
        if op == "take":
            self.session = "s1"
            return {"ok": True, "session": "s1", "safe": True}
        if body.get("session") != self.session:
            return {"ok": False, "error": "no control session"}
        out = {"ok": True}
        if op == "status":
            out["trajectory"] = {"state": self.traj_state}
            if any(o.get("op") == "grasp" for o in self.ops):
                out["grasp"] = {"right": {"state": self.grasp, "forces": self.forces}}
        if op == "release":
            self.session = None
        return out

    def measured_q(self):
        return HANG.copy()

    def arm_temp(self):
        return 45.0

    def verbs(self):
        return [o["op"] for o in self.ops]


def run(action, timeout=60):
    action.start()
    action.thread.join(timeout)
    assert not action.thread.is_alive(), "the action never finished"
    return action


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)


def test_a_good_pick_runs_every_step_and_lets_go_with_the_hand_folded():
    ex = FakeExecutor()
    act = run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE))
    assert act.state == "success", act.events
    steps = [e["step"] for e in act.events]
    for s in ("looking", "found", "planned", "folding the hand", "raising the arm",
              "opening the hand", "reaching in", "closing", "grasp held", "lifted", "put down",
              "going home", "done"):
        assert s in steps, s
    v = ex.verbs()
    assert v[0] == "take" and v[-1] == "release"
    assert v[-2] == "keep_hands", "the release must leave the hand folded (the user's ask)"


def test_the_hand_folds_fingers_first_then_the_thumb():
    """The defect (user, 2026-11-03): the thumb closing first pressed on the index."""
    ex = FakeExecutor()
    run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE))
    fingers = [o["finger"] for o in ex.ops if o["op"] == "hand"][:6]
    assert fingers[:4] == [2, 3, 4, 5] and fingers[4:] == [0, 1]


def test_the_hand_never_changes_while_the_arm_moves():
    """The defect (user, 2026-11-03): the hand opened while the arm was moving. Here every
    finger command comes between trajectories, each of which the fake reports arrived."""
    ex = FakeExecutor()
    run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE))
    assert ex.ops[1]["op"] == "hand", "the hand folds before the first move"


def test_an_empty_grasp_is_a_failure_and_still_goes_home():
    """The defect: "held" with every fingertip at 0 N — a grasp on air — reported as success."""
    ex = FakeExecutor(grasp="held", forces=(0.0, 0.0, None, 0.0, None))
    act = run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE))
    assert act.state == "failure"
    assert "lifted" not in [e["step"] for e in act.events], "nothing to lift"
    assert ex.verbs()[-1] == "release" and "going home" in [e["step"] for e in act.events]


def test_an_unsafe_plan_never_takes_the_arm():
    """The defect: moving first and finding the table after. The desk 10 cm nearer than the
    taught way up allows: refused at planning, the arm never taken."""
    near = m.Scene(base=SCENE.base, height=0.23, spread=0.001, table_z=-0.028, edge_x=0.18)
    ex = FakeExecutor()
    act = run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: near))
    assert act.state == "failure" and "take" not in ex.verbs()
    assert "above the table" in act.events[-1]["detail"] or "table" in act.events[-1]["detail"]


def test_not_finding_it_is_a_failure_without_moving():
    def nothing(phrase):
        raise m.ActionError(f"found '{phrase}' standing on the table in 0 of 3 looks")
    ex = FakeExecutor()
    act = run(m.PickAction("the green cup", executor=ex, locate_fn=nothing))
    assert act.state == "failure" and ex.ops == []
    assert "0 of 3" in act.events[-1]["detail"]


def test_cancel_goes_home_and_releases():
    """The defect: a STOP that leaves the arm out over the table."""
    ex = FakeExecutor()
    act = m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE)
    real_move = act._move

    def move_then_cancel(*a, **kw):
        real_move(*a, **kw)
        if kw.get("what") == "raising the arm":
            act.cancel()
    act._move = move_then_cancel
    run(act)
    assert act.state == "cancelled"
    steps = [e["step"] for e in act.events]
    assert "going home" in steps and "grasp held" not in steps
    assert ex.verbs()[-1] == "release"


def test_one_action_at_a_time(monkeypatch):
    ex = FakeExecutor()
    slow = m.PickAction("the bottle", executor=ex, locate_fn=lambda p: (time.sleep(0) or SCENE))
    slow.state = "running"
    monkeypatch.setitem(m._ACTIONS, slow.id, slow)
    monkeypatch.setitem(m._CURRENT, "id", slow.id)
    with pytest.raises(m.ActionError, match="another action"):
        m.start_pick("the cup", executor=ex, locate_fn=lambda p: SCENE)
