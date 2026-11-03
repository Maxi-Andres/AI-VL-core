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


# --------------------------------------------------------------------------- #
# Placing
# --------------------------------------------------------------------------- #
# A 10 cm tall box, 16 cm across, 34 cm to the robot's right of the bottle: where the sweep of
# 2026-11-03 found the right arm can carry it upright with every check passing.
BOX = m.Scene(base=np.array([0.40, -0.35, -0.028]), height=0.10, spread=0.005, table_z=-0.028,
              edge_x=0.30, chosen="vlm bbox", radius=0.08)


def place(ex, **kw):
    return m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE, into="the box",
                        locate_target_fn=lambda p: BOX, **kw)


def trajectories(ex):
    return [o for o in ex.ops if o["op"] == "trajectory"]


def last_q(traj):
    return np.array(list(traj["points"][-1]["q"].values()))


def test_a_good_place_carries_it_over_lets_go_there_and_goes_home():
    ex = FakeExecutor()
    act = run(place(ex))
    assert act.state == "success", act.events
    assert act.result["placed"] is True
    steps = [e["step"] for e in act.events]
    order = ["grasp held", "lifted", "carrying it", "lowering", "let go", "going home", "done"]
    assert [s for s in steps if s in order] == order
    assert "put down" not in steps, "placing is instead of putting it back"
    assert act.snapshot()["verb"] == "place" and act.snapshot()["into"] == "the box"


def test_it_lets_go_over_the_target_not_where_it_was():
    """The defect: opening the hand anywhere but over the box drops the bottle on the table."""
    ex = FakeExecutor()
    act = run(place(ex))
    k = next(i for i, o in enumerate(ex.ops) if o["op"] == "open")
    before = [o for o in ex.ops[:k] if o["op"] == "trajectory"][-1]
    hand = m.ik.forward(m.SIDE, last_q(before))[0]
    assert np.linalg.norm((hand - BOX.base)[:2]) < 0.01, "let go away from the box"
    assert hand[2] > BOX.base[2] + BOX.height, "let go inside the box's wall"
    assert act.state == "success"


def test_the_carried_object_stays_upright():
    """The defect: the first carry leaned the bottle 23 degrees 15 cm up — the IK's orientation
    step pushed sideways (test_arm_ik), so the wrist never straightened it."""
    plan = m.plan_pick(HANG, SCENE, m.taught_poses(), BOX)
    for q in (plan.place.carry, plan.place.over, plan.place.down):
        assert m.tilt_deg(q, plan.grasp) < 2.0


def test_a_box_in_the_hands_way_is_refused_before_moving():
    """The defect: a target right where the hand comes down beside the bottle."""
    in_the_way = m.Scene(base=np.array([0.44, -0.20, -0.028]), height=0.10, spread=0.005,
                         table_z=-0.028, edge_x=0.30, chosen="vlm bbox", radius=0.08)
    ex = FakeExecutor()
    act = run(m.PickAction("the bottle", executor=ex, locate_fn=lambda p: SCENE, into="the box",
                           locate_target_fn=lambda p: in_the_way))
    assert act.state == "failure" and "take" not in ex.verbs()
    assert "target" in act.events[-1]["detail"]


def test_a_dry_run_plans_and_never_takes_the_arm():
    ex = FakeExecutor()
    act = run(place(ex, dry=True))
    assert act.state == "success" and act.result["dry"] is True
    assert ex.ops == [], "a dry run must not touch the robot"
    assert "carry->over target" in act.result["margins_cm"]


def test_cancel_while_carrying_puts_it_back_where_it_was_first():
    """The defect: a STOP mid-carry folding the hand into a fist in the air, which drops it."""
    ex = FakeExecutor()
    act = place(ex)
    real_move = act._move

    def move_then_cancel(*a, **kw):
        real_move(*a, **kw)
        if kw.get("what") == "carrying it":
            act.cancel()
    act._move = move_then_cancel
    run(act)
    assert act.state == "cancelled"
    k = max(i for i, o in enumerate(ex.ops) if o["op"] == "open")
    before = [o for o in ex.ops[:k] if o["op"] == "trajectory"][-1]
    plan = m.plan_pick(HANG, SCENE, m.taught_poses(), BOX)
    assert np.allclose(last_q(before), plan.grasp, atol=1e-3), "opened away from where it was"
    fist = [i for i, o in enumerate(ex.ops) if o["op"] == "hand" and o.get("finger") == 2]
    assert fist[-1] > k, "the fist comes only after letting go"
    assert ex.verbs()[-1] == "release"
