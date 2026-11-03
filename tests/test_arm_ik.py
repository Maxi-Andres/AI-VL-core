"""arm_ik — seven angles for a hand position, and a grasp planned as the relay's trajectory.

Every test names the defect it catches. Pure math: no robot, no relay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arm_ik as ik

# A ready pose: upper arm down, forearm forward, wrist straight (elbow ~ at the side).
READY = [0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0]
# The bottle the head camera located on 2026-11-03 (perception3d, method "table").
BOTTLE_GRASP = (0.321, -0.062, 0.020)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_arms_mirror_each_other(side):
    """The defect: a sign slip copying one arm's URDF into the other. At the same angles the
    two hands must be mirror images across the robot's centre plane."""
    q = [0.4, 0.2 if side == "left" else -0.2, 0.1 if side == "left" else -0.1, 0.8, 0.0, 0.0, 0.0]
    p, _r, _j = ik.forward(side, q)
    mirror = [0.4, -q[1], -q[2], 0.8, 0.0, 0.0, 0.0]
    p2, _r, _j = ik.forward("left" if side == "right" else "right", mirror)
    assert p == pytest.approx([p2[0], -p2[1], p2[2]], abs=1e-3)


def test_the_right_hand_is_on_the_right():
    p, _r, _j = ik.forward("right", READY)
    assert p[1] < -0.1 and p[0] > 0.1      # beside the right hip, forward of the torso


@pytest.mark.parametrize("target", [(0.30, -0.15, 0.0), (0.25, -0.05, 0.10),
                                    (0.35, -0.20, -0.05), BOTTLE_GRASP])
def test_ik_reaches_what_it_says_it_reaches(target):
    """The defect: an IK that reports success with the hand somewhere else."""
    q = ik.solve("right", target, READY)
    p, _r, _j = ik.forward("right", q)
    assert np.linalg.norm(p - np.asarray(target)) < 0.005


def test_every_angle_stays_inside_what_the_relay_passes_unchanged():
    """The defect: a target the sender clamps — the hand ends up short of the bottle with no
    error anywhere. Planning keeps LIMIT_MARGIN inside the URDF range; the relay clamps at 0.05."""
    lo, hi = ik.limits("right", margin=0.05)
    for target in [(0.30, -0.15, 0.0), BOTTLE_GRASP, (0.15, -0.30, 0.15)]:
        try:
            q = ik.solve("right", target, READY)
        except ik.Unreachable:
            continue
        assert (q >= lo).all() and (q <= hi).all()


def test_out_of_reach_is_an_error_with_the_miss():
    """The defect: "the closest it could get" sent as if it were the target."""
    with pytest.raises(ik.Unreachable, match="cm"):
        ik.solve("right", (1.2, -0.2, 0.0), READY)


def test_the_answer_is_the_nearest_one_to_the_measured_pose():
    """The defect: a far solution (elbow flipped, shoulder wound round) for a target the arm is
    already next to — a big, needless swing. A target 2 cm away must move every joint a little."""
    p, _r, _j = ik.forward("right", READY)
    q = ik.solve("right", p + np.array([0.02, 0.0, 0.0]), READY)
    assert np.abs(q - np.asarray(READY)).max() < 0.3


def test_the_grasp_comes_in_fingers_first_with_the_palm_to_the_bottle():
    """The defect: reaching the right point with the hand turned so the fingers cannot close
    round the bottle (palm down, or facing away)."""
    plan = ik.plan_grasp("right", BOTTLE_GRASP, READY)
    q = np.asarray(plan["poses"]["grasp"])
    _p, rot, _j = ik.forward("right", q)
    approach = np.asarray(plan["approach"])
    fingers = rot @ ik.FINGER_AXIS
    palm = rot @ ik.PALM_NORMAL["right"]
    assert fingers @ approach > 0.8                       # fingers roughly along the approach
    assert palm @ np.cross([0, 0, 1], approach) > 0.7     # palm to the bottle (its left)


def test_the_plan_never_asks_the_relay_for_more_than_its_speed_limit():
    """The defect: a trajectory the relay refuses (or the sender cannot keep up with) because a
    segment is too short for how far a joint goes."""
    plan = ik.plan_grasp("right", BOTTLE_GRASP, READY)
    prev, t0 = dict(zip(plan["names"], READY, strict=True)), 0.0
    for p in plan["points"]:
        dt = p["t"] - t0
        for name, v in p["q"].items():
            assert abs(v - prev[name]) / dt <= ik.MAX_VEL + 1e-9
        prev, t0 = p["q"], p["t"]
    assert [p["label"] for p in plan["points"]] == ["pre_grasp", "grasp", "lift"]


def test_the_lift_goes_up_and_the_pre_grasp_stays_back():
    plan = ik.plan_grasp("right", BOTTLE_GRASP, READY)
    pos = {k: ik.forward("right", v)[0] for k, v in plan["poses"].items()}
    assert pos["lift"][2] - pos["grasp"][2] == pytest.approx(ik.LIFT_M, abs=0.01)
    back = np.asarray(plan["approach"]) @ (pos["grasp"] - pos["pre_grasp"])
    assert back == pytest.approx(ik.PRE_GRASP_M, abs=0.01)
