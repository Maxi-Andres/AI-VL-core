#!/usr/bin/env python3
"""
arm_ik.py — joint angles that put the G1's hand where perception says the object is.

perception3d answers "where is the bottle" in torso_link; this answers "which seven arm angles
put the hand there", and plans the three moves of a grasp (pre-grasp, grasp, lift) as the
relay's `trajectory` points. Pure math, no robot: what runs it decides whether to send it.

THE CHAIN is Unitree's URDF, g1_29dof_lock_waist_with_hand_rev_1_0 (the variant this robot is:
waist roll and pitch locked, joint_ctl_core.hpp). Both arms hang from torso_link, the same frame
the head camera's points are in, so nothing between camera and arm moves: not the waist, not
the legs, not how the robot leans.

THE HAND IS NOT IN THE URDF. Ours are BrainCo Revo 2, and Unitree's file ends at a rubber hand's
palm. Where the grasp centre sits — the point that must coincide with the object's axis when the
fingers close — is TOOL, measured with the head camera on 2026-11-03 (see TOOL).

IK is damped least squares on the position (3), with the orientation (the finger direction and
the palm normal) solved in what the position leaves free, seeded from the arm's measured pose so the answer is the nearest
solution, not a far one. Limits are the relay's own (URDF range minus its 0.05 rad margin) and a
little more, so a target is never one the sender would clamp. An unreachable target is an error
with the miss in metres, never "the closest it could get" silently.
"""
from __future__ import annotations

import json
import math

import numpy as np

# (relay name, origin xyz, origin rpy, axis, (lower, upper)) — parent first, from torso_link.
CHAINS = {
    "left": (
        ("left_shoulder_pitch", (0.0039563, 0.10022, 0.24778),
         (0.27931, 5.4949e-05, -0.00019159), (0, 1, 0), (-3.0892, 2.6704)),
        ("left_shoulder_roll", (0.0, 0.038, -0.013831), (-0.27925, 0.0, 0.0), (1, 0, 0),
         (-1.5882, 2.2515)),
        ("left_shoulder_yaw", (0.0, 0.00624, -0.1032), (0.0, 0.0, 0.0), (0, 0, 1),
         (-2.618, 2.618)),
        ("left_elbow", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0, 1, 0), (-1.0472, 2.0944)),
        ("left_wrist_roll", (0.1, 0.00188791, -0.01), (0.0, 0.0, 0.0), (1, 0, 0),
         (-1.972222054, 1.972222054)),
        ("left_wrist_pitch", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 1, 0),
         (-1.614429558, 1.614429558)),
        ("left_wrist_yaw", (0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 1),
         (-1.614429558, 1.614429558)),
    ),
    "right": (
        ("right_shoulder_pitch", (0.0039563, -0.10021, 0.24778),
         (-0.27931, 5.4949e-05, 0.00019159), (0, 1, 0), (-3.0892, 2.6704)),
        ("right_shoulder_roll", (0.0, -0.038, -0.013831), (0.27925, 0.0, 0.0), (1, 0, 0),
         (-2.2515, 1.5882)),
        ("right_shoulder_yaw", (0.0, -0.00624, -0.1032), (0.0, 0.0, 0.0), (0, 0, 1),
         (-2.618, 2.618)),
        ("right_elbow", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0, 1, 0),
         (-1.0472, 2.0944)),
        ("right_wrist_roll", (0.1, -0.00188791, -0.01), (0.0, 0.0, 0.0), (1, 0, 0),
         (-1.972222054, 1.972222054)),
        ("right_wrist_pitch", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 1, 0),
         (-1.614429558, 1.614429558)),
        ("right_wrist_yaw", (0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 1),
         (-1.614429558, 1.614429558)),
    ),
}

# The relay clamps every target to the URDF range minus 0.05 rad (joint_ctl_core.hpp, rule 4);
# planning 0.05 inside THAT keeps every target one it passes through unchanged.
LIMIT_MARGIN = 0.10

# In the wrist_yaw link: x runs along the fingers, the palm faces +y on the right hand and -y on
# the left (Unitree's zero pose: palms facing each other, thumbs up). UNVERIFIED for the BrainCo:
# check which way its palm faces at zero before trusting an orientation. As asked by
# plan_grasp (fingers along the approach, palm to the object), the right arm cannot meet both
# for the 2026-11-03 bottle: the best of 300 random seeds was 25 degrees off in all (wrist
# limits), so a plan reports its orientation error instead of pretending.
FINGER_AXIS = np.array([1.0, 0.0, 0.0])
PALM_NORMAL = {"right": np.array([0.0, 1.0, 0.0]), "left": np.array([0.0, -1.0, 0.0])}

# The grasp centre in the wrist_yaw link — where the object's axis must be when the fingers close.
# MEASURED 2026-11-03 with the head camera (the hand held still in view, its depth points
# projected on the model's finger axis): the BrainCo runs from the wrist_yaw origin to ~0.24 m
# along the fingers, and the model and camera agree on where it is. 0.14 along the fingers is
# the middle of the palm; out from the palm by a bottle's radius plus the fingers' own thickness,
# so the fingers pass beside the object instead of grazing it.
TOOL = {"right": (0.14, 0.055, 0.0), "left": (0.14, -0.055, 0.0)}
# How far past the grasp centre the fingertips reach, along the fingers: 0.24 - 0.14. The
# pre-grasp is set from THIS, not from the centre — the first computed reach stopped its centre
# 10 cm short and the fingertips, 14 cm further on, knocked the bottle over.
FINGERTIP_M = 0.10
TIP_CLEARANCE_M = 0.05

# How much moving each joint "costs" the solver (shoulder pitch, roll, yaw, elbow, wrist x3).
# Opening the shoulder sideways (roll) is the most expensive: left free, the solver lifted the
# hand by swinging the arm out 74 degrees with the wrist twisted — it reached, but looked wrong
# and swept a wide arc (2026-11-03, the user's sim screenshots). A person bends the elbow and
# turns the forearm in; these costs make that the cheaper answer.
JOINT_COST = np.array([1.5, 4.0, 1.0, 1.0, 1.5, 1.5, 1.5])

# The way up, as joint postures (right arm; mirrored for the left), before the arm reaches out —
# what the user posed from the Joints page on 2026-11-03 and the robot then ran: TUCK, the upper
# arm back, the elbow at 90 degrees and the hand in front of the belly; LIFT, the forearm raised
# from there and the wrist straightened, which clears the table edge. Postures, not hand points:
# the point is that the ARM is folded, whatever the IK would pick to put the hand there.
TUCK = {"right": np.array([0.573, -0.215, 0.044, -0.036, -0.055, -0.673, 0.773]),
        "left": np.array([0.573, 0.215, -0.044, -0.036, 0.055, -0.673, -0.773])}
LIFT = {"right": np.array([0.3, -0.215, 0.044, -0.6, 0.0, 0.0, 0.0]),
        "left": np.array([0.3, 0.215, -0.044, -0.6, 0.0, 0.0, 0.0])}

def load_tool(path=None):
    """{side: (x, y, z)} from a measured tool file, else TOOL; plus whether it was measured."""
    if path:
        try:
            with open(path) as fh:
                d = json.load(fh)
            return {s: tuple(d[s]) for s in ("left", "right")}, True
        except FileNotFoundError:
            pass
    return dict(TOOL), False


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _axis_angle(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def joint_names(side):
    return [j[0] for j in CHAINS[side]]


def limits(side, margin=LIMIT_MARGIN):
    lo = np.array([j[4][0] + margin for j in CHAINS[side]])
    hi = np.array([j[4][1] - margin for j in CHAINS[side]])
    return lo, hi


def forward(side, q, tool=None):
    """(tool position, tool rotation, [(joint origin, joint axis)] in torso_link) for angles q.

    `q` is 7 angles in chain order. `tool` is the grasp centre in the last link (TOOL)."""
    tool = TOOL[side] if tool is None else tool
    rot, pos, joints = np.eye(3), np.zeros(3), []
    for (_name, xyz, rpy, axis, _lim), angle in zip(CHAINS[side], q, strict=True):
        pos = pos + rot @ np.asarray(xyz)
        rot = rot @ _rpy(*rpy)
        joints.append((pos.copy(), rot @ np.asarray(axis, dtype=np.float64)))
        rot = rot @ _axis_angle(axis, angle)
    return pos + rot @ np.asarray(tool), rot, joints


def _jacobian(p_tool, joints):
    jp = np.column_stack([np.cross(a, p_tool - o) for o, a in joints])
    jr = np.column_stack([a for _o, a in joints])
    return jp, jr


class Unreachable(ValueError):
    pass


def solve(side, target, seed, finger_dir=None, palm_dir=None, tool=None, iters=300,
          tol=0.005, damping=0.02):
    """7 angles putting the tool at `target` (torso_link, metres), or Unreachable.

    `finger_dir` / `palm_dir`: where the fingers should point and the palm should face (unit
    vectors in torso_link), each optional. POSITION FIRST: the orientation is improved only in
    what the position leaves free (the null space of its Jacobian — 7 joints, 3 for position),
    a step at a time, each kept only if the hand is still on the target, so it can never pull
    the hand off it; it is met as well as the arm and its limits allow. A weighted mix of the
    two left the hand 4 cm short of the bottle (2026-11-03). Seeded from
    `seed` (the measured pose), the answer is the nearest solution to it."""
    lo, hi = limits(side)
    target = np.asarray(target, dtype=np.float64)
    palm_axis = PALM_NORMAL[side]
    wants = [(w / np.linalg.norm(w), local)
             for w, local in ((finger_dir, FINGER_AXIS), (palm_dir, palm_axis)) if w is not None]

    def reach(q, steps):
        """Position-only damped least squares from q."""
        for _ in range(steps):
            p, _rot, joints = forward(side, q, tool)
            ep = target - p
            if np.linalg.norm(ep) < tol * 0.1:
                break
            jp, _jr = _jacobian(p, joints)
            # Weighted least squares: each joint's step scaled down by its cost (JOINT_COST).
            winv = np.diag(1.0 / JOINT_COST)
            dq = winv @ jp.T @ np.linalg.solve(jp @ winv @ jp.T + damping ** 2 * np.eye(3), ep)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
        return q

    def miss(q):
        return float(np.linalg.norm(target - forward(side, q, tool)[0]))

    def orient_cost(q):
        rot = forward(side, q, tool)[1]
        return sum(1.0 - (rot @ local) @ w for w, local in wants)

    # 1. Position alone, from the measured pose: the nearest arm that puts the hand there.
    q = reach(np.clip(np.asarray(seed, dtype=np.float64), lo, hi), iters)
    if miss(q) > tol:
        raise Unreachable(f"{side} hand misses the target by {miss(q) * 100:.1f} cm "
                          f"(out of reach, or only past a joint limit)")
    # 2. Orientation, in small steps along what the position leaves free; each step is put back
    # on the target and kept only if it is still there AND the hand turned the right way, so the
    # orientation can never cost position. A step that does not help is halved, then given up.
    cost, step = orient_cost(q) if wants else 0.0, 0.1
    while wants and step > 1e-3 and cost > 1e-5:
        p, rot, joints = forward(side, q, tool)
        jp, jr = _jacobian(p, joints)
        rows, errs = [], []
        for w, local in wants:
            have = rot @ local
            errs.append(np.cross(have, w))
            skew = np.array([[0, -have[2], have[1]], [have[2], 0, -have[0]],
                             [-have[1], have[0], 0]])
            rows.append(-skew @ jr)
        jp_pinv = jp.T @ np.linalg.inv(jp @ jp.T + damping ** 2 * np.eye(3))
        null = np.eye(7) - jp_pinv @ jp
        jon = np.vstack(rows) @ null
        do = null @ (jon.T @ np.linalg.solve(jon @ jon.T + 0.1 ** 2 * np.eye(len(errs) * 3),
                                             np.concatenate(errs)))
        n = np.linalg.norm(do)
        if n < 1e-9:
            break
        trial = reach(np.clip(q + do / n * step, lo, hi), 30)
        new = orient_cost(trial)
        if miss(trial) <= tol * 0.5 and new < cost - 1e-6:
            q, cost, step = trial, new, min(step * 1.5, 0.2)
        else:
            step /= 2
    return q


def orientation_error_deg(side, q, finger_dir=None, palm_dir=None, tool=None):
    """How far (degrees) the fingers and the palm are from where they were asked to point."""
    _p, rot, _j = forward(side, q, tool)
    out = {}
    for key, want, local in (("fingers", finger_dir, FINGER_AXIS),
                             ("palm", palm_dir, PALM_NORMAL[side])):
        if want is not None:
            w = np.asarray(want, dtype=np.float64)
            c = float(np.clip((rot @ local) @ (w / np.linalg.norm(w)), -1, 1))
            out[key] = round(math.degrees(math.acos(c)), 1)
    return out


# --------------------------------------------------------------------------- #
# A grasp: pre-grasp -> grasp -> lift, as relay trajectory points
# --------------------------------------------------------------------------- #
# The relay refuses anything faster than JOINT_CTL_MAX_VEL (0.5 rad/s); plans use a fraction of
# it so the sender never lags its own clock.
MAX_VEL = 0.5
PLAN_VEL_FRACTION = 0.5
MIN_SEGMENT_S = 1.5
# The hand starts this far back along the approach: fingertips TIP_CLEARANCE_M short of an
# object of radius ~3.5 cm (see FINGERTIP_M).
PRE_GRASP_M = FINGERTIP_M + 0.035 + TIP_CLEARANCE_M
LIFT_M = 0.08


def approach_direction(side, grasp_point):
    """Horizontal, from the shoulder to the object: how the hand comes in with its fingers
    forward and its palm to the object's side, the way the hand held the bottle on 2026-10-30."""
    shoulder = np.asarray(CHAINS[side][0][1])
    d = np.asarray(grasp_point, dtype=np.float64) - shoulder
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n < 0.05:
        raise Unreachable("the object is right at the shoulder")
    return d / n


def plan_grasp(side, grasp_point, seed, tool=None, up=(0.0, 0.0, 1.0)):
    """{"names", "poses": {pre_grasp, grasp, lift}: 7 angles each, "points": the relay's
    trajectory points (cumulative t, joint -> rad), "approach", "orientation_error_deg"}, or
    Unreachable."""
    grasp = np.asarray(grasp_point, dtype=np.float64)
    approach = approach_direction(side, grasp)
    up = np.asarray(up, dtype=np.float64)
    # The palm faces the object's axis across the approach: for the right hand, to its left.
    palm = np.cross(up, approach) if side == "right" else np.cross(approach, up)
    targets = {"pre_grasp": grasp - PRE_GRASP_M * approach, "grasp": grasp,
               "lift": grasp + LIFT_M * up}
    poses, errors, q = {}, {}, np.asarray(seed, dtype=np.float64)
    for name in ("pre_grasp", "grasp", "lift"):
        q = solve(side, targets[name], q, finger_dir=approach, palm_dir=palm, tool=tool)
        poses[name] = q
        errors[name] = orientation_error_deg(side, q, approach, palm, tool)
    names = joint_names(side)
    points, t, prev = [], 0.0, np.asarray(seed, dtype=np.float64)
    for name in ("pre_grasp", "grasp", "lift"):
        step = float(np.abs(poses[name] - prev).max())
        t += max(MIN_SEGMENT_S, step / (MAX_VEL * PLAN_VEL_FRACTION))
        points.append({"t": round(t, 2), "label": name,
                       "q": {n: round(float(v), 4) for n, v in zip(names, poses[name],
                                                                 strict=True)}})
        prev = poses[name]
    return {"names": names, "poses": {k: [round(float(v), 4) for v in q]
                                      for k, q in poses.items()},
            "points": points, "approach": [round(float(v), 4) for v in approach],
            "orientation_error_deg": errors}
