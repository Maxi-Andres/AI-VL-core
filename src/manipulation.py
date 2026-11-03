#!/usr/bin/env python3
"""
manipulation.py — "pick up the bottle" (and "put it in the box") as an ACTION: started,
followed step by step, cancelled.

The first computed grasp worked as a field script (tools/g1_reach/grasp_side.py, 3 of 3 on
2026-11-03). This is the same procedure as something an agent can call: it takes a phrase,
reports every step as it happens (the trace the /agent tab shows), and ends success | failure |
uncertain | cancelled, the action contract of G1.md §6.9. The robot moves only through the
executor (its /joints/control, the relay's limits behind it); the camera answers through
locate_object; the arm through arm_ik.

THE PROCEDURE, each part learnt the hard way on the robot (the commits of 2026-11-03):

    locate     3 times, agreeing within 1.5 cm, the object standing on the table
    plan       the way up is the user's, TAUGHT from the Joints page (config/g1_taught_poses.json);
               the IK only plans from its last posture: out to the object's right, then in
               beside it; every segment checked against the table edge where the right arm
               passes and against the object, with a pessimistic hand (a fist while folded),
               and so is the relay's own retreat to the safe pose
    move       fist first (fingers, then thumb), the arm still; the way up; open the hand far
               from the object, the arm still (fingers, then the thumb turned); slide in; the
               four fingers, then the grasp by touch; lift, hold, put down; open; slide out; fist;
               home the same way; the hand kept folded through the release

    place      (with `into`) instead of putting it back: up until its bottom clears the target,
               over the target, down to just above its top, open, straight up clear of it; home
               from there. The carried object's bottom and the target are in every check

The object is any phrase locate_object can resolve (YOLO finds, the VLM chooses); so is the
target, by the VLM alone (no YOLO class is "box"). What is specific is the grasp: from the side,
the right hand, an upright object on a table, put down from above.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

import numpy as np

import arm_ik as ik
import perception3d as p3
from vlm_common import PROJECT_ROOT

EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8090").rstrip("/")
TAUGHT_POSES = os.path.join(PROJECT_ROOT, "config", "g1_taught_poses.json")
ROBOT = "g1"
SIDE = "right"
RIGHT_Q = list(range(22, 29))       # rt/lowstate indices of the right arm's 7 joints

# The geometry of the grasp (see the module docstring and grasp_side's history).
SIDE_M = 0.08            # the hand waits this far to the object's right, beside it
WIDE_M = 0.18            # and comes down from above even further out
LIFT_M = 0.08
HOLD_S = 0.5
GRASP_AT = 0.4           # of the object's height, from the table: its widest part, for a bottle
LOCATE_RUNS = 3
LOCATE_AGREE_M = 0.015
CLEAR_M = 0.04           # every arm point over the table stays this far above it
OBJECT_GAP_M = 0.06      # the fingertips pass the object's axis no nearer on the way in
HAND_TIP = 0.38          # the hand's far end along the fingers, pessimistic (see reach_stage1)
HAND_TIP_CLOSED = 0.29   # the same, folded into a fist
START_MAX_C = 60         # the right arm cool enough to start (it heats ~1 C/s held out)
APPROACH_VEL = 0.4       # of the relay's max speed, near the object
FREE_VEL = 0.85          # and away from it: the way up and home (23 s each at 0.4, 2026-11-03)
MIN_SEGMENT_S = 1.2      # no segment shorter, however small
GRASP_WAIT_S = 6.0
# Placing (see plan_place).
CARRY_CLEAR_M = 0.06     # the carried object's bottom passes this far above the target's top
RELEASE_ABOVE_M = 0.02   # and is let go this far above it
HAND_BELOW_M = 0.045     # the hand's lowest part under the grasp point: half the palm
FOOTPRINT_M = 0.03       # the target is this much wider than it looked, for the checks
MAX_TILT_DEG = 15        # the carried object leans no more than this off upright
TARGET_AGREE_M = 0.03    # a VLM box is looser than YOLO's: the target's looks agree this well
# [thumb flex, thumb rotation, index, middle, ring, pinky], 0 open .. 1000 closed.
FIST = [600, 800, 899, 899, 899, 899]      # as the user set it from the Joints page
OPEN_WIDE = [0, 950, 0, 0, 0, 0]           # the thumb turned perpendicular: what held the bottle


class ActionError(RuntimeError):
    """The action cannot go on; the message says why, in words for the trace."""


# --------------------------------------------------------------------------- #
# The executor, the only way to the robot
# --------------------------------------------------------------------------- #
class Executor:
    """The executor's joint routes. Injected, so the action can be tested without a robot."""

    def __init__(self, url=EXECUTOR_URL, robot=ROBOT):
        self.url, self.robot = url, robot

    def control(self, body):
        req = urllib.request.Request(f"{self.url}/joints/control?robot={self.robot}",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            try:
                out = json.loads(exc.read() or b"{}")
            except ValueError:
                out = {}
            return {"ok": False, "http": exc.code, **out}

    def joints(self, tries=4):
        """One line of the joint stream (q, temp, hands). A few tries: the executor answered
        502 to 8 of 10 opens in a row on 2026-11-03 while the relay behind it streamed fine."""
        for k in range(tries):
            try:
                with urllib.request.urlopen(f"{self.url}/joints/stream?robot={self.robot}",
                                            timeout=5) as r:
                    for line in r:
                        d = json.loads(line)
                        if d.get("q"):
                            return d
            except (urllib.error.URLError, TimeoutError):
                if k == tries - 1:
                    raise
                time.sleep(0.5)
        raise ActionError("the joint stream sent nothing")

    def measured_q(self):
        return np.array([self.joints()["q"][i] for i in RIGHT_Q])

    def arm_temp(self):
        t = self.joints().get("temp") or [0] * 29
        return max(t[i] for i in RIGHT_Q)


# --------------------------------------------------------------------------- #
# Seeing: where the object is, and the table under it
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    base: np.ndarray          # the object's base on the table, torso frame
    height: float
    spread: float             # how far the LOCATE_RUNS locates were from their median
    table_z: float
    edge_x: float             # the table's near edge where the right arm passes
    chosen: str = ""
    radius: float = 0.0


def locate(phrase, classes=("bottle", "cup"), runs=LOCATE_RUNS, frames=5, agree=LOCATE_AGREE_M):
    """The object standing on the table, located `runs` times; ActionError if they disagree by
    more than `agree` or it is not found standing on a table."""
    import locate_object as lo
    cal = p3.load_calibration(lo.CALIBRATION)
    found, z = [], None
    chosen = ""
    for _ in range(runs):
        z, _info = lo.fetch_depth(frames=frames)
        frame = lo.grab_color()
        box, how, _cands = lo.choose(frame, list(classes), phrase)
        if box is None:
            continue
        h, w = frame.shape[:2]
        r = p3.locate(z, box, cal, color_size=(w, h))
        if r.get("ok") and r["method"] == "table":
            found.append(r)
            chosen = how
    if len(found) < runs:
        raise ActionError(f"found '{phrase}' standing on the table in {len(found)} of {runs} looks")
    base = np.median([r["torso"] for r in found], axis=0)
    spread = float(max(np.linalg.norm(np.array(r["torso"]) - base) for r in found))
    if spread > agree:
        raise ActionError(f"the {runs} looks disagree by {spread * 100:.1f} cm")
    pts = p3.depth_points(z, cal.depth, step=2)
    T = (p3.rpy_matrix(*p3.D435_RPY) @ (p3.OPTICAL_TO_LINK @ pts.T)).T + np.array(p3.D435_XYZ)
    # The edge where the RIGHT arm passes: the desk can sit at an angle to the robot.
    top = T[(np.abs(T[:, 2] - base[2]) < 0.015) & (T[:, 1] < base[1] + 0.03)
            & (T[:, 1] > base[1] - 0.22)]
    if len(top) < 50:
        raise ActionError("no table seen where the arm would pass")
    return Scene(base=base, height=float(np.median([r["height_m"] for r in found])),
                 spread=spread, table_z=float(base[2]),
                 edge_x=float(np.percentile(top[:, 0], 2)), chosen=chosen,
                 radius=float(np.median([r["radius_m"] for r in found])))


def locate_target(phrase):
    """Where to put something: the VLM's own box (no YOLO class is "box"), looser agreement."""
    return locate(phrase, classes=(), agree=TARGET_AGREE_M)


# --------------------------------------------------------------------------- #
# Planning, and checking the plan
# --------------------------------------------------------------------------- #
def taught_poses(path=TAUGHT_POSES):
    with open(path) as fh:
        poses = json.load(fh)
    names = ik.joint_names(SIDE)
    return {p["name"]: np.array([p["q"][n] for n in names]) for p in poses}


def arm_points(q, tip=HAND_TIP, below=0.0):
    """Elbow, wrist, grasp point, the hand's far end and its lowest part (torso frame) for a
    pose; with `below` > 0, also the bottom of an object held that far under the grasp point."""
    p, _rot, joints = ik.forward(SIDE, q)
    tips = ik.forward(SIDE, q, tool=(tip, 0.0, 0.0))[0]
    pts = [joints[3][0], joints[4][0], joints[6][0], p, tips, p - (0.0, 0.0, HAND_BELOW_M)]
    if below > 0:
        pts.append(p - (0.0, 0.0, below))
    return pts


def table_margin(qa, qb, table_z, edge_x, tip=HAND_TIP, steps=25, below=0.0):
    """The lowest any arm point (and a held object's bottom) gets above the table while past
    its edge, along qa -> qb (joint-space, as the relay interpolates). inf if nothing is ever
    over the table."""
    worst = np.inf
    for s in np.linspace(0, 1, steps):
        for pt in arm_points(qa + s * (qb - qa), tip, below):
            if pt[0] > edge_x:
                worst = min(worst, pt[2] - table_z)
    return worst


def target_margin(qa, qb, target, tip=HAND_TIP, steps=25, below=0.0):
    """The lowest any arm point (and a held object's bottom) gets above the TARGET's top while
    over its footprint, along qa -> qb. Negative = through its wall. inf if never over it."""
    worst, top = np.inf, target.base[2] + target.height
    for s in np.linspace(0, 1, steps):
        for pt in arm_points(qa + s * (qb - qa), tip, below):
            if np.linalg.norm((pt - target.base)[:2]) < target.radius + FOOTPRINT_M:
                worst = min(worst, pt[2] - top)
    return worst


def grasp_height(scene):
    """How far above the table the hand closes: GRASP_AT of the object, but never so low that
    the hand's lower edge comes within CLEAR_M + 1 cm of the table (a 22 cm bottle at 0.4 left
    it at 4.0 cm, refused, 2026-11-03)."""
    return max(GRASP_AT * scene.height, HAND_BELOW_M + CLEAR_M + 0.01)


def tilt_deg(q, q_ref):
    """How far an object held upright at pose q_ref leans off upright at pose q."""
    up = np.array([0.0, 0.0, 1.0])
    axis = ik.forward(SIDE, q)[1] @ ik.forward(SIDE, q_ref)[1].T @ up
    return float(np.degrees(np.arccos(np.clip(axis @ up, -1.0, 1.0))))


def object_gap(qa, qb, base, height, steps=30):
    """How near the hand's far end and grasp point come to the object's axis along qa -> qb."""
    gap = np.inf
    for s in np.linspace(0, 1, steps):
        for tool in ((ik.TOOL[SIDE][0] + ik.FINGERTIP_M, 0.0, 0.0), ik.TOOL[SIDE]):
            p = ik.forward(SIDE, qa + s * (qb - qa), tool=tool)[0]
            if base[2] < p[2] < base[2] + height + 0.03:
                gap = min(gap, float(np.linalg.norm((p - base)[:2])))
    return gap


@dataclass
class Placing:
    carry: np.ndarray         # above where it was, its bottom clear of the target
    over: np.ndarray          # above the target
    down: np.ndarray          # let go here
    clear: np.ndarray         # straight up from there, the fingers above the object's top


@dataclass
class Plan:
    way_up: list              # taught postures, hanging -> ready
    straight: np.ndarray      # ready with the wrist straightened, the forearm raised: SAFE pose
    wide: np.ndarray
    beside: np.ndarray
    grasp: np.ndarray
    lifted: np.ndarray
    margins: dict = field(default_factory=dict)
    need: dict = field(default_factory=dict)     # a margin's own minimum, if not CLEAR_M
    gap: float = np.inf
    place: Placing | None = None

    def short(self):
        """(segment, margin) of the worst margin against what it needs, or None if all pass."""
        bad = [(m - self.need.get(k, CLEAR_M), k) for k, m in self.margins.items()]
        worst = min(bad)
        return (worst[1], self.margins[worst[1]]) if worst[0] < 0 else None


def plan_place(plan, scene, target):
    """The carry of a place, appended to `plan` and checked with the object hanging from the
    hand, which keeps it upright (MAX_TILT_DEG)."""
    up = np.array([0.0, 0.0, 1.0])
    below = grasp_height(scene)                         # the object's bottom under the grasp
    top = target.base[2] + target.height
    lifted = ik.forward(SIDE, plan.lifted)[0]
    carry_z = max(lifted[2], top + CARRY_CLEAR_M + below)
    down_z = top + RELEASE_ABOVE_M + below
    # Up clear of the object's top in case it lands standing (on a floor at the table's
    # height), the hand's lowest part above it.
    clear_z = down_z + max(CARRY_CLEAR_M - RELEASE_ABOVE_M,
                           scene.height - below - target.height + HAND_BELOW_M + 0.02)
    over_xy = target.base[:2]
    # Only the object's axis is held up, as it was held at the grasp; the turn about it does not
    # lean it, so it is left to the arm: more room for the carry than the whole orientation.
    held_axis = ik.forward(SIDE, plan.grasp)[1].T @ up
    try:
        q = plan.lifted
        poses = []
        for p in ((*lifted[:2], carry_z), (*over_xy, carry_z), (*over_xy, down_z),
                  (*over_xy, clear_z)):
            q = ik.solve(SIDE, np.array(p), q, keep=[(held_axis, up)])
            poses.append(q)
    except ik.Unreachable as exc:
        raise ActionError(f"the hand cannot carry it there: {exc}") from None
    placing = Placing(*poses)
    lean = max(tilt_deg(q, plan.grasp) for q in poses[:3])
    if lean > MAX_TILT_DEG:
        raise ActionError(f"it would lean {lean:.0f} degrees on the way to the target")
    edge = min(scene.edge_x, target.edge_x) - 0.03
    held = [("lift->carry", plan.lifted, placing.carry), ("carry->over", placing.carry,
             placing.over), ("over->down", placing.over, placing.down)]
    for name, a, b in held:
        plan.margins[name] = table_margin(a, b, scene.table_z, edge, below=below)
        plan.margins[f"{name} target"] = target_margin(a, b, target, below=below)
    # Going down onto the target is the point: the object's bottom ends RELEASE_ABOVE_M over it,
    # so there, and on the relay's retreat that may start there, that is what it needs.
    plan.need["over->down target"] = plan.need["retreat target"] = RELEASE_ABOVE_M - 0.005
    for name, a, b in (("down->clear", placing.down, placing.clear),
                       ("clear->straight", placing.clear, plan.straight)):
        plan.margins[name] = table_margin(a, b, scene.table_z, edge)
        plan.margins[f"{name} target"] = target_margin(a, b, target)
    for q in poses[:3]:                    # the relay's own straight retreat, still holding it
        plan.margins["retreat"] = min(plan.margins["retreat"], table_margin(
            q, plan.straight, scene.table_z, edge, below=below))
        plan.margins["retreat target"] = min(plan.margins.get("retreat target", np.inf),
                                             target_margin(q, plan.straight, target, below=below))
    plan.place = placing
    return plan


def plan_pick(q0, scene, taught, place_at=None):
    """The joint postures of a pick, checked; with `place_at` (a Scene), the carry to it too.
    ActionError with the reason if any check fails."""
    up = np.array([0.0, 0.0, 1.0])
    target = scene.base + grasp_height(scene) * up
    approach = ik.approach_direction(SIDE, target)
    palm = np.cross(up, approach)
    way_up = [taught[n] for n in ("way_up_2", "way_up_3", "way_up_4", "ready")]
    straight = taught["ready"].copy()
    straight[3] = -0.6
    straight[4:] = (0.0, -0.3, 0.0)
    try:
        q_wide = ik.solve(SIDE, target - WIDE_M * palm + 0.05 * up, ik.LIFT[SIDE],
                          finger_dir=approach, palm_dir=palm)
        q_beside = ik.solve(SIDE, target - SIDE_M * palm, q_wide, finger_dir=approach,
                            palm_dir=palm)
        q_grasp = ik.solve(SIDE, target, q_beside, finger_dir=approach, palm_dir=palm)
        q_lifted = ik.solve(SIDE, target + LIFT_M * up, q_grasp, finger_dir=approach,
                            palm_dir=palm)
    except ik.Unreachable as exc:
        raise ActionError(f"the hand cannot reach it: {exc}") from None
    plan = Plan(way_up, straight, q_wide, q_beside, q_grasp, q_lifted)
    path = [q0, *way_up, straight, q_wide, q_beside, q_grasp, q_lifted]
    names = ["hang", "up2", "up3", "up4", "ready", "straight", "wide", "beside", "grasp", "lift"]
    for k, (a, b) in enumerate(itertools.pairwise(path)):
        # A fist up to `wide`, the hand open from there on.
        tip = HAND_TIP_CLOSED if k < 6 else HAND_TIP
        plan.margins[f"{names[k]}->{names[k + 1]}"] = table_margin(
            a, b, scene.table_z, scene.edge_x - 0.03, tip)
    for q in (q_wide, q_beside, q_grasp, q_lifted):   # the relay's own straight retreat
        plan.margins["retreat"] = min(plan.margins.get("retreat", np.inf),
                                      table_margin(q, straight, scene.table_z,
                                                   scene.edge_x - 0.03))
    plan.gap = min(object_gap(straight, q_wide, scene.base, scene.height),
                   object_gap(q_wide, q_beside, scene.base, scene.height))
    if place_at is not None:
        # The way out to the object must not pass through the target either.
        for k, (a, b) in enumerate(itertools.pairwise(path)):
            plan.margins[f"{names[k]}->{names[k + 1]} target"] = target_margin(
                a, b, place_at, HAND_TIP_CLOSED if k < 6 else HAND_TIP)
        plan_place(plan, scene, place_at)
    if (short := plan.short()) is not None:
        seg, worst = short
        what = "the target" if seg.endswith("target") else "the table"
        raise ActionError(f"the arm would pass {worst * 100:.1f} cm above {what} ({seg})")
    if plan.gap < OBJECT_GAP_M:
        raise ActionError(f"the hand would pass {plan.gap * 100:.1f} cm from the object")
    return plan


# --------------------------------------------------------------------------- #
# The action
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    pass


class PickAction:
    """Pick up `phrase`, lift it, put it back (or, with `into`, put it in/on that), go home —
    on its own thread, step by step. `dry` locates and plans, and never takes the arm.

    snapshot() is what the agent and the UI read: state (running | success | failure |
    uncertain | cancelled), the step running now, every event so far, and the result."""

    def __init__(self, phrase, executor=None, locate_fn=locate, taught=None, hold=True,
                 into=None, locate_target_fn=locate_target, dry=False):
        self.id = uuid.uuid4().hex[:12]
        self.phrase, self.into = phrase, into
        self.ex = executor or Executor()
        self.locate_fn, self.locate_target_fn = locate_fn, locate_target_fn
        self.taught = taught
        self.hold, self.dry = hold, dry
        self.state, self.step, self.result = "running", "starting", None
        self.events = []
        self.cancelled = threading.Event()
        self.session = None
        # What the way home undoes: the poses reached since `wide`, in order, and the hand.
        self._trail, self._closed, self._released = [], False, False
        self._alive = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=f"pick-{self.id}", daemon=True)

    # -- reporting ------------------------------------------------------- #
    def emit(self, step, detail=""):
        with self._lock:
            self.step = step
            self.events.append({"t": round(time.time(), 2), "step": step, "detail": detail})

    def snapshot(self, since=0):
        with self._lock:
            return {"id": self.id, "verb": "place" if self.into else "pick",
                    "object": self.phrase, "into": self.into, "state": self.state,
                    "step": self.step, "result": self.result, "events": self.events[since:]}

    def start(self):
        self.thread.start()
        return self

    def cancel(self):
        self.cancelled.set()

    # -- the robot ------------------------------------------------------- #
    def _check(self):
        if self.cancelled.is_set():
            raise Cancelled()

    def _op(self, **body):
        return self.ex.control({"session": self.session, **body})

    def _hand(self, fingers):
        items = fingers.items() if isinstance(fingers, dict) else enumerate(fingers)
        for k, pos in items:
            out = self._op(op="hand", side=SIDE, finger=k, pos=pos)
            if not out.get("ok"):
                raise ActionError(f"the hand refused finger {k}: {out.get('error')}")

    def _fist(self):
        self._hand({k: FIST[k] for k in (2, 3, 4, 5)})       # the fingers first ...
        time.sleep(1.0)
        self._hand({0: FIST[0], 1: FIST[1]})                 # ... then the thumb
        time.sleep(1.0)

    def _move(self, q_from, poses, vel=FREE_VEL, what="moving"):
        pts, t, prev = [], 0.0, q_from
        names = ik.joint_names(SIDE)
        for q in poses:
            t += max(MIN_SEGMENT_S, float(np.abs(q - prev).max()) / (ik.MAX_VEL * vel))
            pts.append({"t": round(t, 2),
                        "q": {n: round(float(v), 4) for n, v in zip(names, q, strict=True)}})
            prev = q
        out = self._op(op="trajectory", points=pts)
        if not out.get("ok"):
            raise ActionError(f"{what}: the relay refused it ({out.get('error')})")
        while True:
            time.sleep(0.3)
            st = self._op(op="status")
            if st.get("retreat"):
                raise ActionError(f"the relay took the arm to safety ({st['retreat']['reason']})")
            if self.cancelled.is_set():
                self._op(op="cancel")
            tr = st.get("trajectory") or {}
            if tr.get("state") == "running":
                continue
            if tr.get("state") != "success":
                if tr.get("state") == "cancelled":
                    raise Cancelled()
                raise ActionError(f"{what}: {tr.get('state')} ({tr.get('reason')})")
            return

    def _go(self, poses, vel=FREE_VEL, what="moving"):
        """A move from the last pose of the trail, which it then extends."""
        self._move(self._trail[-1], poses, vel=vel, what=what)
        self._trail.extend(poses)

    def _open(self):
        self._op(op="open", side=SIDE)
        time.sleep(1.5)
        self._closed = False

    def _keepalive(self):
        while not self._alive.is_set():
            self._op(op="keepalive")
            time.sleep(0.3)

    # -- the procedure --------------------------------------------------- #
    def _run(self):
        plan = q0 = None
        try:
            self.emit("looking", f"finding '{self.phrase}'")
            scene = self.locate_fn(self.phrase)
            self.emit("found", f"{scene.chosen}: base at x {scene.base[0]:.2f} y "
                               f"{scene.base[1]:+.2f} m, {scene.height * 100:.0f} cm tall")
            target = None
            if self.into:
                self._check()
                self.emit("looking", f"finding '{self.into}'")
                target = self.locate_target_fn(self.into)
                self.emit("found", f"{target.chosen}: at x {target.base[0]:.2f} y "
                                   f"{target.base[1]:+.2f} m, {target.height * 100:.0f} cm tall, "
                                   f"{target.radius * 200:.0f} cm across")
            self._check()
            q0 = self.ex.measured_q()
            plan = plan_pick(q0, scene, self.taught or taught_poses(), target)
            self.emit("planned", f"{min(plan.margins.values()) * 100:.0f} cm over the table at "
                                 f"worst, the hand {plan.gap * 100:.0f} cm from it on the way in")
            if self.dry:
                self.result = {"dry": True, "margins_cm": {
                    k: round(v * 100, 1) for k, v in plan.margins.items() if np.isfinite(v)}}
                self.state = "success"
                return
            while (t := self.ex.arm_temp()) > START_MAX_C:
                self.emit("cooling", f"the arm is at {t:.0f} C, waiting for {START_MAX_C}")
                self._check()
                time.sleep(5)
            names = ik.joint_names(SIDE)
            took = self.ex.control({"op": "take", "safe": dict(zip(
                names, [round(float(v), 4) for v in plan.straight], strict=True))})
            if not took.get("ok"):
                raise ActionError(f"could not take the arm: {took.get('error')}")
            self.session = took["session"]
            threading.Thread(target=self._keepalive, daemon=True).start()

            self.emit("folding the hand")
            self._fist()
            self._check()
            self.emit("raising the arm", "the taught way up, close to the body")
            self._move(q0, [*plan.way_up, plan.straight, plan.wide], what="raising the arm")
            self._trail = [plan.wide]
            self._check()
            self.emit("opening the hand", "the arm still, away from the object")
            self._hand([0, 0, 0, 0, 0, 0])
            time.sleep(1.0)
            self._hand(OPEN_WIDE)
            time.sleep(1.5)
            self._go([plan.beside], vel=APPROACH_VEL, what="coming beside it")
            self._check()
            self.emit("reaching in")
            self._go([plan.grasp], vel=APPROACH_VEL, what="reaching in")
            self.emit("closing", "the fingers, then the thumb by touch")
            self._closed = True
            self._hand({2: 1000, 3: 1000, 4: 1000, 5: 1000})
            time.sleep(1.2)
            self._op(op="grasp", side=SIDE, thumb=OPEN_WIDE[1])
            state, forces = self._wait_grasp()
            held = state == "held" and forces and sum(f or 0 for f in forces) > 0.5
            self.emit("grasp " + ("held" if held else state or "unknown"),
                      "fingertips " + ", ".join("-" if f is None else f"{f:.1f} N"
                                                for f in (forces or [])))
            placed = False
            if held and self.hold:
                self._go([plan.lifted], vel=APPROACH_VEL, what="lifting")
                self.emit("lifted", f"{LIFT_M * 100:.0f} cm")
                self._check()
                if plan.place is not None:
                    pl = plan.place
                    self.emit("carrying it", f"to '{self.into}', its bottom clear of it")
                    self._go([pl.carry, pl.over], vel=APPROACH_VEL, what="carrying it")
                    self._check()
                    self.emit("lowering", "to just above it")
                    self._go([pl.down], vel=APPROACH_VEL, what="lowering")
                    self._released = True     # from here on, home is straight up and away
                    self._open()
                    self.emit("let go", f"in '{self.into}'")
                    self._go([pl.clear], vel=APPROACH_VEL, what="moving clear")
                    placed = True
                else:
                    time.sleep(HOLD_S)
                    self._go([plan.grasp], vel=APPROACH_VEL, what="putting it down")
                    self.emit("put down")
            if not self._released:
                self._open()
            self.result = {"held": bool(held), "forces": forces}
            if plan.place is not None:
                self.result["placed"] = placed
            self.state = "success" if held and (placed or plan.place is None) else "failure"
        except Cancelled:
            self.state = "cancelled"
            self.emit("cancelled", "going home")
        except ActionError as exc:
            self.state = "failure"
            self.emit("failed", str(exc))
        except Exception as exc:                 # the trace must say it, whatever it is
            self.state = "failure"
            self.emit("failed", f"{type(exc).__name__}: {exc}")
        finally:
            self._go_home(plan, q0)

    def _wait_grasp(self):
        deadline = time.monotonic() + GRASP_WAIT_S
        while time.monotonic() < deadline:
            time.sleep(0.3)
            st = self._op(op="status")
            if st.get("retreat"):
                return "retreat", None
            g = (st.get("grasp") or {}).get(SIDE) or {}
            if g.get("state") in ("held", "empty", "uncertain"):
                return g["state"], g.get("forces")
        return None, None

    def _way_back(self, plan):
        """The poses back to `wide` (or, once let go over the target, to `clear`), and whether
        the hand opens on the way: a held object goes back down where it was found first."""
        if self._released:
            return [plan.place.clear], [], False
        trail = self._trail
        at = [k for k, q in enumerate(trail) if q is plan.grasp]
        if not at:
            return trail[::-1], [], False
        return trail[at[-1]:][::-1], [plan.beside, plan.wide], self._closed

    def _back(self, here, poses):
        while poses and np.abs(poses[0] - here).max() < 0.03:
            poses = poses[1:]                     # already there
        if not poses:
            return here
        self.emit("backing out")
        self._move(here, poses, vel=APPROACH_VEL, what="backing out")
        return poses[-1]

    def _go_home(self, plan, q0):
        """Back the way the arm went out, the hand folded, then let go with it still folded.
        If the relay already took the arm to safety, it releases on its own."""
        if self.session is None:
            return
        try:
            st = self._op(op="status")
            if not st.get("retreat") and st.get("ok", True) and plan is not None:
                cancelled = self.cancelled.is_set()
                self.cancelled.clear()            # the way home itself must not be cancelled
                here = self.ex.measured_q()
                to_grasp, then, open_there = self._way_back(plan)
                here = self._back(here, to_grasp)
                if open_there:
                    self.emit("putting it down", "where it was")
                    self._open()
                here = self._back(here, then)
                self.emit("going home", "folding the hand, then down the taught way")
                self._fist()
                self._move(here, [plan.straight, *plan.way_up[::-1], q0], what="going home")
                if cancelled:
                    self.cancelled.set()
            self._op(op="keep_hands")
        except Exception as exc:                 # the release below still runs
            self.emit("home failed", str(exc))
        finally:
            self._alive.set()
            self._op(op="release")
            self.emit("done", self.state)


# --------------------------------------------------------------------------- #
# The registry the service uses: one action at a time per robot
# --------------------------------------------------------------------------- #
_ACTIONS = {}
_CURRENT = {"id": None}


def start_pick(phrase, **kw):
    """Start a PickAction (with `into=`, a place). One at a time: ActionError if one runs."""
    cur = _ACTIONS.get(_CURRENT["id"])
    if cur is not None and cur.state == "running":
        raise ActionError(f"another action is running: {cur.snapshot()['verb']} '{cur.phrase}'")
    act = PickAction(phrase, **kw).start()
    _ACTIONS[act.id] = act
    _CURRENT["id"] = act.id
    return act


def get(action_id):
    return _ACTIONS.get(action_id)


def cancel_all():
    for act in _ACTIONS.values():
        if act.state == "running":
            act.cancel()
