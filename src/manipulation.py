#!/usr/bin/env python3
"""
manipulation.py — "pick up the bottle" as an ACTION: started, followed step by step, cancelled.

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

The object is any phrase locate_object can resolve (YOLO finds, the VLM chooses). What is
specific is the grasp: from the side, the right hand, an upright object on a table.
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
APPROACH_VEL = 0.4       # of the relay's max speed
GRASP_WAIT_S = 6.0
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

    def joints(self):
        """One line of the joint stream (q, temp, hands)."""
        with urllib.request.urlopen(f"{self.url}/joints/stream?robot={self.robot}",
                                    timeout=5) as r:
            for line in r:
                d = json.loads(line)
                if d.get("q"):
                    return d
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


def locate(phrase, classes=("bottle", "cup"), runs=LOCATE_RUNS, frames=5):
    """The object standing on the table, located `runs` times; ActionError if they disagree or
    it is not found standing on a table."""
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
    if spread > LOCATE_AGREE_M:
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
                 edge_x=float(np.percentile(top[:, 0], 2)), chosen=chosen)


# --------------------------------------------------------------------------- #
# Planning, and checking the plan
# --------------------------------------------------------------------------- #
def taught_poses(path=TAUGHT_POSES):
    with open(path) as fh:
        poses = json.load(fh)
    names = ik.joint_names(SIDE)
    return {p["name"]: np.array([p["q"][n] for n in names]) for p in poses}


def arm_points(q, tip=HAND_TIP):
    """Elbow, wrist, grasp point and the hand's far end (torso frame) for a pose."""
    p, _rot, joints = ik.forward(SIDE, q)
    tips = ik.forward(SIDE, q, tool=(tip, 0.0, 0.0))[0]
    return [joints[3][0], joints[4][0], joints[6][0], p, tips]


def table_margin(qa, qb, table_z, edge_x, tip=HAND_TIP, steps=25):
    """The lowest any arm point gets above the table while past its edge, along qa -> qb
    (joint-space, as the relay interpolates). inf if nothing is ever over the table."""
    worst = np.inf
    for s in np.linspace(0, 1, steps):
        for pt in arm_points(qa + s * (qb - qa), tip):
            if pt[0] > edge_x:
                worst = min(worst, pt[2] - table_z)
    return worst


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
class Plan:
    way_up: list              # taught postures, hanging -> ready
    straight: np.ndarray      # ready with the wrist straightened, the forearm raised: SAFE pose
    wide: np.ndarray
    beside: np.ndarray
    grasp: np.ndarray
    lifted: np.ndarray
    margins: dict = field(default_factory=dict)
    gap: float = np.inf


def plan_pick(q0, scene, taught):
    """The joint postures of a pick, checked. ActionError with the reason if any check fails."""
    up = np.array([0.0, 0.0, 1.0])
    target = scene.base + GRASP_AT * scene.height * up
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
    worst = min(plan.margins.values())
    if worst < CLEAR_M:
        seg = min(plan.margins, key=plan.margins.get)
        raise ActionError(f"the arm would pass {worst * 100:.1f} cm above the table ({seg})")
    if plan.gap < OBJECT_GAP_M:
        raise ActionError(f"the hand would pass {plan.gap * 100:.1f} cm from the object")
    return plan


# --------------------------------------------------------------------------- #
# The action
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    pass


class PickAction:
    """Pick up `phrase`, lift it, put it back, go home — on its own thread, step by step.

    snapshot() is what the agent and the UI read: state (running | success | failure |
    uncertain | cancelled), the step running now, every event so far, and the result."""

    def __init__(self, phrase, executor=None, locate_fn=locate, taught=None, hold=True):
        self.id = uuid.uuid4().hex[:12]
        self.phrase = phrase
        self.ex = executor or Executor()
        self.locate_fn = locate_fn
        self.taught = taught
        self.hold = hold
        self.state, self.step, self.result = "running", "starting", None
        self.events = []
        self.cancelled = threading.Event()
        self.session = None
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
            return {"id": self.id, "verb": "pick", "object": self.phrase, "state": self.state,
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

    def _move(self, q_from, poses, vel=0.4, what="moving"):
        pts, t, prev = [], 0.0, q_from
        names = ik.joint_names(SIDE)
        for q in poses:
            t += max(2.0, float(np.abs(q - prev).max()) / (ik.MAX_VEL * vel))
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

    def _keepalive(self):
        while not self._alive.is_set():
            self._op(op="keepalive")
            time.sleep(0.3)

    # -- the procedure --------------------------------------------------- #
    def _run(self):
        stage = None            # how far out the arm got: what the way home has to undo
        plan = q0 = None
        try:
            self.emit("looking", f"finding '{self.phrase}'")
            scene = self.locate_fn(self.phrase)
            self.emit("found", f"{scene.chosen}: base at x {scene.base[0]:.2f} y "
                               f"{scene.base[1]:+.2f} m, {scene.height * 100:.0f} cm tall")
            self._check()
            q0 = self.ex.measured_q()
            plan = plan_pick(q0, scene, self.taught or taught_poses())
            self.emit("planned", f"{min(plan.margins.values()) * 100:.0f} cm over the table at "
                                 f"worst, the hand {plan.gap * 100:.0f} cm from it on the way in")
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
            stage = "up"
            self._move(q0, [*plan.way_up, plan.straight, plan.wide], what="raising the arm")
            stage = "wide"
            self._check()
            self.emit("opening the hand", "the arm still, away from the object")
            self._hand([0, 0, 0, 0, 0, 0])
            time.sleep(1.0)
            self._hand(OPEN_WIDE)
            time.sleep(1.5)
            self._move(plan.wide, [plan.beside], what="coming beside it")
            stage = "beside"
            self._check()
            self.emit("reaching in")
            self._move(plan.beside, [plan.grasp], vel=APPROACH_VEL, what="reaching in")
            stage = "grasp"
            self.emit("closing", "the fingers, then the thumb by touch")
            self._hand({2: 1000, 3: 1000, 4: 1000, 5: 1000})
            time.sleep(1.2)
            self._op(op="grasp", side=SIDE, thumb=OPEN_WIDE[1])
            state, forces = self._wait_grasp()
            held = state == "held" and forces and sum(f or 0 for f in forces) > 0.5
            self.emit("grasp " + ("held" if held else state or "unknown"),
                      "fingertips " + ", ".join("-" if f is None else f"{f:.1f} N"
                                                for f in (forces or [])))
            if held and self.hold:
                self._move(plan.grasp, [plan.lifted], vel=APPROACH_VEL, what="lifting")
                self.emit("lifted", f"{LIFT_M * 100:.0f} cm")
                time.sleep(HOLD_S)
                self._move(plan.lifted, [plan.grasp], vel=APPROACH_VEL, what="putting it down")
                self.emit("put down")
            self._op(op="open", side=SIDE)
            time.sleep(1.5)
            self.result = {"held": bool(held), "forces": forces}
            self.state = "success" if held else "failure"
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
            self._go_home(plan, q0, stage)

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

    def _go_home(self, plan, q0, stage):
        """Back the way the arm went out, the hand folded, then let go with it still folded.
        If the relay already took the arm to safety, it releases on its own."""
        if self.session is None:
            return
        try:
            st = self._op(op="status")
            if not st.get("retreat") and st.get("ok", True) and plan is not None:
                cancelled = self.cancelled.is_set()
                self.cancelled.clear()            # the way home itself must not be cancelled
                back = {"grasp": [plan.beside, plan.wide], "beside": [plan.wide],
                        "wide": [], "up": []}.get(stage, [])
                here = self.ex.measured_q()
                if back:
                    self.emit("backing out")
                    self._move(here, back, vel=APPROACH_VEL, what="backing out")
                    here = back[-1]
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
    cur = _ACTIONS.get(_CURRENT["id"])
    if cur is not None and cur.state == "running":
        raise ActionError(f"another action is running: pick '{cur.phrase}'")
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
