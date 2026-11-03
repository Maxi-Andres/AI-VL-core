"""The first full grasp, the way the user laid it out (2026-11-03): come in to the RIGHT of the
bottle, open the fingers fully with the thumb turned perpendicular, then slide the hand sideways
until the bottle is in the palm, close by touch, lift, put it back, let go, and retreat the same
way. The hand is never changed while the arm moves.

    python grasp_side.py            # plan + checks only
    python grasp_side.py --go       # and move
"""
import itertools
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from reach_stage1 import CLEAR_M, check_path, locate_three, measured_q, post, run_trajectory, timed

import arm_ik as ik
import locate_object as lo

SIDE_M = 0.08            # the hand waits this far to the right of its grasp position
LIFT_M = 0.08
# Fingers fully open, the thumb's rotation turned perpendicular to them (the value that held the
# bottle on 2026-10-30): [thumb flex, thumb rotation, index, middle, ring, pinky], 0 open..1000.
OPEN_WIDE = [0, 950, 0, 0, 0, 0]
GRASP_WAIT_S = 6.0
START_MAX_C = 60
APPROACH_VEL = 0.4       # of the relay's max speed: the arm out costs heat by the second


def arm_temp():
    import json
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:8090/joints/stream?robot=g1", timeout=5) as r:
        for line in r:
            d = json.loads(line)
            if d.get("temp"):
                return max(d["temp"][i] for i in range(22, 29))
    return 0


def set_hand(session, fingers):
    for k, pos in enumerate(fingers):
        out = post({"op": "hand", "session": session, "side": "right", "finger": k, "pos": pos})
        if not out.get("ok"):
            raise RuntimeError(f"hand finger {k} refused: {out}")


def wait_grasp(session):
    deadline = time.monotonic() + GRASP_WAIT_S
    state = None
    while time.monotonic() < deadline:
        time.sleep(0.3)
        st = post({"op": "status", "session": session})
        if st.get("retreat"):
            return "retreat", None
        g = (st.get("grasp") or {}).get("right") or {}
        state = g.get("state")
        if state in ("held", "empty", "uncertain"):
            return state, g.get("forces")
    return state, None


def main(go):
    q0 = measured_q()
    base, height, spread, edge_x, _frame = locate_three()
    table_z = float(base[2])
    up = np.array([0.0, 0.0, 1.0])
    grasp = base + height / 2 * up
    approach = ik.approach_direction("right", grasp)
    palm = np.cross(up, approach)                 # towards the bottle, from the hand's side
    beside = grasp - SIDE_M * palm
    lifted = grasp + LIFT_M * up
    tuck, lift = ik.TUCK["right"], ik.LIFT["right"]
    q_beside = ik.solve("right", beside, lift, finger_dir=approach, palm_dir=palm)
    q_grasp = ik.solve("right", grasp, q_beside, finger_dir=approach, palm_dir=palm)
    q_lifted = ik.solve("right", lifted, q_grasp, finger_dir=approach, palm_dir=palm)
    path = [q0, tuck, lift, q_beside, q_grasp, q_lifted]
    clear = min(check_path([(a, b)], table_z, edge_x - 0.10) for a, b in itertools.pairwise(path))
    # The fingertips on the way in must pass the bottle by its side, never through it. They are
    # in the palm's plane, on the finger axis — not out by TOOL's offset from the palm.
    tip_tool = (ik.TOOL["right"][0] + ik.FINGERTIP_M, 0.0, 0.0)
    gap = min(np.linalg.norm((ik.forward("right", lift + s * (q_beside - lift), tool=tip_tool)[0]
                              - base)[:2]) for s in np.linspace(0, 1, 30))
    print(f"bottle {np.round(base, 3)} h {height:.3f} (3 within {spread * 100:.1f} cm); "
          f"table {table_z:.3f}, edge {edge_x:.3f}")
    for name, q, target in (("beside", q_beside, beside), ("grasp", q_grasp, grasp),
                            ("lifted", q_lifted, lifted)):
        print(f"{name:7s} {np.round(target, 3)} orientation "
              f"{ik.orientation_error_deg('right', q, approach, palm)} roll {q[1]:.2f}")
    print(f"lowest arm point past the edge {clear * 100:.1f} cm above the table; fingertips pass "
          f"the bottle's axis at {gap * 100:.1f} cm on the way in")
    if clear < CLEAR_M or gap < 0.06:
        raise SystemExit("too close to the table or the bottle: not moving")
    if not go:
        return
    # Cool enough to start: the shoulder heats ~1 C/s holding the arm out and cools as fast.
    while (t := arm_temp()) > START_MAX_C:
        print(f"right arm at {t:.0f} C: waiting to cool under {START_MAX_C} C")
        time.sleep(5)
    # The safe pose: the relay takes the arm back there before any release (warm joint, lost
    # client, release asked) instead of dropping it onto the table.
    took = post({"op": "take", "safe": dict(zip(ik.joint_names("right"),
                                                 [round(float(v), 4) for v in tuck],
                                                 strict=True))})
    if not took.get("ok"):
        raise SystemExit(f"take refused: {took}")
    if not took.get("safe"):
        print("WARNING: this relay does not know the safe pose (pull + restart it on PC2)")
    session = took["session"]
    alive = threading.Event()

    def keepalive():
        while not alive.is_set():
            post({"op": "keepalive", "session": session})
            time.sleep(0.3)
    threading.Thread(target=keepalive, daemon=True).start()
    held = False
    try:
        tr = run_trajectory(session, timed(q0, [tuck, lift]))
        print("to lift:", tr.get("state"), tr.get("error_rad"))
        if tr.get("state") != "success":
            return
        # The hand opens HERE, the arm still and far from the bottle: opened beside it, the
        # thumb's turn swept the bottle away (user, 2026-11-03). Fingers and thumb straight
        # first, the thumb up; only then is it turned perpendicular.
        set_hand(session, [0, 0, 0, 0, 0, 0])
        time.sleep(1.0)
        set_hand(session, OPEN_WIDE)
        time.sleep(1.5)
        tr = run_trajectory(session, timed(lift, [q_beside]))
        print("to beside:", tr.get("state"), tr.get("error_rad"))
        if tr.get("state") != "success":
            return
        cv2.imwrite("grasp_beside_open.jpg", lo.grab_color())
        tr = run_trajectory(session, timed(q_beside, [q_grasp], vel_fraction=APPROACH_VEL))
        print("slide in:", tr.get("state"), tr.get("error_rad"))
        if tr.get("state") != "success":
            return
        cv2.imwrite("grasp_slid_in.jpg", lo.grab_color())
        if "--measure" in sys.argv:
            # Hand and bottle in ONE depth snapshot: where they are relative to each other,
            # whatever the arm model or the bottle's location say. No grasp: back out.
            zz, _ = lo.fetch_depth(frames=5)
            np.save("slid_in_depth.npy", zz)
            np.save("slid_in_q.npy", measured_q())
            np.save("slid_in_plan.npy", np.concatenate([base, [height], grasp]))
            print("measured; backing out without a grasp")
            print("slide out:", run_trajectory(session, timed(q_grasp, [q_beside],
                                                              vel_fraction=APPROACH_VEL)).get("state"))
            return
        print("grasp:", post({"op": "grasp", "session": session, "side": "right",
                              "thumb": OPEN_WIDE[1]}).get("ok"))
        state, forces = wait_grasp(session)
        print("grasp state:", state, forces)
        cv2.imwrite("grasp_closed.jpg", lo.grab_color())
        held = state == "held"
        if held:
            tr = run_trajectory(session, timed(q_grasp, [q_lifted], vel_fraction=APPROACH_VEL))
            print("lift:", tr.get("state"))
            time.sleep(0.5)
            cv2.imwrite("grasp_lifted.jpg", lo.grab_color())
            time.sleep(0.5)
            print("put down:", run_trajectory(session, timed(q_lifted, [q_grasp],
                                                             vel_fraction=APPROACH_VEL)).get("state"))
        print("open:", post({"op": "open", "session": session, "side": "right"}).get("ok"))
        time.sleep(1.5)
        print("slide out:", run_trajectory(session, timed(q_grasp, [q_beside],
                                                          vel_fraction=APPROACH_VEL)).get("state"))
    finally:
        # The relay may already be taking the arm to the safe pose (a warm joint): then it
        # releases by itself, and a trajectory from here would only be refused.
        st = post({"op": "status", "session": session})
        if st.get("retreat"):
            print("the relay retreated to the safe pose:", st["retreat"].get("reason"))
            alive.set()
        else:
            try:
                cur = measured_q()
                print("back:", run_trajectory(session, timed(cur, [lift, tuck, q0])).get("state"))
            except RuntimeError as exc:
                print("back:", exc)
            finally:
                alive.set()
                print("release:", post({"op": "release", "session": session}).get("ok"))


if __name__ == "__main__":
    main("--go" in sys.argv)
