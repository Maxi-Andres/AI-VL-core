"""Reach the way the user posed it (2026-11-03): from hanging, tuck the hand in front of the belly
(upper arm back, elbow at 90 degrees), lift the forearm, then extend towards the bottle — and
stop SHORT of it (the hand has been landing ~8-10 cm further than the model). Hold for a
picture, come back the same way, release.

    python tuck_reach.py [--short 0.20] [--go]
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools

import cv2
import numpy as np
from reach_stage1 import (
    CLEAR_M,
    check_path,
    locate_three,
    measured_q,
    post,
    run_trajectory,
    timed,
)

import arm_ik as ik
import locate_object as lo

# The user's posture, read off the arm after posing it from the Joints page.
TUCK = np.array([0.573, -0.215, 0.044, -0.036, -0.055, -0.673, 0.773])
# The forearm lifted from there and the wrist straightened: clears the table edge.
LIFT = np.array([0.3, -0.215, 0.044, -0.6, 0.0, 0.0, 0.0])


def main():
    short = float(sys.argv[sys.argv.index("--short") + 1]) if "--short" in sys.argv else 0.20
    q0 = measured_q()
    base, height, spread, edge_x, _frame = locate_three()
    table_z = float(base[2])
    up = np.array([0.0, 0.0, 1.0])
    grasp = base + height / 2 * up
    approach = ik.approach_direction("right", grasp)
    palm = np.cross(up, approach)
    stop = grasp - short * approach
    q_stop = ik.solve("right", stop, LIFT, finger_dir=approach, palm_dir=palm)
    path = [q0, TUCK, LIFT, q_stop]
    # The edge 10 cm nearer than measured: the real hand lands that much further.
    clear = min(check_path([(a, b)], table_z, edge_x - 0.10) for a, b in itertools.pairwise(path))
    print(f"bottle {np.round(base, 3)} h {height:.3f} (3 within {spread * 100:.1f} cm); "
          f"table {table_z:.3f}, edge {edge_x:.3f}")
    print(f"stop {np.round(stop, 3)} ({short * 100:.0f} cm short), orientation "
          f"{ik.orientation_error_deg('right', q_stop, approach, palm)}, shoulder roll "
          f"{q_stop[1]:.2f}")
    print(f"lowest arm point past the edge: {clear * 100:.1f} cm above the table")
    if clear < CLEAR_M:
        raise SystemExit("too close to the table: not moving")
    if "--go" not in sys.argv:
        return
    took = post({"op": "take"})
    if not took.get("ok"):
        raise SystemExit(f"take refused: {took}")
    session = took["session"]
    alive = threading.Event()

    def keepalive():
        while not alive.is_set():
            post({"op": "keepalive", "session": session})
            time.sleep(0.3)
    threading.Thread(target=keepalive, daemon=True).start()
    try:
        # The hand is left as it is: opening it first turned the thumb out before the arm moved,
        # which the user does not want (2026-11-03).
        tr = run_trajectory(session, timed(q0, [TUCK, LIFT, q_stop]))
        print("out:", tr.get("state"), tr.get("error_rad"), tr.get("reason"))
        if tr.get("state") == "success":
            time.sleep(1.0)
            qm = measured_q()
            z, _ = lo.fetch_depth(frames=5)
            cv2.imwrite("tuck_reach_stop.jpg", lo.grab_color())
            np.save("tuck_reach_depth.npy", z)
            np.save("tuck_reach_q.npy", qm)
            print("hand (model)", np.round(ik.forward("right", qm)[0], 3), "- picture + depth saved")
            time.sleep(1.0)
        print("back:", run_trajectory(session, timed(q_stop, [LIFT, TUCK, q0])).get("state"))
    finally:
        alive.set()
        print("release:", post({"op": "release", "session": session}).get("ok"))


if __name__ == "__main__":
    main()
