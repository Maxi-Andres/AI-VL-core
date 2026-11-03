"""Replay the 2026-11-03 11:47 hand-eye motion exactly (same points, same IK, hand not opened),
without its measurements: for watching how the arm moves (it raised the shoulder, not the
elbow). Takes the table from the earlier measurement: --table Z --edge X.

    python replay_hand_eye.py --table -0.028 --edge 0.318 [--go]
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from reach_stage1 import CLEAR_M, check_path, measured_q, post, run_trajectory, timed

import arm_ik as ik


def main():
    table_z = float(sys.argv[sys.argv.index("--table") + 1])
    edge_x = float(sys.argv[sys.argv.index("--edge") + 1])
    q0 = measured_q()
    raise_pt = np.array([0.10, -0.18, table_z + 0.20])
    targets = [np.array([0.22, -0.24, table_z + 0.16]), np.array([0.18, -0.20, table_z + 0.22]),
               np.array([0.25, -0.30, table_z + 0.13])]
    fd = np.array([1.0, 0.0, 0.0])
    palm = np.cross([0, 0, 1.0], fd)
    q_raise = ik.solve("right", raise_pt, q0, finger_dir=fd, palm_dir=palm)
    qs, prev = [], q_raise
    for t in targets:
        prev = ik.solve("right", t, prev, finger_dir=fd, palm_dir=palm)
        qs.append(prev)
    segs = [(q0, q_raise), (q_raise, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], q_raise)]
    clear = check_path(segs, table_z, edge_x - 0.10)
    names = ik.joint_names("right")
    print("q0     ", dict(zip(names, np.round(q0, 2), strict=False)))
    print("q_raise", dict(zip(names, np.round(q_raise, 2), strict=False)))
    print(f"lowest arm point past the edge {clear * 100:.1f} cm above the table")
    if clear < CLEAR_M or "--go" not in sys.argv:
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
        cur = q0
        for q in [q_raise, *qs]:
            tr = run_trajectory(session, timed(cur, [q]))
            print(tr.get("state"), tr.get("error_rad"))
            if tr.get("state") != "success":
                break
            cur = q
            time.sleep(2.5)
        print("back:", run_trajectory(session, timed(cur, [q_raise, q0])).get("state"))
    finally:
        alive.set()
        print("release:", post({"op": "release", "session": session}).get("ok"))


if __name__ == "__main__":
    main()
