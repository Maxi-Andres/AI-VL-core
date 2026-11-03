"""Hand-eye check: where the camera SEES the right hand against where the arm model says it is.

The hand goes (close to the body, then up) to three free points above the table, away from the
bottle; at each it holds still for a depth snapshot. The hand's points are the depth readings
near the model's hand that are not the table; their centroid against the model's hand centre is
the offset. The same offset at every point = a fixed error to subtract from every target.

    python hand_eye.py          # plan + checks only
    python hand_eye.py --go     # and move
"""
import json
import sys
import threading
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import cv2
import numpy as np

import arm_ik as ik
import locate_object as lo
import perception3d as p3

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from reach_stage1 import CLEAR_M, check_path, measured_q, post, run_trajectory, timed

CAL = p3.load_calibration(lo.CALIBRATION)
HAND_CENTRE = np.array([0.06, 0.0, 0.0])   # wrist_yaw link: the middle of the BrainCo's palm+fingers
SEARCH_M = 0.14


def torso_cloud(z):
    pts = p3.depth_points(z, CAL.depth, step=1)
    return (p3.rpy_matrix(*p3.D435_RPY) @ (p3.OPTICAL_TO_LINK @ pts.T)).T + np.array(p3.D435_XYZ)


def table_height(T):
    """The table top: the densest height among the points in front of the robot."""
    front = T[(T[:, 0] > 0.25) & (T[:, 0] < 0.6) & (np.abs(T[:, 1]) < 0.3)]
    hist, edges = np.histogram(front[:, 2], bins=np.arange(-0.4, 0.3, 0.005))
    k = int(hist.argmax())
    return float((edges[k] + edges[k + 1]) / 2)


def model_hand(q):
    p, rot, joints = ik.forward("right", q, tool=HAND_CENTRE)
    return p, rot


def seen_hand(T, centre, table_z):
    near = T[(np.linalg.norm(T - centre, axis=1) < SEARCH_M) & (T[:, 2] > table_z + 0.02)]
    if len(near) < 200:
        return None
    c = np.median(near, axis=0)
    near = near[np.linalg.norm(near - c, axis=1) < 0.09]      # drop the forearm's tail
    c = np.median(near, axis=0)
    _u, s, vt = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
    return c, vt[2], len(near)


def main(go):
    q0 = measured_q()
    z, _ = lo.fetch_depth(frames=5)
    T = torso_cloud(z)
    table_z = table_height(T)
    edge_x = float(np.percentile(T[np.abs(T[:, 2] - table_z) < 0.015][:, 0], 2))
    # Close to the body on the way up (user, 2026-11-03), then three free points.
    raise_pt = np.array([0.10, -0.17, table_z + 0.20])
    targets = [np.array([0.27, -0.20, table_z + 0.16]), np.array([0.22, -0.12, table_z + 0.22]),
               np.array([0.30, -0.26, table_z + 0.12])]
    fd = np.array([1.0, 0.0, 0.0])
    palm = np.cross([0, 0, 1.0], fd)
    q_raise = ik.solve("right", raise_pt, q0, finger_dir=fd, palm_dir=palm)
    qs, prev = [], q_raise
    for t in targets:
        prev = ik.solve("right", t, prev, finger_dir=fd, palm_dir=palm)
        qs.append(prev)
    segs = [(q0, q_raise), (q_raise, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], q_raise)]
    clear = check_path(segs, table_z, edge_x)
    print(f"table z {table_z:.3f}, edge x {edge_x:.3f}; lowest arm point past the edge "
          f"{clear * 100:.1f} cm above it (need {CLEAR_M * 100:.0f})")
    print("raise", np.round(raise_pt, 3), "targets", [np.round(t, 3).tolist() for t in targets])
    if clear < CLEAR_M:
        raise SystemExit("too close to the table: not moving")
    if not go:
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
    results = []
    try:
        cur = q0
        for k, q in enumerate([q_raise] + qs):
            tr = run_trajectory(session, timed(cur, [q]))
            if tr.get("state") != "success":
                print("stopped:", tr)
                break
            cur = q
            if k == 0:
                continue
            time.sleep(1.0)
            qm = measured_q()
            zz, _ = lo.fetch_depth(frames=5)
            cv2.imwrite(f"hand_{k}.jpg", lo.grab_color())
            pm, rot = model_hand(qm)
            seen = seen_hand(torso_cloud(zz), pm, table_z)
            if seen is None:
                print(f"point {k}: the camera does not see the hand near the model's")
                continue
            c, normal, n = seen
            off = c - pm
            results.append(off)
            palm_model = rot @ ik.PALM_NORMAL["right"]
            ang = np.degrees(np.arccos(min(1.0, abs(normal @ palm_model))))
            print(f"point {k}: model {np.round(pm, 3)} seen {np.round(c, 3)} offset "
                  f"{np.round(off, 3)} ({np.linalg.norm(off) * 100:.1f} cm, {n} pts); "
                  f"hand plane vs model palm {ang:.0f} deg")
        run_trajectory(session, timed(cur, [q_raise, q0]))
    finally:
        alive.set()
        print("release:", post({"op": "release", "session": session}).get("ok"))
    if results:
        r = np.array(results)
        print("mean offset", np.round(r.mean(axis=0), 3), "spread", np.round(r.std(axis=0), 3))
        json.dump({"offsets": r.tolist()}, open("hand_eye.json", "w"))


if __name__ == "__main__":
    main("--go" in sys.argv)
