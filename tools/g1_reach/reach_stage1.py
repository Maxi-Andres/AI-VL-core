"""Stage 1 of the first computed reach: raise near the body, extend to the pre-grasp, hold for a
picture, come back the same way, release. Refuses to move if the bottle readings disagree or
if any point of the path (hand, wrist, elbow) passes the table edge too low.

    python reach_stage1.py            # plan + checks only
    python reach_stage1.py --go       # and move
"""
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import cv2
import numpy as np

import arm_ik as ik
import locate_object as lo
import perception3d as p3

EXEC = "http://127.0.0.1:8090"
RIGHT = [22, 23, 24, 25, 26, 27, 28]
NAMES = ik.joint_names("right")
CLEAR_M = 0.04          # every arm point past the table edge stays this far above the table


def post(body):
    req = urllib.request.Request(f"{EXEC}/joints/control?robot=g1", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "http": exc.code, **json.loads(exc.read() or b"{}")}


def measured_q():
    with urllib.request.urlopen(f"{EXEC}/joints/stream?robot=g1", timeout=5) as r:
        for line in r:
            d = json.loads(line)
            if d.get("q"):
                return np.array([d["q"][i] for i in RIGHT])


def locate_three():
    cal = p3.load_calibration(lo.CALIBRATION)
    runs = []
    for _ in range(3):
        z, _info = lo.fetch_depth(frames=5)
        frame = lo.grab_color()
        box, how, _ = lo.choose(frame, ["bottle", "cup"], "the water bottle")
        if box is None:
            continue
        h, w = frame.shape[:2]
        r = p3.locate(z, box, cal, color_size=(w, h))
        if r.get("ok") and r["method"] == "table":
            runs.append(r)
    if len(runs) < 3:
        raise SystemExit(f"only {len(runs)} of 3 locates found the bottle standing on the table")
    base = np.median([r["torso"] for r in runs], axis=0)
    spread = max(np.linalg.norm(np.array(r["torso"]) - base) for r in runs)
    height = float(np.median([r["height_m"] for r in runs]))
    if spread > 0.015:
        raise SystemExit(f"the three locates disagree by {spread * 100:.1f} cm: not moving")
    # the table: its top around the base height, and its near edge in front of the robot
    pts = p3.depth_points(z, cal.depth, step=2)
    T = (p3.rpy_matrix(*p3.D435_RPY) @ (p3.OPTICAL_TO_LINK @ pts.T)).T + np.array(p3.D435_XYZ)
    top = T[(np.abs(T[:, 2] - base[2]) < 0.015) & (np.abs(T[:, 1] - base[1]) < 0.25)]
    edge_x = float(np.percentile(top[:, 0], 2))
    return base, height, spread, edge_x, frame


def arm_points(q):
    """Elbow, wrist and hand positions (torso frame) for a pose."""
    p, _rot, joints = ik.forward("right", q)
    return [joints[3][0], joints[4][0], joints[6][0], p]


def check_path(segments, table_z, edge_x):
    worst = np.inf
    for qa, qb in segments:
        for s in np.linspace(0, 1, 25):
            for pt in arm_points(qa + s * (qb - qa)):
                if pt[0] > edge_x - 0.05:
                    worst = min(worst, pt[2] - table_z)
    return worst


def run_trajectory(session, points):
    out = post({"op": "trajectory", "session": session, "points": points})
    if not out.get("ok"):
        raise RuntimeError(f"trajectory refused: {out}")
    while True:
        time.sleep(0.3)
        st = post({"op": "status", "session": session})
        tr = st.get("trajectory") or {}
        if tr.get("state") != "running":
            return tr


def timed(q_from, poses, vel_fraction=0.4):
    pts, t, prev = [], 0.0, q_from
    for q in poses:
        t += max(2.0, float(np.abs(q - prev).max()) / (ik.MAX_VEL * vel_fraction))
        pts.append({"t": round(t, 2), "q": {n: round(float(v), 4) for n, v in zip(NAMES, q)}})
        prev = q
    return pts


def main(go):
    q0 = measured_q()
    base, height, spread, edge_x, frame = locate_three()
    table_z = base[2]
    up = np.array([0.0, 0.0, 1.0])
    grasp = base + height / 2 * up
    approach = ik.approach_direction("right", grasp)
    palm = np.cross(up, approach)
    pre = grasp - ik.PRE_GRASP_M * approach
    raise_pt = np.array([min(pre[0], edge_x - 0.12), (pre[1] - 0.20) / 2, table_z + 0.15])
    q_raise = ik.solve("right", raise_pt, q0, finger_dir=approach, palm_dir=palm)
    q_pre = ik.solve("right", pre, q_raise, finger_dir=approach, palm_dir=palm)
    clear = check_path([(q0, q_raise), (q_raise, q_pre)], table_z, edge_x)
    print(f"bottle base {np.round(base, 3)} height {height:.3f} (3 locates within {spread * 100:.1f} cm)")
    print(f"table z {table_z:.3f}, edge x {edge_x:.3f}; hand now {np.round(ik.forward('right', q0)[0], 3)}")
    print(f"raise {np.round(raise_pt, 3)} orient {ik.orientation_error_deg('right', q_raise, approach, palm)}")
    print(f"pre   {np.round(pre, 3)} orient {ik.orientation_error_deg('right', q_pre, approach, palm)}")
    print(f"lowest arm point past the edge: {clear * 100:.1f} cm above the table (need {CLEAR_M * 100:.0f})")
    out_pts = timed(q0, [q_raise, q_pre])
    back_pts = timed(q_pre, [q_raise, q0])
    print("out", [p["t"] for p in out_pts], "back", [p["t"] for p in back_pts])
    if clear < CLEAR_M:
        raise SystemExit("the path passes too close to the table: not moving")
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
    try:
        post({"op": "open", "session": session, "side": "right"})
        time.sleep(1.0)
        r1 = run_trajectory(session, out_pts)
        print("out:", r1)
        if r1.get("state") == "success":
            time.sleep(1.0)
            f = lo.grab_color()
            cv2.imwrite("at_pre_grasp.jpg", f)
            print("hand at", np.round(ik.forward("right", measured_q())[0], 3), "- picture saved")
            time.sleep(4.0)
        r2 = run_trajectory(session, back_pts)
        print("back:", r2)
    finally:
        alive.set()
        print("release:", post({"op": "release", "session": session}).get("ok"))


if __name__ == "__main__":
    main("--go" in sys.argv)
