"""Save the right arm's current posture (and hand) as a named taught pose.

    python capture_pose.py <name>      -> appended to taught_poses.json
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

import arm_ik as ik


def main(name):
    with urllib.request.urlopen("http://127.0.0.1:8090/joints/stream?robot=g1", timeout=5) as r:
        for line in r:
            d = json.loads(line)
            if d.get("q"):
                break
    names = ik.joint_names("right")
    q = np.array([d["q"][i] for i in range(22, 29)])
    _p, _rot, joints = ik.forward("right", q)
    tips = ik.forward("right", q, tool=(0.38, 0.0, 0.0))[0]
    print(dict(zip(names, np.round(q, 3), strict=True)))
    print("hand:", d.get("right"), "| wrist", np.round(joints[6][0], 3), "far end (long hand)",
          np.round(tips, 3))
    path = Path(__file__).with_name("taught_poses.json")
    data = json.loads(path.read_text()) if path.exists() else []
    data.append({"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "q": dict(zip(names, [round(float(v), 4) for v in q], strict=True)),
                 "hand": d.get("right")})
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"saved as {name}")


if __name__ == "__main__":
    main(sys.argv[1])
