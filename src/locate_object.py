#!/usr/bin/env python3
"""
locate_object.py — "where is the bottle?", answered in metres in the G1's torso frame.

    python3 src/locate_object.py                         # the most confident bottle/cup
    python3 src/locate_object.py --ask "the green bottle"
    python3 src/locate_object.py --ask "the bottle" --what ""   # the VLM alone, no YOLO

Each tool does what it is good at, and nothing else:

    YOLO   finds the objects and their boxes, to the pixel, in milliseconds. It knows 80 COCO
           classes and nothing about "the green one" or "the one on the left".
    VLM    understands the phrase. Given the picture with YOLO's boxes numbered on it, it says
           WHICH number — a choice, not a measurement. Only when YOLO found nothing does it
           draw the box itself (Qwen3-VL's bbox_2d, 0..1000 of each side, measured against
           YOLO's on the same photo 2026-11-03: within ~4 px).
    depth  measures. The box alone says where to look, the robot's depth snapshot says how far
           (perception3d: the depth points that land in the box -> the front surface -> torso).

The colour picture is the robot's own video (the RTSP the PC already receives), the depth one
snapshot through the executor (GET /camera/depth — on on the G1, off on the Go2). The two are
taken a fraction of a second apart, so the robot and the object must be still: this is for
"look, then reach", not for tracking.

Writes one JSON line to stdout and the picture with the chosen box to results/locate/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

import perception3d as p3
from vlm_common import LLAMACPP_HOST, PROJECT_ROOT, extract_json, results_path, stream_chat

EXECUTOR = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8090")
RTSP = os.environ.get("G1_RTSP", "rtsp://127.0.0.1:8554/g1")
CALIBRATION = os.environ.get("G1_D435_CALIBRATION",
                             os.path.join(PROJECT_ROOT, "config", "g1_d435_calibration.json"))


def fetch_depth(executor=EXECUTOR, frames=5):
    """(uint16 millimetres, info) from the robot's depth snapshot. RuntimeError with the
    robot's own reason when it refuses (snapshots off, no camera)."""
    r = requests.get(f"{executor}/camera/depth", params={"robot": "g1", "frames": frames},
                     timeout=15)
    if r.status_code != 200:
        try:
            why = r.json().get("error")
        except ValueError:
            why = r.text[:200]
        raise RuntimeError(f"depth snapshot: HTTP {r.status_code}: {why}")
    z = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    if z is None or z.dtype != np.uint16:
        raise RuntimeError("depth snapshot: not a 16-bit PNG")
    return z, json.loads(r.headers.get("X-Depth-Info", "{}"))


def grab_color(url=RTSP, settle=25):
    """The newest colour frame. RTSP hands out its buffered frames first, so `settle` are
    read and dropped: the one kept is from now, not from when the stream was opened."""
    cap = cv2.VideoCapture(url)
    frame = None
    try:
        for _ in range(settle):
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        cap.release()
    if frame is None:
        raise RuntimeError(f"no colour frame from {url}")
    return frame


def yolo_boxes(frame, classes, model="yolo11m.pt", conf=0.35):
    """[(box in pixels, label, confidence)] for the given COCO class names."""
    from yolo_common import detect
    h, w = frame.shape[:2]
    out = detect(model, frame, conf=conf, classes=classes)
    return [((o["bbox"][0] * w, o["bbox"][1] * h, o["bbox"][2] * w, o["bbox"][3] * h),
             o["type"], o["confidence"]) for o in out["objects"]]


def _jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def numbered(frame, boxes):
    img = frame.copy()
    for i, (box, _label, _conf) in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img, str(i), (x1 + 4, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, str(i), (x1 + 4, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return img


def _ask_vlm(img, prompt, max_tokens=64):
    payload = {"model": "", "stream": True, "format": "json",
               "options": {"temperature": 0, "num_predict": max_tokens},
               "messages": [{"role": "user", "content": prompt, "images": [_jpeg_b64(img)]}]}
    content, *_ = stream_chat(payload, url=LLAMACPP_HOST, timeout=60)
    parsed, ok = extract_json(content)
    return parsed if ok and isinstance(parsed, dict) else {}


def vlm_choose(frame, boxes, phrase):
    """The index into `boxes` the VLM says is `phrase`, or None if it says none is."""
    prompt = (f"The yellow numbered boxes mark the objects found in this picture. Which number "
              f"is: {phrase}? Answer only JSON: {{\"number\": N}}, or {{\"number\": null}} "
              f"if no box is it.")
    n = _ask_vlm(numbered(frame, boxes), prompt).get("number")
    if isinstance(n, int) and 1 <= n <= len(boxes):
        return n - 1
    return None


def vlm_box(frame, phrase):
    """The box the VLM draws for `phrase` itself, in pixels, or None. Qwen3-VL answers in
    0..1000 of each side, whatever the picture's size (module docstring)."""
    got = _ask_vlm(frame, f"Locate: {phrase}. Answer only JSON: "
                          f"{{\"bbox_2d\": [x1, y1, x2, y2]}}, or {{\"bbox_2d\": null}} "
                          f"if it is not in the picture.")
    bb = got.get("bbox_2d")
    if not (isinstance(bb, list) and len(bb) == 4
            and all(isinstance(v, (int, float)) for v in bb)):
        return None
    h, w = frame.shape[:2]
    return (bb[0] / 1000 * w, bb[1] / 1000 * h, bb[2] / 1000 * w, bb[3] / 1000 * h)


def choose(frame, classes, phrase):
    """(box, how it was chosen, candidates) — or (None, why, candidates)."""
    boxes = yolo_boxes(frame, classes) if classes else []
    if boxes and phrase:
        # Asked even with ONE box: "the green bottle" with only a red one in sight is a no.
        i = vlm_choose(frame, boxes, phrase)
        if i is not None:
            return boxes[i][0], f"yolo {boxes[i][1]} #{i + 1}, chosen by the vlm", boxes
    if boxes and not phrase:
        best = max(boxes, key=lambda b: b[2])
        return best[0], f"yolo {best[1]} {best[2]:.2f}, the most confident", boxes
    if phrase:
        box = vlm_box(frame, phrase)
        if box:
            return box, "vlm bbox (yolo had no match)", boxes
    return None, "nothing found", boxes


def annotate(frame, box, result, path):
    img = frame.copy()
    if box:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        sx1, sy1, sx2, sy2 = (int(v) for v in p3.shrink_box(box))
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 160, 0), 1)
    text = (f"torso x{result['torso'][0]:+.3f} y{result['torso'][1]:+.3f} "
            f"z{result['torso'][2]:+.3f} m  ({result['calibration']})"
            if result.get("ok") else result.get("error", "?"))
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ask", default="", help="what to find, in words (the VLM picks it)")
    ap.add_argument("--what", default="bottle,cup",
                    help="YOLO class names to look for, comma-separated ('' = VLM only)")
    ap.add_argument("--frames", type=int, default=5, help="depth frames to median (1-15)")
    ap.add_argument("--calibration", default=CALIBRATION)
    args = ap.parse_args()

    calib = p3.load_calibration(args.calibration)
    t0 = time.monotonic()
    z, info = fetch_depth(frames=args.frames)
    frame = grab_color()
    classes = [c.strip() for c in args.what.split(",") if c.strip()]
    box, how, candidates = choose(frame, classes, args.ask)
    if box is None:
        result = {"ok": False, "error": how}
    else:
        h, w = frame.shape[:2]
        result = p3.locate(z, box, calib, color_size=(w, h), scale_m=info.get("scale_m", 0.001))
    result.update({"chosen": how, "box": [round(v) for v in box] if box else None,
                   "candidates": len(candidates), "depth_valid": info.get("valid"),
                   "seconds": round(time.monotonic() - t0, 2)})
    path = results_path(os.path.join("locate", time.strftime("%Y%m%d-%H%M%S") + ".jpg"))
    annotate(frame, box, result, path)
    result["picture"] = path
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
