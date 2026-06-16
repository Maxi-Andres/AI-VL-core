#!/usr/bin/env python3
"""
yolo_scan.py — Quick YOLO check on a single image (one "scan").

The YOLO counterpart of vlm_scan.py: instead of asking an Ollama VLM, it runs an
Ultralytics YOLO detector in-process on one image and prints the same
{"objects": [...]} JSON contract (type/description/reading/confidence/bbox) plus
latency. By default it ALSO saves a copy of the image with the detected boxes
drawn on it (under results/annotated/) so you can eyeball whether YOLO "hit".

This is the detector that, in the real deployment, runs on the live video stream
(here: one still image; video needs a WebSocket + front/back-end, out of scope).

Don't want to type flags? Run `python3 menu.py` (interactive menu) and pick YOLO.

Requirements:  pip install ultralytics
Usage:
    python3 src/yolo_scan.py fotos/clean/2.jpeg
    python3 src/yolo_scan.py fotos/clean/2.jpeg --model yolo11n.pt
    python3 src/yolo_scan.py fotos/clean/2.jpeg --conf 0.4 --imgsz 1280
    python3 src/yolo_scan.py fotos/clean/2.jpeg --no-save        # skip the annotated image
"""
import argparse
import os
import sys

from yolo_common import (
    KNOWN_MODELS,
    default_annotated_path,
    load_config,
    render_result,
    run_detection,
)


def run_yolo_scan(image, model, conf=0.25, imgsz=640, device=None, save=True,
                  save_path=None, classes=None):
    """Run a YOLO scan on a single image and print the result.

    If `save` is True, the annotated (boxed) image is written to `save_path`
    (default: results/annotated/<image>__<model>.jpg). `classes` is a list of
    class NAMES to keep (None/empty = detect everything). Returns True on success,
    False on a handled error (missing image / missing ultralytics / inference).
    """
    if not os.path.exists(image):
        print(f"[ERROR] Image not found: {image}", file=sys.stderr)
        return False

    annotate_path = None
    if save:
        annotate_path = save_path or default_annotated_path(image, model)

    filt = ", ".join(classes) if classes else "ALL classes"
    print(f"[..] Running YOLO '{model}' on '{image}' "
          f"(conf={conf}, imgsz={imgsz}, filter={filt}) ...")
    try:
        res = run_detection(model, image, conf=conf, imgsz=imgsz, device=device,
                            annotate_path=annotate_path, classes=classes)
    except ImportError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return False
    except Exception as e:  # ultralytics raises plain Exceptions on bad weights/etc.
        print(f"[ERROR] YOLO inference failed: {e}", file=sys.stderr)
        return False

    render_result(model, res)
    return True


def main():
    cfg = load_config()  # defaults come from config.json (yolo_* keys)
    ap = argparse.ArgumentParser(description="YOLO scan on a single image.")
    ap.add_argument("image", nargs="?", default=cfg["yolo_image"],
                    help="Path to the image. Default: the one in config.json")
    ap.add_argument("--model", default=cfg["yolo_model"],
                    help=f"YOLO weights (.pt). Catalogue: {', '.join(KNOWN_MODELS[:6])}, ... "
                         "Auto-downloaded on first use.")
    ap.add_argument("--conf", type=float, default=cfg.get("yolo_conf", 0.25),
                    help="Confidence threshold (drop detections below it)")
    ap.add_argument("--imgsz", type=int, default=cfg.get("yolo_imgsz", 640),
                    help="Inference image size (longer side); bigger = more detail, slower")
    ap.add_argument("--device", default=None,
                    help="Force device (e.g. 'cpu', '0'). Default: ultralytics auto-picks.")
    ap.add_argument("--classes", nargs="+", default=cfg.get("yolo_classes") or None,
                    metavar="NAME",
                    help="Only detect these class names (e.g. --classes person cup). "
                         "Omit = detect all classes.")
    ap.add_argument("--save", dest="save", action="store_true",
                    default=cfg.get("yolo_save", True),
                    help="Save the annotated (boxed) image (default ON)")
    ap.add_argument("--no-save", dest="save", action="store_false",
                    help="Do NOT save the annotated image")
    ap.add_argument("--save-path", default=None,
                    help="Where to write the annotated image. "
                         "Default: results/annotated/<image>__<model>.jpg")
    args = ap.parse_args()

    ok = run_yolo_scan(args.image, args.model, conf=args.conf, imgsz=args.imgsz,
                       device=args.device, save=args.save, save_path=args.save_path,
                       classes=args.classes)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
