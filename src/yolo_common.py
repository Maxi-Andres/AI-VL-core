#!/usr/bin/env python3
"""
yolo_common.py — Core shared by the YOLO menu, smoke test and benchmark.

This is the YOLO sibling of vlm_common.py. The PoC has two detection paths:
  - the VLM (vlm_common.py): an Ollama-served vision-language model that reasons
    over a prompt and returns JSON. Slow but flexible / open-vocabulary.
  - YOLO (this module): an Ultralytics detector that runs IN-PROCESS (no HTTP
    server, no Ollama) and returns boxes in milliseconds. This is what the real
    deployment runs on the live video stream; the VLM is only invoked when the
    user asks a question about a frame.

For now this only does SINGLE IMAGES (the live-video path needs a WebSocket and a
front/back-end that are out of scope here). It deliberately mirrors vlm_common so
the menu/smoke/benchmark feel identical across both paths.

Key differences vs the VLM path:
  - There is no `url`/host: YOLO loads a local `.pt` weights file (ultralytics
    auto-downloads pretrained weights on first use into the working directory).
  - There are no prompts/scopes/think: the "model" IS the detector, and you tune
    `conf` (confidence threshold) and `imgsz` (inference image size) instead of
    max_tokens/num_ctx.
  - It still emits the SAME JSON contract as the VLM ({"objects": [...]} with
    type/description/reading/confidence/bbox) so the downstream VLM->VLA consumer
    (Silk AI Proxy Gateway / F1.9) sees one shape regardless of which detector ran.
    For YOLO `type`/`description` are the class label and `reading` is always null
    (a plain detector reads no instrument value).

`ultralytics` is imported LAZILY (inside the functions that need it) so the VLM
half of the project keeps working even when ultralytics is not installed.
"""
import os
import time
from datetime import datetime

# Reuse everything path/config/UI-related from the VLM core so there is a single
# config.json, a single results/ folder and identical progress bars/timers.
from vlm_common import (  # noqa: F401  (re-exported for the YOLO entry points)
    PROJECT_ROOT,
    as_list,
    dedup,
    fmt_secs,
    image_size,
    list_images,
    load_config as _load_base_config,
    natural_key,
    pctl,
    progress_bar,
    results_path,
    save_config,
)


# --------------------------------------------------------------------------- #
# Lazy ultralytics import
# --------------------------------------------------------------------------- #
_YOLO_CLASS = None


def load_ultralytics():
    """Import and return the ultralytics `YOLO` class (cached).

    Raises a clear, actionable error if the package is missing — this is the only
    extra dependency the YOLO path needs and it is NOT installed by default.
    """
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        try:
            from ultralytics import YOLO  # heavy import; only when actually used
        except ImportError as e:
            raise ImportError(
                "The YOLO path needs the 'ultralytics' package, which is not "
                "installed. Install it with:  pip install ultralytics\n"
                "(it pulls in torch/opencv; the VLM path does NOT need it)."
            ) from e
        _YOLO_CLASS = YOLO
    return _YOLO_CLASS


def ultralytics_available():
    """True if `ultralytics` can be imported (so menus can hint at the install)."""
    try:
        load_ultralytics()
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Ultralytics has no "list installed models" endpoint like Ollama's /api/tags.
# Pretrained weights are just `.pt` files that ultralytics auto-downloads on first
# use. So we offer a curated catalogue of common pretrained detectors (the menu
# also scans the project for any local `*.pt`). All of these are COCO-pretrained
# 80-class detectors; the real PoC would later swap in a custom-trained `.pt`.
#
# Suffix = size/accuracy trade-off: n(ano) < s(mall) < m(edium) < l(arge) < x.
# Nano is the fastest and the natural default for a live stream.
KNOWN_MODELS = [
    # YOLO11 (latest generation)
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLOv8 (widely used, stable)
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
]

# The 80 COCO classes, in the canonical id order every pretrained YOLO uses. This
# is only a FALLBACK so the menu can list/pick classes without loading a model
# (or when ultralytics is not installed). The authoritative list is the loaded
# model's own `.names` (a custom-trained `.pt` may have different classes), which
# `class_names()` reads when it can.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def class_names(model_name):
    """Return the ordered list of class names a model can detect.

    Reads the loaded model's own `.names` (authoritative — works for custom
    weights too). Falls back to the 80 COCO classes if the model can't be loaded
    (e.g. ultralytics not installed, or no network for the first download), so the
    menu can still offer a class filter to configure.
    """
    try:
        model = load_model(model_name)
        names = model.names or {}
        if names:
            return [names[k] for k in sorted(names)]
    except Exception:
        pass
    return list(COCO_CLASSES)


def resolve_class_ids(model, classes):
    """Map a list of class NAMES to the model's class IDS for `predict(classes=)`.

    `classes` None/empty -> None (no filter, detect everything). Names not present
    in the model are silently skipped; if nothing matches, returns None (rather
    than an empty filter that would detect nothing).
    """
    if not classes:
        return None
    names = getattr(model, "names", None) or {}
    name_to_id = {v: k for k, v in names.items()}
    ids = [name_to_id[c] for c in classes if c in name_to_id]
    return ids or None


def list_models(extra_dirs=None):
    """List candidate YOLO weights: any local `*.pt` first, then the catalogue.

    Local `.pt` files (already downloaded or custom-trained) are listed first so
    they are easy to pick; the curated KNOWN_MODELS follow. Deduplicated, order
    preserved.
    """
    found = []
    dirs = [PROJECT_ROOT, os.getcwd()] + list(extra_dirs or [])
    seen_dirs = set()
    for d in dirs:
        d = os.path.abspath(d)
        if d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".pt"):
                found.append(fn)
    # Merge local finds + catalogue, dedup preserving order.
    seen, out = set(), []
    for name in found + KNOWN_MODELS:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


_MODEL_CACHE = {}


def _resolve_weights(model_name):
    """Map a bare weights name to a real file if we already have it.

    ultralytics resolves a bare name (e.g. 'yolo11s.pt') relative to the CURRENT
    working directory and re-downloads it if missing. When the server runs from a
    different folder than the repo, that means re-downloading weights that already
    live in the repo root. So if the name isn't a path that exists, but a file of
    that name sits in PROJECT_ROOT, use that absolute path instead. Otherwise hand
    the name back unchanged and let ultralytics download it as before.
    """
    if os.path.exists(model_name):
        return model_name
    cand = os.path.join(PROJECT_ROOT, model_name)
    return cand if os.path.exists(cand) else model_name


def load_model(model_name):
    """Load (and cache) a YOLO model by weights name/path.

    The first load of a catalogue name (e.g. 'yolov8n.pt') triggers an automatic
    download by ultralytics (unless the file already exists in PROJECT_ROOT, in
    which case we reuse it — see _resolve_weights); afterwards it is cached
    in-process so the benchmark does not reload it for every run.
    """
    if model_name not in _MODEL_CACHE:
        YOLO = load_ultralytics()
        _MODEL_CACHE[model_name] = YOLO(_resolve_weights(model_name))
    return _MODEL_CACHE[model_name]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def result_to_objects(result):
    """Convert one ultralytics Results object into our JSON contract objects.

    Boxes come back already-normalized via `xyxyn` (0..1 of the source image), so
    no manual normalization is needed (unlike the VLM, which returns pixels).
    Each object matches the VLM->VLA contract:
        type/description = class label, reading = null, confidence, bbox(0..1).
    """
    objects = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return objects
    names = result.names or {}
    xyxyn = boxes.xyxyn.tolist()       # normalized [x_min, y_min, x_max, y_max]
    confs = boxes.conf.tolist()
    classes = boxes.cls.tolist()
    for bb, cf, cl in zip(xyxyn, confs, classes):
        label = names.get(int(cl), str(int(cl)))
        objects.append({
            "type": label,
            "description": label,
            "reading": None,
            "confidence": round(float(cf), 4),
            "bbox": [round(float(v), 4) for v in bb],
            # Additive, UI-only hint: the model's class id, so the frontend can
            # color each box with YOLO's own per-class palette. Downstream
            # contract consumers ignore unknown keys.
            "class_id": int(cl),
        })
    return objects


def annotated_dir():
    """Folder where annotated (boxed) images are written. Created on demand."""
    d = os.path.join(PROJECT_ROOT, "results", "annotated")
    os.makedirs(d, exist_ok=True)
    return d


def default_annotated_path(image_path, model_name):
    """Build a NON-colliding output path for the annotated image.

    Name is `<image>__<model>__<timestamp>.jpg`. The timestamp (down to the
    microsecond) makes every scan unique, so repeated scans of the same
    image+model — including with a different class filter — never overwrite each
    other.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    model_tag = os.path.splitext(os.path.basename(model_name))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(annotated_dir(), f"{base}__{model_tag}__{stamp}.jpg")


def save_annotated(result, out_path):
    """Save `result` with its boxes drawn to `out_path`. Returns the path or None.

    Uses ultralytics' own renderer (Results.save); falls back to plot()+cv2 if the
    installed version lacks the `filename` kwarg.
    """
    try:
        result.save(filename=out_path)
        return out_path
    except Exception:
        try:
            import cv2  # bundled with ultralytics
            cv2.imwrite(out_path, result.plot())
            return out_path
        except Exception:
            return None


def detect(model_name, source, conf=0.25, imgsz=640, device=None, classes=None):
    """Run YOLO on ANY ultralytics source and return the JSON-contract objects.

    Unlike run_detection (which takes a file PATH and can save an annotated copy),
    this accepts an in-memory source — a PIL.Image, a numpy array (HWC), a path or
    a URL — so the live backend can feed decoded webcam frames without touching
    disk. Returns the same shape used elsewhere: elapsed / objects / parsed / n /
    speed. `classes` is a list of class NAMES to keep (None/empty = all).
    """
    model = load_model(model_name)
    class_ids = resolve_class_ids(model, classes)
    t0 = time.perf_counter()
    results = model.predict(source=source, conf=conf, imgsz=imgsz, device=device,
                            classes=class_ids, verbose=False)
    elapsed = time.perf_counter() - t0
    result = results[0]
    objects = result_to_objects(result)
    return {
        "elapsed": elapsed,
        "objects": objects,
        "parsed": {"objects": objects},
        "n": len(objects),
        "speed": dict(getattr(result, "speed", {}) or {}),
    }


def run_detection(model_name, image_path, conf=0.25, imgsz=640, device=None,
                  warmup=False, annotate_path=None, classes=None):
    """Run YOLO on a single image and return a dict mirroring vlm_common.query_vlm.

    `conf`   : confidence threshold (drop detections below it).
    `imgsz`  : inference image size (longer side); bigger = more detail, slower.
    `device` : None lets ultralytics auto-pick (GPU if available, else CPU).
    `warmup` : if True, run one untimed inference first to exclude model-load /
               CUDA-init cost from `elapsed` (used by the benchmark).
    `annotate_path` : if set, also save a copy of the image with the detected
               boxes drawn on it (so you can eyeball whether YOLO "hit"); the
               saved path is returned in res["annotated"].
    `classes` : list of class NAMES to keep (e.g. ["person", "cup"]); None/empty
               means no filter (detect everything). Names absent from the model
               are ignored.

    Returns: elapsed (wall seconds), objects, parsed ({"objects": [...]}), ok,
    n (object count), `speed` (ultralytics' per-stage ms) and `annotated` (path or
    None). Raises on a genuine inference error (caller handles it).
    """
    model = load_model(model_name)
    class_ids = resolve_class_ids(model, classes)
    if warmup:
        model.predict(source=image_path, conf=conf, imgsz=imgsz, device=device,
                      classes=class_ids, verbose=False)
    t0 = time.perf_counter()
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz,
                            device=device, classes=class_ids, verbose=False)
    elapsed = time.perf_counter() - t0

    result = results[0]
    objects = result_to_objects(result)
    annotated = save_annotated(result, annotate_path) if annotate_path else None
    return {
        "elapsed": elapsed,
        "objects": objects,
        "parsed": {"objects": objects},
        "ok": True,
        "n": len(objects),
        "speed": dict(getattr(result, "speed", {}) or {}),
        "names": dict(getattr(result, "names", {}) or {}),
        "annotated": annotated,
    }


# --------------------------------------------------------------------------- #
# Rendering (smoke test / menu)
# --------------------------------------------------------------------------- #
def render_result(model_name, res):
    """Print a run_detection result in a readable way (mirrors vlm render_result)."""
    import json
    print("\n========== RESULT (YOLO) ==========")
    print(f"Model:        {model_name}")
    elapsed = res["elapsed"]
    e2e = f"{elapsed * 1000:.0f} ms" if elapsed < 1.0 else f"{elapsed:.3f} s"
    print(f"E2E latency:  {e2e}")
    speed = res.get("speed") or {}
    if speed:
        print("Stage (ms):   " + "  ".join(
            f"{k}={v:.1f}" for k, v in speed.items() if isinstance(v, (int, float))))
    print("-----------------------------------")
    print(json.dumps(res["parsed"], indent=2, ensure_ascii=False))
    print(f"\n[OK] Objects detected: {res['n']}")
    if res.get("annotated"):
        print(f"[OK] Annotated image (with boxes) saved to: {res['annotated']}")
    print("===================================")


# --------------------------------------------------------------------------- #
# Config (YOLO block, merged into the shared config.json)
# --------------------------------------------------------------------------- #
# config.json is shared with the VLM path; these are the YOLO-only keys. We keep
# the same shape as the VLM config: a smoke-test block + an independent benchmark
# block whose swept dimensions (conf, imgsz) are LISTS.
YOLO_DEFAULT_CONFIG = {
    "yolo_model": "yolov8n.pt",        # nano = fastest, good default for live video
    "yolo_image": "fotos/clean/1.jpeg",  # image for the YOLO smoke test
    "yolo_folder": "fotos/clean",      # folder for the YOLO benchmark
    "yolo_conf": 0.25,                 # confidence threshold (single-image scan)
    "yolo_imgsz": 320,                 # inference image size; 320 = smallest/fastest for live video
    "yolo_save": True,                 # save the annotated (boxed) image on a scan
    "yolo_classes": [],                # class NAMES to keep ([] = ALL, no filter)

    # --- YOLO benchmark: its own block, conf/imgsz swept as LISTS -------------
    "yolo_benchmark_models": ["yolov8n.pt", "yolov8s.pt"],
    "yolo_benchmark_runs": 5,          # YOLO is fast, so more runs by default
    "yolo_benchmark_images": [],       # hand-picked images ([] = ALL in the folder)
    "yolo_benchmark_conf": [0.25],     # confidence thresholds to compare (list)
    "yolo_benchmark_imgsz": [640],     # inference sizes to compare (list)
}


def load_config():
    """Load the shared config and fill in any missing YOLO defaults.

    Wraps vlm_common.load_config so a single config.json holds BOTH the VLM and
    YOLO settings. If YOLO keys are absent (e.g. an old config from before YOLO
    existed) they are added and persisted.
    """
    cfg = _load_base_config()
    missing = False
    for k, v in YOLO_DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = list(v) if isinstance(v, list) else v
            missing = True
    if missing:
        save_config(cfg)
    return cfg
