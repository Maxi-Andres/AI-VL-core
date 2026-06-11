# YOLO guide — detection path of the PoC

This document explains, in detail, how the **YOLO path** of this PoC works: how to
run it, every variable you can tune to change what it detects, how to train your
own model on industrial instruments, and what all the YOLO/Ultralytics model
families are for and how they differ.

> The YOLO path is the in-process detector (Ultralytics). In the real deployment
> it runs on the **live video stream**; the VLM is only called when a user asks a
> question about a frame. Here it works on **single images** (live video needs a
> WebSocket + front/back-end, out of scope for now). It emits the **same JSON
> contract** as the VLM so the downstream consumer (Silk AI Proxy Gateway / F1.9)
> sees one shape.

---

## 0. TL;DR

```bash
# Always use the project venv (system Python 3.14 is externally-managed)
.venv/bin/python menu.py                       # menu -> pick YOLO
.venv/bin/python src/yolo_scan.py fotos/clean/1.jpeg
.venv/bin/python src/yolo_benchmark.py fotos/clean --models yolov8n.pt yolo11n.pt
```

- **Change the model** to trade speed vs accuracy: `yolo11n.pt` (fast) … `yolo11x.pt` (accurate).
- **Lower `--conf`** to detect more (and more false) boxes; **raise it** to be stricter.
- **Raise `--imgsz`** (e.g. 1280) to catch small/distant objects, at the cost of speed.
- A COCO-pretrained model only knows **80 generic classes** (it calls a pressure
  gauge a *"clock"*). For industrial instruments you either **train a custom model**
  (section 5) or use an **open-vocabulary** model like YOLO-World / YOLOE (section 7).

---

## 1. Setup

The YOLO path needs the `ultralytics` package (it pulls in `torch` + `opencv`).
Because the system Python is 3.14 and "externally-managed" (PEP 668 — `pip
install` into it is blocked), it is installed in a project virtual environment at
`.venv/`. The easy way (works on a fresh clone too):

```bash
./setup.sh                 # creates .venv/ and installs requirements.txt
source .venv/bin/activate
```

`.venv/` is git-ignored (machine-specific, ~GB), so it is never committed —
`./setup.sh` rebuilds it from `requirements.txt`. Under the hood it does the
equivalent of:

```bash
python3 -m venv --without-pip .venv      # fallback when venv lacks ensurepip
curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt
```

Run **everything** with `.venv/bin/python` (the venv also has `requests`, so it
runs the VLM path too). Pretrained weights (`*.pt`) are **downloaded
automatically** the first time you use a model name, and land in the repo root
(git-ignored).

---

## 2. How to run

### 2.1 Single image — `yolo_scan.py`

```bash
.venv/bin/python src/yolo_scan.py fotos/clean/1.jpeg
.venv/bin/python src/yolo_scan.py fotos/clean/1.jpeg --model yolo11n.pt --conf 0.4 --imgsz 1280
.venv/bin/python src/yolo_scan.py fotos/clean/1.jpeg --no-save      # don't save the boxed image
```

It prints the latency, the parsed JSON (`{"objects": [...]}`), and — unless
`--no-save` — saves a copy of the image **with the detected boxes drawn** to
`results/annotated/<image>__<model>.jpg` so you can eyeball whether it "hit".

### 2.2 Many images / model comparison — `yolo_benchmark.py`

```bash
.venv/bin/python src/yolo_benchmark.py fotos/clean --runs 5
.venv/bin/python src/yolo_benchmark.py fotos/clean --models yolov8n.pt yolo11n.pt yolo11s.pt
.venv/bin/python src/yolo_benchmark.py fotos/clean --imgsz 640 1280 --conf 0.25 0.5
.venv/bin/python src/yolo_benchmark.py fotos/clean --images 1.jpeg 2.jpeg
```

It sweeps the **cartesian product of models × imgsz × conf**, runs each
combination N times per image (with one untimed warmup to exclude model-load /
CUDA-init from the latency), and reports P50/P95/mean/min/max latency (in ms),
average objects per image and errors. The full payload (metrics + what each model
returned per image/run) is saved to `results/yolo_benchmark_<timestamp>.json`.

### 2.3 The menu

`.venv/bin/python menu.py` → choose **YOLO**. The YOLO menu lets you pick the
model (`.pt`), the image, `conf`/`imgsz`, toggle saving the annotated image, and
open the benchmark submenu — all persisted to `config.json`.

---

## 3. Variables that change the detection

These are the knobs that affect **what** and **how much** YOLO detects. The ones
**wired into this project** are marked ✅ (CLI flag + `config.json` key); the rest
are standard Ultralytics `predict()` arguments you can add to `run_detection()` in
`src/yolo_common.py` if you need them.

| Variable | Default | Effect | In project |
|----------|---------|--------|------------|
| **`model`** | `yolov8n.pt` | The weights = the detector. Size and family decide speed/accuracy and which classes it knows. See sections 4 & 6. | ✅ `--model` / `yolo_model` |
| **`conf`** (confidence) | `0.25` | Minimum score to keep a box. **Lower** → more detections (and more false positives); **higher** → only confident ones. The single biggest lever for precision/recall. | ✅ `--conf` / `yolo_conf` |
| **`imgsz`** (inference size) | `640` | The image is resized so its longer side = `imgsz` before inference. **Bigger** (e.g. 1280) → small/distant objects become detectable, but slower and more memory. Must be a multiple of 32. | ✅ `--imgsz` / `yolo_imgsz` |
| **`device`** | auto | `None`/auto picks GPU if available else CPU. Force with `cpu`, `0` (first GPU), `0,1`, or `mps` (Apple). | ✅ `--device` |
| **`iou`** (NMS threshold) | `0.7` | Non-Max-Suppression overlap threshold. **Lower** → more aggressive merging (fewer overlapping boxes); **higher** → keeps more overlapping boxes. Tune when objects are close together or get double-detected. | add to `predict()` |
| **`max_det`** | `300` | Hard cap on detections per image. Raise it for very crowded scenes. | add to `predict()` |
| **`classes`** | all | Filter to a subset of class IDs, e.g. `classes=[0]` (only "person"). Useful to ignore irrelevant COCO classes. | add to `predict()` |
| **`agnostic_nms`** | `False` | If `True`, NMS ignores class — overlapping boxes of different classes get merged. | add to `predict()` |
| **`augment`** (TTA) | `False` | Test-Time Augmentation: runs several augmented views and merges them. Slightly better recall, much slower. | add to `predict()` |
| **`half`** (FP16) | `False` | Half-precision inference on GPU — faster, less VRAM, negligible accuracy loss. | add to `predict()` |
| **`retina_masks`** | `False` | (segmentation models) higher-quality masks. | add to `predict()` |

**Recommended workflow to tune detection on your images:**
1. Start with the default model and `conf=0.25`, `imgsz=640`.
2. If it **misses** objects → lower `conf` (e.g. 0.10) and/or raise `imgsz` (e.g. 1280).
3. If it produces **junk** boxes → raise `conf` (e.g. 0.4–0.5).
4. If boxes are **doubled/overlapping** → lower `iou` (e.g. 0.5).
5. Compare options objectively with the benchmark: `--conf 0.15 0.25 0.5 --imgsz 640 1280`.

### 3.1 Output mapping (the JSON contract)

Each detection becomes one object in `{"objects": [...]}`:

| Contract key | YOLO source | Notes |
|--------------|-------------|-------|
| `type` | class label (e.g. `"clock"`) | For a COCO model this is one of 80 generic names; for a custom model it's **your** class names. |
| `description` | class label (same) | Free text; with a custom dataset you control these names. |
| `reading` | `null` | A plain detector reads no instrument value (that's the VLM's job). |
| `confidence` | box score 0–1 | |
| `bbox` | `boxes.xyxyn` | `[x_min, y_min, x_max, y_max]` normalized 0–1 (already normalized by Ultralytics; no pixel conversion needed, unlike the VLM). |

---

## 4. Model **sizes** (n / s / m / l / x)

Within one family (e.g. YOLO11 or YOLOv8) the suffix is the **scale**: more
parameters → more accuracy, less speed, more VRAM. Same architecture, different
width/depth.

| Size | Nickname | Relative speed | Relative accuracy | Typical use |
|------|----------|----------------|-------------------|-------------|
| `n` | nano | ⚡⚡⚡⚡⚡ fastest | lowest | edge / live video / CPU |
| `s` | small | ⚡⚡⚡⚡ | low-mid | real-time on modest GPU |
| `m` | medium | ⚡⚡⚡ | mid | balanced |
| `l` | large | ⚡⚡ | high | accuracy-leaning, good GPU |
| `x` | extra-large | ⚡ slowest | highest | offline / max accuracy |

Approximate figures (COCO val, 640px) for **YOLO11** vs **YOLOv8** — for intuition,
not exact:

| Model | Params | COCO mAP50-95 |
|-------|-------:|--------------:|
| yolo11n | ~2.6 M | ~39.5 |
| yolo11s | ~9.4 M | ~47.0 |
| yolo11m | ~20 M | ~51.5 |
| yolo11l | ~25 M | ~53.4 |
| yolo11x | ~57 M | ~54.7 |
| yolov8n | ~3.2 M | ~37.3 |
| yolov8s | ~11 M | ~44.9 |
| yolov8m | ~26 M | ~50.2 |
| yolov8l | ~44 M | ~52.9 |
| yolov8x | ~68 M | ~53.9 |

> For a **live video** stream, start with `n` or `s`. Only move up if accuracy is
> insufficient and you have the GPU headroom. The benchmark is exactly the tool to
> find that sweet spot.

### 4.1 Tasks (model suffix `-seg` / `-pose` / `-cls` / `-obb`)

Most modern families ship the same scale in several **task** variants:

| Suffix | Task | Output | Example |
|--------|------|--------|---------|
| *(none)* | **Detect** | bounding boxes | `yolo11n.pt` |
| `-seg` | **Segment** | boxes + pixel masks | `yolo11n-seg.pt` |
| `-pose` | **Pose / keypoints** | skeleton keypoints | `yolo11n-pose.pt` |
| `-cls` | **Classify** | whole-image label | `yolo11n-cls.pt` |
| `-obb` | **Oriented boxes** | rotated boxes (angle) | `yolo11n-obb.pt` |

This PoC uses **Detect**. `-obb` is interesting for industrial scenes where
instruments are mounted at angles; `-seg` if you later need exact pixel masks.

---

## 5. Training your own model (custom industrial classes)

A COCO-pretrained model only knows 80 generic things and will mislabel a pressure
gauge as `clock`. To detect *pressure gauge*, *thermometer*, *flow meter*, etc.,
train a custom detector. You almost always **fine-tune from a pretrained checkpoint**
(transfer learning) rather than from scratch — far less data and time.

### 5.1 Dataset format (Ultralytics YOLO)

```
my_dataset/
├── images/
│   ├── train/  img001.jpg ...
│   └── val/    img050.jpg ...
├── labels/
│   ├── train/  img001.txt ...   # one .txt per image, same basename
│   └── val/    img050.txt ...
└── data.yaml
```

Each label `.txt` has one line per object, **normalized 0–1**:

```
<class_id> <x_center> <y_center> <width> <height>
0 0.51 0.43 0.62 0.55
```

`data.yaml`:

```yaml
path: ./my_dataset        # dataset root
train: images/train
val: images/val
names:                    # class_id -> name (these become `type`/`description`)
  0: pressure_gauge
  1: thermometer
  2: flow_meter
  3: control_valve
```

> Labeling tools: **Roboflow**, **CVAT**, **Label Studio**, `labelImg`. They export
> directly in this format. Aim for at least a few hundred boxes per class to start.

### 5.2 Train

CLI:

```bash
.venv/bin/yolo detect train model=yolo11n.pt data=my_dataset/data.yaml \
    epochs=100 imgsz=640 batch=16
```

Python (equivalent):

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")          # start from pretrained weights
model.train(data="my_dataset/data.yaml", epochs=100, imgsz=640, batch=16)
```

Key training arguments:

| Arg | Meaning | Tip |
|-----|---------|-----|
| `model` | starting weights | `yolo11n.pt` (fast) → `yolo11m.pt` (more capacity). |
| `data` | path to `data.yaml` | required. |
| `epochs` | passes over the dataset | 100 is a sane start; watch for overfitting. |
| `imgsz` | training resolution | match your inference `imgsz` (e.g. 640 or 1280). |
| `batch` | images per step | `-1` = auto (fills GPU); lower if you hit OOM. |
| `patience` | early-stop epochs | stops if val stops improving. |
| `lr0` / `lrf` | initial / final learning rate | defaults are good; tune only if needed. |
| `optimizer` | `auto`/`SGD`/`AdamW` | `auto` is fine. |
| `freeze` | freeze first N layers | freeze the backbone for tiny datasets. |
| `pretrained` | start from pretrained | `True` (transfer learning) almost always. |
| `augment` / `mosaic` / `fliplr` / `hsv_*` | data augmentation | on by default; disable `fliplr` if orientation matters. |
| `device` | `0`, `cpu`, `0,1` | GPU strongly recommended for training. |
| `cache` | cache images in RAM/disk | speeds up epochs. |

Training writes to `runs/detect/train/`, with the best checkpoint at
`runs/detect/train/weights/best.pt`.

### 5.3 Validate & predict with your model

```bash
.venv/bin/yolo detect val   model=runs/detect/train/weights/best.pt data=my_dataset/data.yaml
```

Then use it in this project — `--model` accepts any path:

```bash
.venv/bin/python src/yolo_scan.py fotos/clean/1.jpeg --model runs/detect/train/weights/best.pt
```

(Or copy `best.pt` to the repo root and it shows up in the menu's model list.)

### 5.4 Export for deployment (edge / live video)

```bash
.venv/bin/yolo export model=best.pt format=onnx        # also: engine (TensorRT), openvino, tflite, coreml, ncnn
```

- **ONNX** — portable, runs in many runtimes.
- **TensorRT (`engine`)** — fastest on NVIDIA GPUs; ideal for the live-video box.
- **OpenVINO** — fast on Intel CPUs/iGPUs.
- **TFLite / NCNN** — mobile / embedded.

---

## 6. Model **families** — what they are and when to use them

The list below covers everything Ultralytics can load. They split into three groups:
**real-time detectors** (the YOLO line + RT-DETR), **promptable/open-vocabulary**
detectors (YOLO-World, YOLOE), and **segment-anything** models (SAM family).

### 6.1 Real-time detector lineage (the classic YOLO line)

| Model | Year / author | What it is | Use it for |
|-------|---------------|------------|------------|
| **YOLOv3** | 2018, Joseph Redmon | The classic efficient real-time detector that defined the family. | Legacy/reference; rarely the best choice today. |
| **YOLOv4** | 2020, Alexey Bochkovskiy | Darknet-native update to v3 (better accuracy/speed). | Legacy; use if you have a Darknet pipeline. |
| **YOLOv5** | 2020, Ultralytics | First PyTorch-native Ultralytics YOLO; great speed/accuracy trade-offs, huge ecosystem. | Stable, very well documented; fine baseline. |
| **YOLOv6** | 2022, Meituan | Industry-focused (used in delivery robots); strong on small models. | Edge/industrial deployment. |
| **YOLOv7** | 2022, v4 authors | High accuracy at the time. **Inference only** in Ultralytics. | Running existing v7 weights. |
| **YOLOv8** | 2023, Ultralytics | Versatile, multi-task (detect/seg/pose/cls/obb), anchor-free, very mature. | **Safe default**; widely supported, lots of examples. |
| **YOLOv9** | 2024, experimental | Adds Programmable Gradient Information (PGI) for better gradient flow. | Research / squeezing extra accuracy. |
| **YOLOv10** | 2024, Tsinghua | **NMS-free** training → lower end-to-end latency, efficiency-driven. | Latency-critical real-time inference. |
| **YOLO11** | 2024, Ultralytics | Current Ultralytics flagship; better accuracy with fewer params than v8, all tasks. | **Recommended modern default** (best efficiency/accuracy). |
| **YOLO26** | newest, Ultralytics | Next-gen, optimized for **edge** with end-to-end **NMS-free** inference. | Edge boxes / latest-and-greatest live video. |

> For this PoC's live video: **YOLO11n/s** (best modern trade-off) or **YOLOv10/YOLO26**
> if NMS-free end-to-end latency matters. **YOLOv8** remains the safest, most
> example-rich baseline. The benchmark lets you compare them head-to-head.

### 6.2 Transformer-based detector

| Model | What it is | Use it for |
|-------|------------|------------|
| **RT-DETR** (Baidu) | Real-time **Detection Transformer**. NMS-free by design, often very accurate on complex scenes. | When you want transformer accuracy at real-time speeds and have a decent GPU. |

### 6.3 Open-vocabulary detectors (no training for new classes) ⭐ relevant for this PoC

| Model | What it is | Use it for |
|-------|------------|------------|
| **YOLO-World** (Tencent) | Real-time **open-vocabulary** detection: you give **text prompts** (e.g. `"pressure gauge", "valve"`) and it detects them **without training**. | Detect industrial instruments by name **without** building a dataset. Great for a fast PoC. |
| **YOLOE** | Improved open-vocabulary detector that keeps YOLO real-time speed while detecting **arbitrary classes** beyond its training data (text/visual prompts). | Same idea as YOLO-World, newer; arbitrary classes at YOLO speed. |

> **Why this matters here:** a COCO model calls a manometer a `clock`. Instead of
> labeling a dataset, you can prompt YOLO-World/YOLOE with `"pressure gauge"`,
> `"thermometer"`, `"flow meter"` and get the right boxes immediately. Trade-off:
> open-vocab is usually a bit slower and less precise on niche objects than a
> model **trained** on your exact instruments. A common path: prototype with
> open-vocab, then train a custom model (section 5) once you have labeled data.

### 6.4 Segment Anything family (segmentation, not detection)

These produce **masks**, not class-labeled boxes. They don't fit the current
`{"objects": [...]}` box contract directly, but are useful for measurement,
masking, or auto-labeling.

| Model | What it is | Use it for |
|-------|------------|------------|
| **SAM** (Meta) | Original **Segment Anything**: prompt with points/boxes, get a mask of (almost) anything. | Zero-shot segmentation; auto-generating masks. |
| **SAM2** (Meta) | Next-gen SAM for **images and video** (temporal). | Segmenting/ tracking masks in video. |
| **SAM3** (Meta) | Adds **Promptable Concept Segmentation** via **text** and image exemplars. | Text-prompted segmentation of concepts. |
| **MobileSAM** (Kyung Hee Univ.) | Lightweight SAM for **mobile/edge**. | SAM on constrained hardware. |
| **FastSAM** (CAS) | CNN-based, much faster approximation of SAM. | Real-time-ish segmentation when SAM is too slow. |

### 6.5 Neural Architecture Search

| Model | What it is | Use it for |
|-------|------------|------------|
| **YOLO-NAS** (Deci) | Detector found via **Neural Architecture Search**; strong accuracy/latency, quantization-friendly. | Production deployments wanting NAS-optimized speed; INT8 quantization. |

---

## 7. Choosing for **this** PoC (industrial instruments)

1. **Quick demo, no dataset:** use **YOLO-World** or **YOLOE** with text prompts
   (`"pressure gauge"`, `"valve"`, `"thermometer"`). No labeling, immediate results.
2. **Best accuracy on your exact instruments:** **train a custom YOLO11/YOLOv8**
   model (section 5) on a labeled set, then point `--model` at `best.pt`.
3. **Live video latency is the priority:** **YOLO11n/s**, **YOLOv10**, or **YOLO26**
   (NMS-free, edge-optimized); export to **TensorRT** for the deployment box.
4. **Generic COCO models** (`yolov8n.pt`, `yolo11n.pt`) are only useful here for
   plumbing/latency tests — they don't know industrial classes (hence `clock`).

Always validate the choice with the benchmark:

```bash
.venv/bin/python src/yolo_benchmark.py fotos/clean \
    --models yolo11n.pt yolo11s.pt yolov8n.pt --imgsz 640 1280 --conf 0.25
```

---

## 8. Where things live (this repo)

| Path | Purpose |
|------|---------|
| `src/yolo_common.py` | Ultralytics loader, `run_detection()`, JSON mapping, annotated-image saving, config. |
| `src/yolo_scan.py` | Single-image scan (CLI + `run_yolo_scan()`). |
| `src/yolo_benchmark.py` | Sweep models × imgsz × conf, latency report. |
| `config.json` | `yolo_*` keys (model, image, conf, imgsz, save, benchmark_*). |
| `results/annotated/` | Saved boxed images (`<image>__<model>.jpg`), git-ignored. |
| `results/yolo_benchmark_*.json` | Benchmark results, git-ignored. |
| `.venv/` | Virtual env with `ultralytics` + `torch` (run everything with `.venv/bin/python`). |

To add a tuning knob (e.g. `iou`, `max_det`, `classes`) to the project, pass it
through `run_detection()` in `src/yolo_common.py` to `model.predict(...)` and add a
matching CLI flag / config key.
