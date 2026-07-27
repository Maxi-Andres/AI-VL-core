#!/usr/bin/env python3
"""
yolo_benchmark.py — Latency benchmark (P50/P95) and object counts for YOLO.

The YOLO counterpart of vlm_benchmark.py. Runs the SAME set of images, N times
per image, sweeping the product MODELS × imgsz × conf, and reports latency
(P50/P95/mean/min/max), average objects per image and errors per combination.
Use it to pick the YOLO detector for the live-video path: there are many
pretrained sizes (n/s/m/l/x across YOLOv8/v10/11) and `pip install ultralytics`
gives you all of them, so comparing them is the whole point.

Why no JSON-rate column (unlike the VLM benchmark)? YOLO always returns a
structured result, so "valid JSON" is trivially 100%. The interesting axes here
are speed and how many objects each size/threshold finds.

NOTE on timing: the FIRST inference of a model includes weights download +
load + CUDA warmup, which would pollute latency. So each combination runs one
untimed warmup inference before the measured runs (see run_detection(warmup=...)).

Results are saved to results/ (not loose in the project root).

Don't want to type flags? Run `python3 menu.py` -> pick YOLO -> Benchmark.

Requirements:  pip install ultralytics
Usage:
    python3 src/yolo_benchmark.py fotos/clean --runs 5
    python3 src/yolo_benchmark.py fotos/clean --models yolov8n.pt yolo11n.pt
    python3 src/yolo_benchmark.py fotos/clean --imgsz 640 1280 --conf 0.25 0.5
    python3 src/yolo_benchmark.py fotos/clean --images 1.jpeg 14.jpeg
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

from yolo_common import (
    as_list,
    fmt_secs,
    list_images,
    load_config,
    pctl,
    progress_bar,
    results_path,
    run_detection,
    ultralytics_available,
)


def fmt_ms(s):
    """Format a latency. YOLO is millisecond-scale, so show ms under 1s.

    (fmt_secs from vlm_common is tuned for the seconds-scale VLM and would round
    every YOLO time down to '0.0s'.)
    """
    if s != s:  # NaN
        return "  -  "
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    return fmt_secs(s)


def combo_label(c):
    """Compact, unique label for a sweep combination."""
    return f"{c['model']} imgsz{c['imgsz']} conf{c['conf']}"


def _stats(latencies, objs, errors, total):
    """Build the metrics dict for a combination (model × imgsz × conf)."""
    return {
        "p50": pctl(latencies, 50),
        "p95": pctl(latencies, 95),
        "mean": statistics.mean(latencies) if latencies else float("nan"),
        "min": min(latencies) if latencies else float("nan"),
        "max": max(latencies) if latencies else float("nan"),
        "total_time": sum(latencies),
        "avg_objects": statistics.mean(objs) if objs else 0.0,
        "errors": errors,
        "n": total,
    }


def run_yolo_benchmark(images, models, runs=5, conf=0.25, imgsz=640, device=None,
                       out=None):
    """Run the YOLO benchmark sweeping the product models × imgsz × conf.

    `images` : list of concrete paths (expand a folder with list_images first).
    `models` : list of YOLO weights names (.pt) to compare.
    `conf`   : scalar or LIST of confidence thresholds to compare.
    `imgsz`  : scalar or LIST of inference sizes to compare.
    `device` : None lets ultralytics auto-pick (GPU if available).

    Each combination is one row. Prints latency + avg objects per combination and
    a final verdict (fastest), then saves the full payload (metrics + per-image
    detections) to a timestamped file under results/.
    """
    images = [p for p in images if os.path.exists(p)]
    if not images:
        print("[ERROR] No valid images to run.", file=sys.stderr)
        return None
    if not models:
        print("[ERROR] No models to run.", file=sys.stderr)
        return None
    if not ultralytics_available():
        print("[ERROR] The 'ultralytics' package is not installed. "
              "Install it with:  pip install ultralytics", file=sys.stderr)
        return None

    conf_list = as_list(conf, [0.25])
    imgsz_list = as_list(imgsz, [640])

    combos = [
        {"model": m, "conf": cf, "imgsz": sz}
        for m in models
        for sz in imgsz_list
        for cf in conf_list
    ]
    total_calls = len(images) * runs * len(combos)

    print("=" * 72)
    print(" BENCHMARK YOLO")
    print("=" * 72)
    print(f"  Images : {len(images)}  ->  {', '.join(os.path.basename(i) for i in images)}")
    print(f"  Models : {len(models)}  ->  {', '.join(models)}")
    print(f"  imgsz  : {', '.join(str(x) for x in imgsz_list)}")
    print(f"  conf   : {', '.join(str(x) for x in conf_list)}")
    print(f"  Runs/img: {runs}     Device: {device or 'auto'}")
    print(f"  Total calls: {total_calls}  ({len(combos)} combination(s) model×imgsz×conf)")
    print("  (each combination does 1 untimed warmup inference before measuring)")
    print("=" * 72 + "\n")

    results = {}
    done = 0
    bench_t0 = time.perf_counter()

    for combo in combos:
        model = combo["model"]
        label = combo_label(combo)
        print(f"=== Model: {model}  |  imgsz={combo['imgsz']}  conf={combo['conf']} ===")
        latencies, errors, total, objs = [], 0, 0, []
        per_image = {}     # img -> list of latencies
        detections = {}    # img -> list of per-run {result/ok/elapsed_s}
        warmed = False

        for img in images:
            name = os.path.basename(img)
            per_image.setdefault(name, [])
            detections.setdefault(name, [])
            for r in range(1, runs + 1):
                total += 1
                done += 1
                avg_so_far = statistics.mean(latencies) if latencies else 0.0
                eta = avg_so_far * (total_calls - done + 1)
                progress_bar(done - 1, total_calls,
                             suffix=f"{label} | {name} run {r}/{runs} | "
                                    f"avg {fmt_secs(avg_so_far)} | ETA {fmt_secs(eta)}")
                try:
                    # Warm up once per combination (excludes load/CUDA-init time).
                    res = run_detection(model, img, conf=combo["conf"],
                                        imgsz=combo["imgsz"], device=device,
                                        warmup=not warmed)
                    warmed = True
                    latencies.append(res["elapsed"])
                    per_image[name].append(res["elapsed"])
                    objs.append(res["n"])
                    detections[name].append({
                        "run": r, "elapsed_s": round(res["elapsed"], 3),
                        "ok": True, "result": res["parsed"],
                    })
                except Exception as e:
                    errors += 1
                    detections[name].append({"run": r, "ok": False, "error": str(e)})
                    progress_bar(done, total_calls, suffix=f"ERROR on {name}")
                    print(f"\n  [!] {name}: ERROR {e}")
                    continue
            avg_so_far = statistics.mean(latencies) if latencies else 0.0
            eta = avg_so_far * (total_calls - done)
            progress_bar(done, total_calls,
                         suffix=f"{label} | done {name} | avg {fmt_secs(avg_so_far)} | "
                                f"ETA {fmt_secs(eta)}")
        progress_bar(done, total_calls, suffix=f"{label} complete")

        print(f"\n  Time per image ({label}):")
        for name, lats in per_image.items():
            if lats:
                print(f"    {name:28s} avg {fmt_ms(statistics.mean(lats)):>7s}  "
                      f"(min {fmt_ms(min(lats))} / max {fmt_ms(max(lats))})")
            else:
                print(f"    {name:28s} no data (errors)")

        stats = _stats(latencies, objs, errors, total)
        stats["model"] = model
        stats["conf"] = combo["conf"]
        stats["imgsz"] = combo["imgsz"]
        stats["detections"] = detections
        results[label] = stats
        print()

    bench_wall = time.perf_counter() - bench_t0

    # ---------------- Comparison summary + times ----------------
    mw = max([len("Model")] + [len(r["model"]) for r in results.values()])
    header = (f"{'Model':{mw}s}  {'imgsz':>6s} {'conf':>5s} "
              f"{'P50':>7s} {'P95':>7s} {'Mean':>7s} {'Min':>7s} {'Max':>7s} "
              f"{'Total':>8s} {'~obj':>5s} {'Errs':>5s}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results.values():
        print(f"{r['model']:{mw}s}  {r['imgsz']:>6d} {r['conf']:>5.2f} "
              f"{fmt_ms(r['p50']):>7s} {fmt_ms(r['p95']):>7s} {fmt_ms(r['mean']):>7s} "
              f"{fmt_ms(r['min']):>7s} {fmt_ms(r['max']):>7s} "
              f"{fmt_ms(r['total_time']):>8s} {r['avg_objects']:5.1f} {r['errors']:5d}")
    print("=" * len(header))
    print(f"Total benchmark time (wall clock): {fmt_secs(bench_wall)}  "
          f"|  {total_calls} calls")
    print("Doc target (F1.8): P95 < 1.5 s on-prem (trivial for YOLO; watch ~obj/precision).")

    # Verdict: fastest combination (all YOLO combos return valid structured output).
    valid = {k: r for k, r in results.items()
             if r["mean"] == r["mean"] and r["errors"] < r["n"]}
    pool = valid or {k: r for k, r in results.items() if r["mean"] == r["mean"]}
    if pool and len(results) > 1:
        best = min(pool, key=lambda k: pool[k]["mean"])
        r = results[best]
        print(f"-> Fastest combination: {best}.")
        print(f"   To pin it in config.json: \"yolo_model\"={r['model']!r}, "
              f"\"yolo_imgsz\"={r['imgsz']}, \"yolo_conf\"={r['conf']}.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"config": {"timestamp": stamp, "runs": runs,
                          "models": models, "imgsz": imgsz_list, "conf": conf_list,
                          "device": device or "auto",
                          "images": [os.path.basename(i) for i in images],
                          "total_wall_s": round(bench_wall, 2)},
               "results": results}
    out = out or results_path(f"yolo_benchmark_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out}")
    return results


def main():
    cfg = load_config()  # defaults come from config.json (yolo_benchmark_* keys)
    ap = argparse.ArgumentParser(description="YOLO latency benchmark across models/imgsz/conf.")
    ap.add_argument("folder", nargs="?", default=cfg["yolo_folder"],
                    help="Folder with images. Default: the one in config.json")
    ap.add_argument("--images", nargs="+", default=None,
                    help="Names of specific images inside the folder. Default: ALL.")
    ap.add_argument("--models", nargs="+", default=cfg["yolo_benchmark_models"],
                    help="YOLO weights (.pt) to compare")
    ap.add_argument("--runs", type=int, default=cfg["yolo_benchmark_runs"],
                    help="Repetitions per image")
    ap.add_argument("--conf", nargs="+", type=float,
                    default=cfg.get("yolo_benchmark_conf", [0.25]),
                    help="One or SEVERAL confidence thresholds to compare (e.g. --conf 0.25 0.5)")
    ap.add_argument("--imgsz", nargs="+", type=int,
                    default=cfg.get("yolo_benchmark_imgsz", [640]),
                    help="One or SEVERAL inference sizes to compare (e.g. --imgsz 640 1280)")
    ap.add_argument("--device", default=None,
                    help="Force device (e.g. 'cpu', '0'). Default: ultralytics auto-picks.")
    ap.add_argument("--out", default=None,
                    help="Output file path. Default: results/yolo_benchmark_<timestamp>.json")
    args = ap.parse_args()

    all_imgs = list_images(args.folder)
    if args.images:
        wanted = set(args.images)
        images = [p for p in all_imgs if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in images}
        if missing:
            print(f"[!] Not found in {args.folder}: {', '.join(sorted(missing))}",
                  file=sys.stderr)
    else:
        images = all_imgs

    res = run_yolo_benchmark(images, args.models, runs=args.runs, conf=args.conf,
                             imgsz=args.imgsz, device=args.device, out=args.out)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
