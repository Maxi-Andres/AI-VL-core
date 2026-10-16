#!/usr/bin/env python3
"""
asr_benchmark.py — Which speech-to-text model? Word error rate, latency and VRAM.

Transcribes real Argentine speakers (OpenSLR 61, "Crowdsourced high-quality Argentinian
Spanish speech", one short sentence per speaker) with each configuration and compares the
text with what was said. It is the test behind linux/run.sh's choice (Whisper small on the
GPU in int8, measured 2026-10-16; results in docs/MODEL_BENCHMARKS.md).

  1. --fetch downloads the clips once into asr_data/ (git-ignored): one sentence from each
     of N different speakers, through the Hugging Face datasets-server (which rate-limits:
     the downloader retries 429/5xx and resumes).
  2. Each configuration runs in its OWN process, so the VRAM it reports is its own:
     peak nvidia-smi during the run minus what was used before it loaded.
  3. WER is word-level edit distance over the reference word count, after lower-casing and
     dropping punctuation; "no accents" also drops the tildes (Whisper often writes "mas"
     for "más", which a listener would never notice).

Engines: faster-whisper (always; any size/device/compute type) and NVIDIA Parakeet TDT
0.6B v3 through onnx-asr, when `pip install "onnx-asr[cpu,hub]"` is present (not a
dependency of the app).

Usage:
    python3 src/asr_benchmark.py --fetch                 # once: 30 speakers into asr_data/
    python3 src/asr_benchmark.py                         # the default sweep
    python3 src/asr_benchmark.py --configs small:cuda:int8_float16 base:cpu:int8
    python3 src/asr_benchmark.py --configs parakeet:cpu:int8
"""
import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime

from vlm_common import PROJECT_ROOT, results_path

DATA_DIR = os.path.join(PROJECT_ROOT, "asr_data")
DATASET = "ylacombe/google-argentinian-spanish"   # OpenSLR 61 on Hugging Face
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=" + DATASET +
            "&config={cfg}&split=train&offset={off}&length=3")

# model:device:compute — what was compared on 2026-10-16.
DEFAULT_CONFIGS = [
    "base:cpu:int8", "small:cpu:int8", "base:cuda:float16", "small:cuda:float16",
    "small:cuda:int8_float16", "medium:cuda:float16", "large-v3-turbo:cuda:int8_float16",
    "large-v3-turbo:cuda:float16", "parakeet:cpu:int8",
]


# --------------------------------------------------------------------------- #
# Scoring (also used by tts_benchmark.py)
# --------------------------------------------------------------------------- #
def words(text, accents=True):
    """Lower-cased words without punctuation; accents=False also strips the tildes."""
    text = text.lower()
    if not accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", text).split()


def edit_distance(ref, hyp):
    """Word-level Levenshtein distance between two word lists."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[len(hyp)]


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0


# --------------------------------------------------------------------------- #
# The test set
# --------------------------------------------------------------------------- #
def _get(url, path=None, tries=8):
    for i in range(tries):
        try:
            if path:
                return urllib.request.urlretrieve(url, path)
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if getattr(e, "code", 500) not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
            time.sleep(8 * (i + 1))
    return None


def fetch(speakers_per_gender=15):
    """One sentence from each of N female and N male speakers into asr_data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    meta = []
    for cfg in ("female", "male"):
        seen = set()
        for off in range(0, 100 * speakers_per_gender, 100):
            if len(seen) >= speakers_per_gender:
                break
            for r in _get(ROWS_URL.format(cfg=cfg, off=off)).get("rows", []):
                row = r["row"]
                if row["speaker_id"] in seen:
                    continue
                seen.add(row["speaker_id"])
                name = f"{cfg}_{r['row_idx']}.wav"
                path = os.path.join(DATA_DIR, name)
                if not os.path.exists(path):
                    _get(row["audio"][0]["src"], path)
                    time.sleep(1)
                meta.append({"file": name, "text": row["text"], "speaker": row["speaker_id"]})
                break
            time.sleep(1.5)
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[OK] {len(meta)} clips in {DATA_DIR}")


# --------------------------------------------------------------------------- #
# One configuration (runs in a child process: see main)
# --------------------------------------------------------------------------- #
def _whisper(model, device, compute):
    import asr_common
    m = asr_common.load_model(model, device, compute)
    if device == "cuda" and m.model.device != "cuda":
        raise RuntimeError("Whisper could not use the GPU (asr_common fell back to the CPU)")

    def run(path):
        with open(path, "rb") as f:
            segments, _ = m.transcribe(io.BytesIO(f.read()), language="es", beam_size=5)
        return "".join(s.text for s in segments).strip()
    return run


def _parakeet(device, quant):
    import onnx_asr
    import soundfile
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",
                            quantization=None if quant == "none" else quant,
                            providers=[provider])

    def run(path):
        wav, sr = soundfile.read(path, dtype="float32")
        return m.recognize(wav, sample_rate=sr)
    return run


def run_config(config):
    name, device, compute = config.split(":")
    with open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    clips = [(os.path.join(DATA_DIR, x["file"]), x["text"]) for x in meta]
    before = vram_used_mib()
    t0 = time.perf_counter()
    run = _parakeet(device, compute) if name == "parakeet" else _whisper(name, device, compute)
    load = time.perf_counter() - t0
    run(clips[0][0])                                   # warm-up, not counted
    peak, lat, err, err_na, n, outs = vram_used_mib(), [], 0, 0, 0, []
    for path, ref in clips:
        t = time.perf_counter()
        hyp = run(path)
        lat.append(time.perf_counter() - t)
        peak = max(peak, vram_used_mib())
        err += edit_distance(words(ref), words(hyp))
        err_na += edit_distance(words(ref, False), words(hyp, False))
        n += len(words(ref))
        outs.append({"said": ref, "heard": hyp})
    lat.sort()
    return {"config": config, "engine": "parakeet-tdt-0.6b-v3" if name == "parakeet"
            else f"whisper {name}", "device": device, "compute": compute,
            "load_s": round(load, 2), "mean_s": round(statistics.mean(lat), 3),
            "p90_s": round(lat[int(len(lat) * 0.9)], 3), "wer": round(100 * err / n, 1),
            "wer_no_accents": round(100 * err_na / n, 1),
            "vram_mib": max(0, peak - before) if device == "cuda" else 0, "clips": outs}


def main():
    ap = argparse.ArgumentParser(description="Speech-to-text WER, latency and VRAM.")
    ap.add_argument("--fetch", action="store_true", help="Download the test clips and exit.")
    ap.add_argument("--speakers", type=int, default=15, help="Speakers per gender (--fetch).")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                    help="model:device:compute, e.g. small:cuda:int8_float16 parakeet:cpu:int8")
    ap.add_argument("--one", help=argparse.SUPPRESS)   # child process: one config, JSON out
    args = ap.parse_args()

    if args.fetch:
        fetch(args.speakers)
        return
    if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
        sys.exit("[ERROR] no test clips: run with --fetch first")
    if args.one:
        print(json.dumps(run_config(args.one), ensure_ascii=False))
        return

    rows = []
    for config in args.configs:
        print(f"=== {config}", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", config],
                           capture_output=True, text=True, timeout=3600)
        try:
            row = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            print(f"  [!] failed: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
            continue
        rows.append(row)
        print(f"  WER {row['wer']}% (no accents {row['wer_no_accents']}%)  "
              f"{row['mean_s']}s/clip  VRAM {row['vram_mib']} MiB", flush=True)

    print(f"\n{'engine':24s} {'device':6s} {'compute':13s} {'WER':>6s} {'no acc':>7s} "
          f"{'mean':>7s} {'p90':>7s} {'VRAM':>7s}")
    for r in rows:
        print(f"{r['engine']:24s} {r['device']:6s} {r['compute']:13s} {r['wer']:5.1f}% "
              f"{r['wer_no_accents']:6.1f}% {r['mean_s']:6.3f}s {r['p90_s']:6.3f}s "
              f"{r['vram_mib']:5d}M")
    out = results_path(f"asr_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "rows": rows}, f, ensure_ascii=False, indent=1)
        f.write("\n")  # pre-commit's end-of-file-fixer
    print(f"\n[OK] Saved to {out}")


if __name__ == "__main__":
    main()
