#!/usr/bin/env python3
"""
vlm_server_benchmark.py — Which VLM should the app serve? Speed AND answers, model by model.

vlm_benchmark.py sweeps Ollama models × prompts for the detection JSON. This one answers
the question the voice assistant asks: for each candidate model served the way the app
serves it (llama-server, vision encoder on the GPU; see vlm_common.py, top), how long
until the first word, how fast it talks, how much VRAM it takes from YOLO, and whether
its answers are right. Optionally an Ollama model too, as the "before" row.

For each llama-server model it:
  1. starts ../linux/vlm-server.sh with that model on its own port (8082 by default, so
     a running app on 8081 is not touched) and times the load until /health answers;
  2. samples nvidia-smi the whole time: VRAM = peak minus what was used before it started;
  3. ASK — the voice path, query_vlm_stream: a Spanish question on photos scaled to
     768 px like the browser sends them; time to first piece, total, pieces per second;
  4. FACTS — questions with a known answer (a reading, a brand, a count), checked by regex;
  5. COMMANDS — command_common.interpret on spoken G1 commands, checked against the skill;
  6. DETECT — query_vlm on the scene photo, scope "all": valid JSON, objects, time;
  7. stops the server and waits for its VRAM to come back.

Every request gets a FRESH frame (one pixel changed): both llama-server and Ollama
cache image embeddings, and the first version of this script, which reused the bytes,
measured the second run of each photo at 0.3 s with no image work at all (2026-10-16). The
app never sends the same frame twice, so neither does this.

Usage:
    python3 src/vlm_server_benchmark.py                       # every model already downloaded
    python3 src/vlm_server_benchmark.py --models qwen3vl-4b-q4 gemma3-4b
    python3 src/vlm_server_benchmark.py --ollama qwen3-vl:4b-instruct   # add the "before" row
    python3 src/vlm_server_benchmark.py --models qwen3vl-4b-q4 --image-min-tokens 0 1024
    python3 src/vlm_server_benchmark.py --list

Results: results/vlm_server_benchmark_<timestamp>.json (every answer, for reading) and a
table on stdout. Model files live in vlm_models/ (git-ignored); --list says which are there.
"""
import argparse
import base64
import io
import itertools
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests

import command_common
import vlm_common
from vlm_common import OLLAMA_HOST, PROJECT_ROOT, results_path

MODELS_DIR = os.path.join(PROJECT_ROOT, "vlm_models")
SERVER_SH = os.path.join(os.path.dirname(PROJECT_ROOT), "linux", "vlm-server.sh")

# name -> (model file, mmproj file, Hugging Face repo they come from), smallest first.
CANDIDATES = {
    "smolvlm2-2.2b-q8": ("SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                         "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                         "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"),
    "qwen3vl-2b-q8": ("Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                      "Qwen/Qwen3-VL-2B-Instruct-GGUF"),
    "qwen3vl-4b-q4": ("Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
                      "Qwen/Qwen3-VL-4B-Instruct-GGUF"),
    "gemma3-4b-q4": ("gemma-3-4b-it-Q4_K_M.gguf", "mmproj-gemma-3-4b-it-f16.gguf",
                     "ggml-org/gemma-3-4b-it-GGUF"),
    "qwen3vl-4b-q8": ("Qwen3VL-4B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
                      "Qwen/Qwen3-VL-4B-Instruct-GGUF"),
    "internvl3-8b-q4": ("InternVL3-8B-Instruct-Q4_K_M.gguf",
                        "mmproj-InternVL3-8B-Instruct-Q8_0.gguf",
                        "ggml-org/InternVL3-8B-Instruct-GGUF"),
    "qwen3vl-8b-q4": ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
                      "Qwen/Qwen3-VL-8B-Instruct-GGUF"),
    "minicpm-v4.5-q4": ("MiniCPM-V-4_5-Q4_K_M.gguf", "mmproj-MiniCPM-V-4_5-f16.gguf",
                        "openbmb/MiniCPM-V-4_5-gguf"),
}

MAX_SIDE = 768          # what the browser sends (LivePage vlmMaxSize)
JPEG_QUALITY = 85       # and at what quality (captureFrame)
_FRESH = itertools.count(1)

ASK = [  # the voice path: open questions, Spanish, as the user speaks them
    ("fotos/ciudad/101.jpeg", "¿Qué ves en esta imagen? Respondé en español, en dos oraciones."),
    ("fotos/clean/14.jpeg", "¿Qué es este aparato y qué está midiendo? Respondé en español."),
    ("fotos/ciudad/100.jpeg", "Describí la escena en español, brevemente."),
]

# (image, question, regexes that must ALL match the answer, case-insensitive)
FACTS = [
    ("fotos/ciudad/101.jpeg", "¿Cuántas personas hay en la imagen? Respondé solo con el número.",
     [r"\b(2|dos)\b"]),
    ("fotos/ciudad/101.jpeg", "¿De qué marca es la impresora? Respondé solo la marca.",
     [r"\bhp\b|hewlett"]),
    ("fotos/clean/14.jpeg", "¿Qué valor en kW muestra la pantalla? Respondé solo el número.",
     [r"17[.,]84"]),
    ("fotos/clean/14.jpeg", "¿Qué modelo y marca es el instrumento?",
     [r"8220", r"chauvin|arnoux"]),
    ("fotos/clean/1.jpeg", "¿De qué marca es el manómetro y en qué unidades mide?",
     [r"beyca", r"kg"]),
    ("fotos/clean/5.jpeg", "¿Qué marca tiene el termómetro?", [r"de\s*wit"]),
    ("fotos/ciudad/100.jpeg", "¿Cuántas bicicletas hay en la calle? Respondé solo con el número.",
     [r"\b(1|una?|uno)\b"]),
    ("fotos/ciudad/100.jpeg", "¿De qué color es el auto que está en el centro de la calle?",
     [r"negr|black|oscur"]),
]

# (spoken command, the G1 skill it must become)
COMMANDS = [
    ("caminá un metro para adelante", "walk"),
    ("girá a la izquierda", "turn"),
    ("pará", "stop"),
    ("saludá con la mano", "wave_hand"),
    ("dame la mano", "shake_hand"),
    ("sentate", "sit"),
    ("levantate", "stand_up"),
    ("bailá algo", "dance"),
]

DETECT_IMAGE = "fotos/ciudad/101.jpeg"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_frame(rel):
    """The photo the way the browser would send it: longest side 768 (JPEG 85 on encode)."""
    from PIL import Image
    with Image.open(os.path.join(PROJECT_ROOT, rel)) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        return im, im.size


def fresh(frames, rel):
    """frames[rel] as base64 with one pixel changed, a different one each call, so no
    cache recognizes it."""
    n = next(_FRESH)
    im = frames[rel][0].copy()
    w, h = im.size
    im.putpixel(((n * 7919) % w, (n * 104729) % h), (n % 256, (n * 31) % 256, (n * 97) % 256))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class VramSampler(threading.Thread):
    """Peak VRAM used while a model is being tested, sampled twice a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = vram_used_mib()
            if v:
                self.peak = max(self.peak, v)
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(3)


def wait_vram_back(baseline, slack=300, timeout=30):
    """After a server stops, wait until VRAM is back near where it started."""
    end = time.time() + timeout
    while time.time() < end:
        v = vram_used_mib()
        if v is None or v <= baseline + slack:
            return
        time.sleep(1)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "min": None, "max": None}
    return {"mean": round(statistics.mean(xs), 3), "min": round(min(xs), 3),
            "max": round(max(xs), 3)}


# --------------------------------------------------------------------------- #
# The tests — each one goes through the same functions the service calls
# --------------------------------------------------------------------------- #
def ask_once(img, prompt, model, url):
    t0 = time.perf_counter()
    first, pieces, text = None, 0, []
    for piece in vlm_common.query_vlm_stream(img, model, prompt, max_tokens=200,
                                             num_ctx=4096, url=url):
        if first is None:
            first = time.perf_counter() - t0
        pieces += 1
        text.append(piece)
    total = time.perf_counter() - t0
    rate = (pieces - 1) / (total - first) if first is not None and pieces > 1 else None
    return {"first_s": first, "total_s": total, "pieces": pieces, "pieces_per_s": rate,
            "answer": "".join(text)}


def run_tests(model, url, frames, runs):
    out = {"ask": [], "facts": [], "commands": [], "detect": None}

    # Warm-up (not counted): the first request pays allocator and graph setup.
    t0 = time.perf_counter()
    ask_once(fresh(frames, ASK[0][0]), "Hola", model, url)
    out["warmup_s"] = round(time.perf_counter() - t0, 3)

    for r in range(runs):
        for rel, prompt in ASK:
            res = ask_once(fresh(frames, rel), prompt, model, url)
            res.update(image=rel, run=r + 1)
            out["ask"].append(res)
            first = "none" if res["first_s"] is None else f"{res['first_s']:.2f}s"
            print(f"    ask   {os.path.basename(rel):10s} first {first} "
                  f"total {res['total_s']:.2f}s  {res['answer'][:70]!r}", flush=True)

    for rel, question, checks in FACTS:
        res = ask_once(fresh(frames, rel), question, model, url)
        ok = all(re.search(c, res["answer"], re.I) for c in checks)
        out["facts"].append({"image": rel, "question": question, "answer": res["answer"],
                             "ok": ok, "total_s": res["total_s"]})
        print(f"    fact  {'OK ' if ok else 'BAD'} {question[:45]!r} -> "
              f"{res['answer'][:60]!r}", flush=True)

    for text, want in COMMANDS:
        res = command_common.interpret(text, model, robot="g1", url=url, num_ctx=4096,
                                       max_tokens=256)
        ok = res["skill"] == want
        out["commands"].append({"text": text, "want": want, "got": res["skill"],
                                "params": res["params"], "ok": ok,
                                "elapsed_s": res["elapsed_ms"] / 1000})
        print(f"    cmd   {'OK ' if ok else 'BAD'} {text!r} -> {res['skill']} "
              f"{res['params']} ({res['elapsed_ms'] / 1000:.2f}s)", flush=True)

    res = vlm_common.query_vlm(fresh(frames, DETECT_IMAGE), model, scope="all", max_tokens=2048, think=False,
                               url=url, num_ctx=4096, size=frames[DETECT_IMAGE][1])
    objs = res["parsed"].get("objects", []) if res["ok"] and isinstance(res["parsed"], dict) else []
    people = sum(1 for o in objs if "person" in json.dumps(o).lower()
                 or "persona" in json.dumps(o).lower())
    out["detect"] = {"ok": res["ok"], "objects": len(objs), "people": people,
                     "finish": res["finish_reason"], "elapsed_s": res["elapsed"],
                     "out_tokens": res["out_tokens"], "raw": res["content"][:4000]}
    print(f"    detect ok={res['ok']} objects={len(objs)} people={people} "
          f"finish={res['finish_reason']} {res['elapsed']:.2f}s", flush=True)
    return out


def summarize(name, backend, load_s, vram_mib, t):
    ask = t["ask"]
    return {
        "model": name, "backend": backend, "load_s": load_s, "vram_mib": vram_mib,
        "warmup_s": t.get("warmup_s"),
        "first_s": stats([a["first_s"] for a in ask]),
        "total_s": stats([a["total_s"] for a in ask]),
        "pieces_per_s": stats([a["pieces_per_s"] for a in ask]),
        "facts_ok": sum(f["ok"] for f in t["facts"]), "facts_n": len(t["facts"]),
        "commands_ok": sum(c["ok"] for c in t["commands"]), "commands_n": len(t["commands"]),
        "command_s": stats([c["elapsed_s"] for c in t["commands"]]),
        "detect": {k: v for k, v in t["detect"].items() if k != "raw"},
    }


# --------------------------------------------------------------------------- #
# One model under llama-server / under Ollama
# --------------------------------------------------------------------------- #
def bench_llamacpp(name, port, frames, runs, log_dir, min_tokens=0):
    """`min_tokens` > 0 starts the server with --image-min-tokens; the row is then
    labelled name@min_tokens."""
    label = f"{name}@{min_tokens}" if min_tokens else name
    model_file, mmproj_file, _repo = CANDIDATES[name]
    model, mmproj = (os.path.join(MODELS_DIR, f) for f in (model_file, mmproj_file))
    url = f"http://127.0.0.1:{port}"
    os.environ["VLM_BACKEND"] = "llamacpp"

    baseline = vram_used_mib() or 0
    sampler = VramSampler()
    sampler.start()
    log_path = os.path.join(log_dir, f"vlm_server_{label}.log")
    env = {**os.environ, "VLM_PORT": str(port)}
    env.pop("VLM_IMAGE_MIN_TOKENS", None)
    if min_tokens:
        env["VLM_IMAGE_MIN_TOKENS"] = str(min_tokens)
    with open(log_path, "w") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen([SERVER_SH, model, mmproj, name], stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=env)
        try:
            load_s = None
            while proc.poll() is None and time.perf_counter() - t0 < 180:
                try:
                    if requests.get(url + "/health", timeout=1).status_code == 200:
                        load_s = round(time.perf_counter() - t0, 2)
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            if load_s is None:
                raise RuntimeError(f"server did not come up (see {log_path})")
            print(f"  loaded in {load_s}s", flush=True)
            t = run_tests(label, url, frames, runs)
        finally:
            sampler.stop()
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(5)
            wait_vram_back(baseline)
    return summarize(label, "llama-server", load_s, sampler.peak - baseline, t), t


def bench_ollama(tag, frames, runs):
    os.environ["VLM_BACKEND"] = "ollama"
    requests.post(OLLAMA_HOST + "/api/generate", json={"model": tag, "keep_alive": 0},
                  timeout=30)
    baseline = vram_used_mib() or 0
    sampler = VramSampler()
    sampler.start()
    try:
        t0 = time.perf_counter()
        requests.post(OLLAMA_HOST + "/api/generate", json={"model": tag, "keep_alive": "10m"},
                      timeout=180).raise_for_status()
        load_s = round(time.perf_counter() - t0, 2)
        print(f"  loaded in {load_s}s", flush=True)
        t = run_tests(tag, OLLAMA_HOST, frames, runs)
    finally:
        sampler.stop()
        requests.post(OLLAMA_HOST + "/api/generate", json={"model": tag, "keep_alive": 0},
                      timeout=30)
        wait_vram_back(baseline)
    return summarize(tag, "ollama", load_s, sampler.peak - baseline, t), t


# --------------------------------------------------------------------------- #
def print_table(rows):
    def f(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"
    head = (f"{'model':22s} {'backend':12s} {'load':>5s} {'VRAM':>6s} {'1st word':>8s} "
            f"{'answer':>7s} {'tok/s':>6s} {'facts':>6s} {'cmds':>5s} {'cmd s':>6s} "
            f"{'detect':>16s}")
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        d = r["detect"]
        det = f"{'ok' if d['ok'] else 'BAD'} {d['objects']}obj {d['elapsed_s']:.1f}s"
        print(f"{r['model']:22s} {r['backend']:12s} {f(r['load_s'], 1):>5s} "
              f"{(str(r['vram_mib']) + 'M'):>6s} {f(r['first_s']['mean']):>8s} "
              f"{f(r['total_s']['mean']):>7s} {f(r['pieces_per_s']['mean'], 0):>6s} "
              f"{r['facts_ok']}/{r['facts_n']:<4d} {r['commands_ok']}/{r['commands_n']:<3d} "
              f"{f(r['command_s']['mean']):>6s} {det:>16s}")


def main():
    ap = argparse.ArgumentParser(description="Speed and answer quality of each candidate VLM.")
    ap.add_argument("--models", nargs="+", choices=list(CANDIDATES),
                    help="llama-server models to test. Default: every one downloaded.")
    ap.add_argument("--ollama", nargs="*", default=[],
                    help="Ollama tags to test too, as the comparison row.")
    ap.add_argument("--runs", type=int, default=2, help="Rounds of the ASK questions.")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--image-min-tokens", nargs="+", type=int, default=[0],
                    help="One or several --image-min-tokens floors to run each model with; "
                         "0 = the model's own (default). Ollama forces 1024.")
    ap.add_argument("--list", action="store_true", help="Which model files are present.")
    args = ap.parse_args()

    present = {n: all(os.path.isfile(os.path.join(MODELS_DIR, f)) for f in c[:2])
               for n, c in CANDIDATES.items()}
    if args.list:
        for n, c in CANDIDATES.items():
            print(f"  {'present' if present[n] else 'MISSING':8s} {n:18s} {c[2]}")
        return
    names = args.models or [n for n in CANDIDATES if present[n]]
    missing = [n for n in names if not present[n]]
    if missing:
        sys.exit(f"[ERROR] model files missing for: {', '.join(missing)} (see --list)")

    frames = {rel: load_frame(rel) for rel in
              {r for r, _ in ASK} | {r for r, _, _ in FACTS} | {DETECT_IMAGE}}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, detail = [], {}
    jobs = [("ollama", t, 0) for t in args.ollama] + [
        ("llamacpp", n, mt) for n in names for mt in args.image_min_tokens]
    for kind, name, mt in jobs:
        print(f"\n=== {name}{f' @{mt} image tokens' if mt else ''} ({kind}) ===", flush=True)
        try:
            if kind == "ollama":
                row, t = bench_ollama(name, frames, args.runs)
            else:
                row, t = bench_llamacpp(name, args.port, frames, args.runs,
                                        os.path.dirname(results_path("x")), mt)
        except Exception as e:  # one broken model must not sink the rest of the run
            print(f"  [!] {name}: {e}", flush=True)
            detail[f"{name}@{mt}" if mt else name] = {"error": str(e)}
            continue
        rows.append(row)
        detail[row["model"]] = t

    if rows:
        print_table(rows)
    out = results_path(f"vlm_server_benchmark_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": stamp, "max_side": MAX_SIDE, "runs": args.runs,
                   "summary": rows, "detail": detail}, fh, indent=2, ensure_ascii=False)
        fh.write("\n")  # pre-commit's end-of-file-fixer
    print(f"\n[OK] Saved to {out}")


if __name__ == "__main__":
    main()
