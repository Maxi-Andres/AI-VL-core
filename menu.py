#!/usr/bin/env python3
"""
menu.py — Interactive menu for the PoC (VLM and YOLO).

Run THIS file (IDE Play button or `python3 menu.py`). It first asks which path
you want:

  - VLM  : the Ollama vision-language model (reasons over a prompt, returns JSON).
  - YOLO : the Ultralytics detector that runs in-process (what the real
           deployment runs on the live video stream).

Each path then offers the same two ways to analyze:

  1) SCAN: one image, prints the result + JSON (YOLO also saves the boxed image).
  2) BENCHMARK: a submenu to choose WHICH images, WHICH models, how many runs,
     and the per-path knobs (VLM: prompt/tokens/ctx/think; YOLO: imgsz/conf),
     then runs the cartesian product with a progress bar and a timing report
     (per image, total, average, P50/P95).

Choices are saved to a single config.json (VLM keys + yolo_* keys), so next time
it starts with whatever you used last. To use it WITHOUT the menu (command line
with flags), see README.md.

Requirements:  pip install requests   (YOLO path also: pip install ultralytics)
"""
import os
import sys

# The code lives in src/. We add it to the path so it can be imported whether
# you run `python3 menu.py` from the root or from the IDE (Play button).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import requests

from vlm_common import (
    PROMPT_VARIANTS,
    SCOPES,
    list_images,
    model_supports_thinking,
    save_config,
)
# yolo_common.load_config is a superset of the VLM one: it loads the shared
# config.json AND fills in the YOLO-only defaults, so the single menu can drive
# both paths off one config object.
from yolo_common import (
    KNOWN_MODELS,
    list_models as list_yolo_models,
    load_config,
    ultralytics_available,
)
from vlm_scan import run_vlm_scan
from vlm_benchmark import run_benchmark
from yolo_scan import run_yolo_scan
from yolo_benchmark import run_yolo_benchmark


# --------------------------------------------------------------------------- #
# UI helpers (console input)
# --------------------------------------------------------------------------- #
def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        raise SystemExit(0)


def list_ollama_models(url):
    """List the installed models by querying /api/tags. [] on failure."""
    host = url.split("/v1")[0]
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, KeyError, ValueError):
        return []


def choose_from_list(title, options, current):
    """Numbered menu (single selection). Empty Enter = keep the current one."""
    print(f"\n{title}  (current: {current})")
    for i, opt in enumerate(options, 1):
        marca = " <-- current" if opt == current else ""
        print(f"  {i}) {opt}{marca}")
    print("  0) type a value manually")
    sel = ask("Choose a number (Enter = keep current): ")
    if sel == "":
        return current
    if sel == "0":
        return ask("Value: ") or current
    if sel.isdigit() and 1 <= int(sel) <= len(options):
        return options[int(sel) - 1]
    print("[!] Invalid option, keeping the current one.")
    return current


def choose_multi(title, options, current):
    """MULTIPLE selection. Returns a list (a subset of `options`).

    Accepts: '1,3,5'  |  ranges '1-4'  |  combined '1-3,7'  |  'all'  |  'none'.
    Empty Enter = keep the current selection.
    """
    print(f"\n{title}")
    cur = set(current)
    for i, opt in enumerate(options, 1):
        marca = " *" if opt in cur else ""
        print(f"  {i}) {opt}{marca}")
    print("  (items marked with * are the current selection)")
    sel = ask("Choose (e.g.: 1,3,5  or  1-4,7  or  'all'  or  'none'; Enter = keep): ")
    if sel == "":
        return current
    low = sel.lower()
    if low in ("all", "*"):
        return list(options)
    if low in ("none", "0"):
        return []
    chosen = []
    for part in sel.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                for k in range(int(a), int(b) + 1):
                    if 1 <= k <= len(options):
                        chosen.append(options[k - 1])
        elif part.isdigit():
            k = int(part)
            if 1 <= k <= len(options):
                chosen.append(options[k - 1])
    # dedup while preserving order
    seen, out = set(), []
    for o in chosen:
        if o not in seen:
            seen.add(o)
            out.append(o)
    if not out:
        print("[!] Could not parse the selection, keeping the current one.")
        return current
    return out


# --------------------------------------------------------------------------- #
# CONFIG actions (smoke test)
# --------------------------------------------------------------------------- #
def think_note(model, url):
    """A one-liner saying whether THIS model reasons (capability 'thinking') or not."""
    if model_supports_thinking(model, url):
        return (f"  ℹ '{model}' DOES reason (capability 'thinking'). NOTE: on qwen3-vl + "
                f"Ollama 0.30.6 the think=OFF flag is ignored, it always reasons.")
    return (f"  ℹ '{model}' does NOT reason (no 'thinking' capability): reasoning is "
            f"truly OFF. This is the model to use if you do NOT want it to think.")


def pick_model(cfg):
    models = list_ollama_models(cfg["url"])
    if not models:
        print("[!] Could not list Ollama models (is it running?). Type it manually.")
        cfg["model"] = ask(f"Model (current: {cfg['model']}): ") or cfg["model"]
    else:
        cfg["model"] = choose_from_list("Ollama model", models, cfg["model"])
    print(think_note(cfg["model"], cfg["url"]))


def pick_image(cfg):
    folder = ask(f"Image folder (Enter = {cfg['folder']}): ") or cfg["folder"]
    cfg["folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No images in {folder}. Type the path manually.")
        cfg["image"] = ask(f"Image (current: {cfg['image']}): ") or cfg["image"]
    else:
        cfg["image"] = choose_from_list("Image", imgs, cfg["image"])


def pick_scope(cfg, key="scope"):
    keys = list(SCOPES)
    labels = [f"{k} — {SCOPES[k]['label']}" for k in keys]
    current = cfg.get(key, keys[0])
    current_label = f"{current} — {SCOPES[current]['label']}"
    chosen = choose_from_list("Detection mode", labels, current_label)
    cfg[key] = chosen.split(" — ")[0].strip()
    if cfg[key] not in SCOPES:
        cfg[key] = keys[0]


def pick_variant(cfg, scope_key="scope", variant_key="variant"):
    """Choose the active prompt variant for the given scope."""
    scope = cfg[scope_key]
    variants = list(PROMPT_VARIANTS[scope])
    current = cfg.get(variant_key) if cfg.get(variant_key) in variants else variants[0]
    chosen = choose_from_list(f"Prompt variant (scope: {scope})", variants, current)
    cfg[variant_key] = chosen


def explain_ctx():
    print("\n  ── max_tokens vs num_ctx ─────────────────────────────────────────")
    print("  • num_ctx    = the FULL context window: input (system+user+")
    print("                 image) + output (reasoning+answer). It's the number")
    print("                 you see in `ollama ps`. Higher num_ctx = the image goes")
    print("                 in at higher resolution (more detail) but prefill is SLOWER.")
    print("  • max_tokens = ceiling on what the model GENERATES (reasoning+answer),")
    print("                 i.e. num_predict. If it hits this, it cuts off -> finish_reason")
    print("                 'length' and the JSON may come back empty.")
    print("  • They combine: real output = min(max_tokens, num_ctx − input_tokens).")
    print("                 If num_ctx is too small, the image eats into the answer's")
    print("                 space and it stalls; if too large, it takes longer.")
    print("  ──────────────────────────────────────────────────────────────────")


def set_ctx(cfg, mt_key="max_tokens", nc_key="num_ctx"):
    explain_ctx()
    v = ask(f"max_tokens (current: {cfg[mt_key]}, Enter = keep): ")
    if v.isdigit():
        cfg[mt_key] = int(v)
    c = ask(f"num_ctx (current: {cfg[nc_key]}, Enter = keep): ")
    if c.isdigit():
        cfg[nc_key] = int(c)


def toggle_think(cfg, key="think", models=None):
    cfg[key] = not cfg[key]
    state = "ON" if cfg[key] else "OFF"
    print(f"-> Reasoning (think): {state}.")
    # Tell the truth based on the models in play (the flag only matters if the model supports it).
    models = models or [cfg.get("model")]
    thinkers = [m for m in models if m and model_supports_thinking(m, cfg["url"])]
    nonthinkers = [m for m in models if m and m not in thinkers]
    if thinkers:
        print(f"   NOTE: {', '.join(thinkers)} reason(s) by design (qwen3-vl). On Ollama "
              f"0.30.6 the think=OFF flag does NOT turn it off: they will reason anyway.")
        print(f"   The REAL switch is the model: if you do NOT want it to think, use one "
              f"without 'thinking' (e.g. qwen2.5vl:7b).")
    if nonthinkers:
        print(f"   {', '.join(nonthinkers)} has/have no 'thinking': never reason(s), "
              f"truly OFF.")


def as_list(v):
    """Normalize scalar/list/None to a list (for old configs with loose values)."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def int_list(raw):
    """'4096, 8192' or '4096 8192' -> [4096, 8192]. [] if nothing is parseable."""
    parts = raw.replace(",", " ").split()
    return [int(p) for p in parts if p.lstrip("-").isdigit()]


def bench_set_ctx(cfg):
    """Benchmark max_tokens / num_ctx: accept SEVERAL values to compare them."""
    explain_ctx()
    print("  (you can enter SEVERAL values separated by comma/space to compare them,")
    print("   e.g.: '4096 8192' runs both and puts them in the table)")
    cur_mt = cfg.get("benchmark_max_tokens")
    vals = int_list(ask(f"max_tokens to compare (current: {cur_mt}, Enter = keep): "))
    if vals:
        cfg["benchmark_max_tokens"] = vals
    cur_nc = cfg.get("benchmark_num_ctx")
    vals = int_list(ask(f"num_ctx to compare (current: {cur_nc}, Enter = keep): "))
    if vals:
        cfg["benchmark_num_ctx"] = vals


def bench_pick_think(cfg):
    """Which think values to compare in the benchmark: ON, OFF, or both."""
    print(f"\nWhich reasoning (think) values to compare?  (current: {cfg.get('benchmark_think')})")
    print("  1) ON only")
    print("  2) OFF only")
    print("  3) Both (ON and OFF) — compares the two in the same run")
    mapping = {"1": [True], "2": [False], "3": [True, False]}
    sel = ask("Choose a number (Enter = keep): ")
    if sel in mapping:
        cfg["benchmark_think"] = mapping[sel]
    print("  ℹ remember: the think flag only changes anything on models that support it. "
          "qwen3-vl ignores OFF (always reasons); qwen2.5vl never reasons. This dimension "
          "is mostly useful for when you get a model that DOES honor the flag.")


def show_config(cfg):
    print("\n" + "=" * 52)
    print(" CURRENT CONFIG (config.json)")
    print("=" * 52)
    print("  [Scan]")
    print(f"    Model            : {cfg['model']}")
    print(f"    Image            : {cfg['image']}")
    print(f"    Detection mode   : {cfg['scope']} ({SCOPES[cfg['scope']]['label']})")
    print(f"    Prompt variant   : {cfg.get('variant', '(default)')}")
    print(f"    think            : {'ON' if cfg['think'] else 'OFF'}")
    print(f"    max_tokens/num_ctx: {cfg['max_tokens']} / {cfg['num_ctx']}")
    print("  [Benchmark]")
    n_img = cfg.get("benchmark_images") or "ALL"
    print(f"    Folder           : {cfg['folder']}")
    print(f"    Images           : {n_img}")
    print(f"    Models           : {', '.join(cfg['benchmark_models'])}")
    print(f"    Runs/image       : {cfg['benchmark_runs']}")
    variants = cfg.get("benchmark_variants") or [cfg.get("benchmark_variant")]
    print(f"    Mode / prompts   : {cfg.get('benchmark_scope')} / {', '.join(v for v in variants if v)}")
    think_vals = as_list(cfg.get("benchmark_think"))
    print(f"    think to compare : {['ON' if t else 'OFF' for t in think_vals]}")
    print(f"    max_tokens (list): {as_list(cfg.get('benchmark_max_tokens'))}")
    print(f"    num_ctx (list)   : {as_list(cfg.get('benchmark_num_ctx'))}")
    print(f"  Ollama URL         : {cfg['url']}")
    print("=" * 52)


# --------------------------------------------------------------------------- #
# BENCHMARK submenu
# --------------------------------------------------------------------------- #
def bench_pick_images(cfg):
    folder = ask(f"Image folder (Enter = {cfg['folder']}): ") or cfg["folder"]
    cfg["folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No images in {folder}.")
        return
    names = [os.path.basename(p) for p in imgs]
    current = cfg.get("benchmark_images") or names  # [] meant ALL
    chosen = choose_multi(f"Benchmark images in '{folder}'", names, current)
    # If all were chosen, save [] (= ALL, adapts if you add photos).
    cfg["benchmark_images"] = [] if set(chosen) == set(names) else chosen
    sel = cfg["benchmark_images"] or names
    print(f"-> {len(sel)} image(s) selected.")


def bench_pick_models(cfg):
    models = list_ollama_models(cfg["url"])
    if not models:
        print("[!] Could not list Ollama models. Type them separated by commas.")
        raw = ask(f"Models (Enter = {', '.join(cfg['benchmark_models'])}): ")
        if raw:
            cfg["benchmark_models"] = [m.strip() for m in raw.split(",") if m.strip()]
        return
    chosen = choose_multi("Models to compare", models, cfg["benchmark_models"])
    if chosen:
        cfg["benchmark_models"] = chosen
    for m in cfg["benchmark_models"]:
        print(think_note(m, cfg["url"]))


def bench_set_runs(cfg):
    runs = ask(f"Runs per image (current: {cfg['benchmark_runs']}): ")
    if runs.isdigit() and int(runs) >= 1:
        cfg["benchmark_runs"] = int(runs)


def bench_pick_variants(cfg):
    """MULTIPLE selection of prompt variants to compare in the benchmark.

    Comparing several prompts in one run replaces the old 05_prompt_test.py.
    """
    scope = cfg.get("benchmark_scope", "industrial")
    options = list(PROMPT_VARIANTS[scope])
    current = [v for v in (cfg.get("benchmark_variants") or []) if v in options]
    if not current:
        current = options[:1]
    chosen = choose_multi(f"Prompt variants to compare (scope: {scope})", options, current)
    cfg["benchmark_variants"] = chosen or current


def bench_run(cfg):
    """Resolve the chosen images and fire off the benchmark."""
    all_imgs = list_images(cfg["folder"])
    if not all_imgs:
        print(f"[!] No images in {cfg['folder']}.")
        return
    sel = set(cfg.get("benchmark_images") or [])
    images = [p for p in all_imgs if os.path.basename(p) in sel] if sel else all_imgs
    run_benchmark(images, cfg["benchmark_models"], runs=cfg["benchmark_runs"],
                  scope=cfg.get("benchmark_scope", "industrial"),
                  max_tokens=cfg.get("benchmark_max_tokens", 4096),
                  think=cfg.get("benchmark_think", True), url=cfg["url"],
                  num_ctx=cfg.get("benchmark_num_ctx", 8192),
                  variants=cfg.get("benchmark_variants"))


BENCH_MENU = """
┌────────────────────────────────────────────────┐
│              BENCHMARK — configure             │
├────────────────────────────────────────────────┤
│   1) Choose images (which / how many)          │
│   2) Choose models                             │
│   3) Runs per image                            │
│   4) Detection mode (industrial / all)         │
│   5) Prompts to compare (1 or several)         │
│   6) max_tokens / num_ctx (1 or several each)  │
│   7) think to compare (ON / OFF / both)        │
│                                                │
│   8) ▶ RUN BENCHMARK                           │
│   0) Back to main menu                         │
└────────────────────────────────────────────────┘"""


def benchmark_menu(cfg):
    while True:
        # Quick summary of what's going to run.
        all_imgs = list_images(cfg["folder"])
        n_img = len(cfg.get("benchmark_images") or all_imgs)
        runs = cfg["benchmark_runs"]
        variants = cfg.get("benchmark_variants") or []
        n_models = len(cfg["benchmark_models"])
        # The sweep dimensions can be lists; we count their lengths.
        mt = as_list(cfg.get("benchmark_max_tokens"))
        nc = as_list(cfg.get("benchmark_num_ctx"))
        th = as_list(cfg.get("benchmark_think"))
        n_combo = n_models * max(len(variants), 1) * len(mt) * len(nc) * len(th)
        n_calls = n_img * runs * n_combo
        print(f"\n  >> {n_img} img × {runs} runs × {n_combo} combos "
              f"({n_models} mod × {len(variants)} prompt × {len(mt)} maxtok × {len(nc)} ctx "
              f"× {len(th)} think) = {n_calls} calls")
        print(f"     maxtok={mt} | num_ctx={nc} | think={['ON' if x else 'OFF' for x in th]} "
              f"| prompts={', '.join(variants)}")
        print(BENCH_MENU)
        choice = ask("Option: ")
        if choice == "1":
            bench_pick_images(cfg); save_config(cfg)
        elif choice == "2":
            bench_pick_models(cfg); save_config(cfg)
        elif choice == "3":
            bench_set_runs(cfg); save_config(cfg)
        elif choice == "4":
            pick_scope(cfg, key="benchmark_scope")
            # If the variants no longer exist for the new scope, reset.
            scope = cfg["benchmark_scope"]
            valid = list(PROMPT_VARIANTS[scope])
            kept = [v for v in (cfg.get("benchmark_variants") or []) if v in valid]
            cfg["benchmark_variants"] = kept or valid[:1]
            save_config(cfg)
        elif choice == "5":
            bench_pick_variants(cfg)
            save_config(cfg)
        elif choice == "6":
            bench_set_ctx(cfg)
            save_config(cfg)
        elif choice == "7":
            bench_pick_think(cfg)
            save_config(cfg)
        elif choice == "8":
            save_config(cfg)
            bench_run(cfg)
        elif choice == "0" or choice.lower() in ("q", "back"):
            return
        else:
            print("[!] Invalid option.")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
MENU = """
┌────────────────────────────────────────────────┐
│                VLM — menu                      │
├────────────────────────────────────────────────┤
│  ANALYZE                                       │
│   1) Scan (1 image, live reasoning)            │
│   2) Benchmark (models × prompts, submenu)     │
│                                                │
│  CONFIGURE SCAN (saved to config)              │
│   3) Model                                     │
│   4) Image                                     │
│   5) Detection mode (industrial / all)         │
│   6) Prompt variant                            │
│   7) Reasoning think (ON/OFF)                  │
│   8) max_tokens / num_ctx                      │
│   9) Show current config                       │
│                                                │
│   0) Back (path selection)                     │
└────────────────────────────────────────────────┘"""


def vlm_menu(cfg):
    """Interactive menu for the VLM (Ollama) path."""
    show_config(cfg)
    while True:
        print(MENU)
        choice = ask("Option: ")

        if choice == "1":
            save_config(cfg)
            run_vlm_scan(cfg["image"], cfg["model"], scope=cfg["scope"],
                         max_tokens=cfg["max_tokens"], think=cfg["think"],
                         url=cfg["url"], num_ctx=cfg["num_ctx"],
                         variant=cfg.get("variant"))
        elif choice == "2":
            benchmark_menu(cfg)
        elif choice == "3":
            pick_model(cfg); save_config(cfg)
        elif choice == "4":
            pick_image(cfg); save_config(cfg)
        elif choice == "5":
            pick_scope(cfg); save_config(cfg)
        elif choice == "6":
            pick_variant(cfg); save_config(cfg)
        elif choice == "7":
            toggle_think(cfg); save_config(cfg)
        elif choice == "8":
            set_ctx(cfg); save_config(cfg)
        elif choice == "9":
            show_config(cfg)
        elif choice == "0" or choice.lower() in ("q", "back"):
            save_config(cfg)
            return
        else:
            print("[!] Invalid option.")


# --------------------------------------------------------------------------- #
# YOLO path: config actions + menus
# --------------------------------------------------------------------------- #
def float_list(raw):
    """'0.25, 0.5' or '0.25 0.5' -> [0.25, 0.5]. [] if nothing is parseable."""
    parts = raw.replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def yolo_pick_model(cfg):
    models = list_yolo_models()
    cfg["yolo_model"] = choose_from_list("YOLO weights (.pt)", models, cfg["yolo_model"])
    print("  ℹ pretrained weights are auto-downloaded on first use. "
          "Smaller (n < s < m < l < x) = faster, less accurate.")


def yolo_pick_image(cfg):
    folder = ask(f"Image folder (Enter = {cfg['yolo_folder']}): ") or cfg["yolo_folder"]
    cfg["yolo_folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No images in {folder}. Type the path manually.")
        cfg["yolo_image"] = ask(f"Image (current: {cfg['yolo_image']}): ") or cfg["yolo_image"]
    else:
        cfg["yolo_image"] = choose_from_list("Image", imgs, cfg["yolo_image"])


def yolo_set_params(cfg):
    print("\n  ── conf vs imgsz ───────────────────────────────────────────────")
    print("  • conf  = confidence threshold; detections below it are dropped.")
    print("            Lower = more (and more false) boxes; higher = stricter.")
    print("  • imgsz = inference image size (longer side). Bigger = more detail")
    print("            on small objects but slower. Common: 640, 1280.")
    print("  ────────────────────────────────────────────────────────────────")
    v = ask(f"conf (current: {cfg['yolo_conf']}, Enter = keep): ")
    try:
        if v != "":
            cfg["yolo_conf"] = float(v)
    except ValueError:
        print("[!] Not a number, keeping the current one.")
    s = ask(f"imgsz (current: {cfg['yolo_imgsz']}, Enter = keep): ")
    if s.isdigit():
        cfg["yolo_imgsz"] = int(s)


def yolo_show_config(cfg):
    print("\n" + "=" * 52)
    print(" CURRENT YOLO CONFIG (config.json)")
    print("=" * 52)
    print("  [Scan]")
    print(f"    Model            : {cfg['yolo_model']}")
    print(f"    Image            : {cfg['yolo_image']}")
    print(f"    conf / imgsz     : {cfg['yolo_conf']} / {cfg['yolo_imgsz']}")
    print(f"    Save annotated   : {'ON' if cfg.get('yolo_save', True) else 'OFF'}")
    print("  [Benchmark]")
    n_img = cfg.get("yolo_benchmark_images") or "ALL"
    print(f"    Folder           : {cfg['yolo_folder']}")
    print(f"    Images           : {n_img}")
    print(f"    Models           : {', '.join(cfg['yolo_benchmark_models'])}")
    print(f"    Runs/image       : {cfg['yolo_benchmark_runs']}")
    print(f"    conf (list)      : {as_list(cfg.get('yolo_benchmark_conf'))}")
    print(f"    imgsz (list)     : {as_list(cfg.get('yolo_benchmark_imgsz'))}")
    print("=" * 52)


def yolo_bench_pick_images(cfg):
    folder = ask(f"Image folder (Enter = {cfg['yolo_folder']}): ") or cfg["yolo_folder"]
    cfg["yolo_folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No images in {folder}.")
        return
    names = [os.path.basename(p) for p in imgs]
    current = cfg.get("yolo_benchmark_images") or names
    chosen = choose_multi(f"Benchmark images in '{folder}'", names, current)
    cfg["yolo_benchmark_images"] = [] if set(chosen) == set(names) else chosen
    sel = cfg["yolo_benchmark_images"] or names
    print(f"-> {len(sel)} image(s) selected.")


def yolo_bench_pick_models(cfg):
    models = list_yolo_models()
    chosen = choose_multi("YOLO models to compare", models, cfg["yolo_benchmark_models"])
    if chosen:
        cfg["yolo_benchmark_models"] = chosen


def yolo_bench_set_runs(cfg):
    runs = ask(f"Runs per image (current: {cfg['yolo_benchmark_runs']}): ")
    if runs.isdigit() and int(runs) >= 1:
        cfg["yolo_benchmark_runs"] = int(runs)


def yolo_bench_set_params(cfg):
    print("  (you can enter SEVERAL values separated by comma/space to compare them)")
    vals = float_list(ask(f"conf to compare (current: {cfg.get('yolo_benchmark_conf')}, Enter = keep): "))
    if vals:
        cfg["yolo_benchmark_conf"] = vals
    vals = int_list(ask(f"imgsz to compare (current: {cfg.get('yolo_benchmark_imgsz')}, Enter = keep): "))
    if vals:
        cfg["yolo_benchmark_imgsz"] = vals


def yolo_bench_run(cfg):
    all_imgs = list_images(cfg["yolo_folder"])
    if not all_imgs:
        print(f"[!] No images in {cfg['yolo_folder']}.")
        return
    sel = set(cfg.get("yolo_benchmark_images") or [])
    images = [p for p in all_imgs if os.path.basename(p) in sel] if sel else all_imgs
    run_yolo_benchmark(images, cfg["yolo_benchmark_models"],
                       runs=cfg["yolo_benchmark_runs"],
                       conf=cfg.get("yolo_benchmark_conf", [0.25]),
                       imgsz=cfg.get("yolo_benchmark_imgsz", [640]))


YOLO_BENCH_MENU = """
┌────────────────────────────────────────────────┐
│           YOLO BENCHMARK — configure           │
├────────────────────────────────────────────────┤
│   1) Choose images (which / how many)          │
│   2) Choose models                             │
│   3) Runs per image                            │
│   4) conf / imgsz (1 or several each)          │
│                                                │
│   5) ▶ RUN BENCHMARK                           │
│   0) Back to YOLO menu                         │
└────────────────────────────────────────────────┘"""


def yolo_benchmark_menu(cfg):
    while True:
        all_imgs = list_images(cfg["yolo_folder"])
        n_img = len(cfg.get("yolo_benchmark_images") or all_imgs)
        runs = cfg["yolo_benchmark_runs"]
        n_models = len(cfg["yolo_benchmark_models"])
        cf = as_list(cfg.get("yolo_benchmark_conf"))
        sz = as_list(cfg.get("yolo_benchmark_imgsz"))
        n_combo = n_models * len(cf) * len(sz)
        n_calls = n_img * runs * n_combo
        print(f"\n  >> {n_img} img × {runs} runs × {n_combo} combos "
              f"({n_models} mod × {len(sz)} imgsz × {len(cf)} conf) = {n_calls} calls")
        print(f"     imgsz={sz} | conf={cf}")
        print(YOLO_BENCH_MENU)
        choice = ask("Option: ")
        if choice == "1":
            yolo_bench_pick_images(cfg); save_config(cfg)
        elif choice == "2":
            yolo_bench_pick_models(cfg); save_config(cfg)
        elif choice == "3":
            yolo_bench_set_runs(cfg); save_config(cfg)
        elif choice == "4":
            yolo_bench_set_params(cfg); save_config(cfg)
        elif choice == "5":
            save_config(cfg)
            yolo_bench_run(cfg)
        elif choice == "0" or choice.lower() in ("q", "back"):
            return
        else:
            print("[!] Invalid option.")


YOLO_MENU = """
┌────────────────────────────────────────────────┐
│                YOLO — menu                     │
├────────────────────────────────────────────────┤
│  ANALYZE                                       │
│   1) Scan (1 image, boxes + JSON)              │
│   2) Benchmark (models × imgsz × conf, submenu)│
│                                                │
│  CONFIGURE SCAN (saved to config)              │
│   3) Model (weights .pt)                       │
│   4) Image                                     │
│   5) conf / imgsz                              │
│   6) Save annotated image (ON/OFF)             │
│   7) Show current config                       │
│                                                │
│   0) Back (path selection)                     │
└────────────────────────────────────────────────┘"""


def yolo_toggle_save(cfg):
    cfg["yolo_save"] = not cfg.get("yolo_save", True)
    state = "ON" if cfg["yolo_save"] else "OFF"
    print(f"-> Save annotated (boxed) image on a scan: {state}.")
    if cfg["yolo_save"]:
        print("   Annotated images go to results/annotated/<image>__<model>.jpg")


def yolo_menu(cfg):
    """Interactive menu for the YOLO (Ultralytics) path."""
    if not ultralytics_available():
        print("\n[!] The 'ultralytics' package is not installed, so the YOLO path "
              "cannot run yet.\n    Install it with:  pip install ultralytics")
        print("    (You can still browse/configure this menu; runs will report the error.)")
    yolo_show_config(cfg)
    while True:
        print(YOLO_MENU)
        choice = ask("Option: ")
        if choice == "1":
            save_config(cfg)
            run_yolo_scan(cfg["yolo_image"], cfg["yolo_model"],
                          conf=cfg["yolo_conf"], imgsz=cfg["yolo_imgsz"],
                          save=cfg.get("yolo_save", True))
        elif choice == "2":
            yolo_benchmark_menu(cfg)
        elif choice == "3":
            yolo_pick_model(cfg); save_config(cfg)
        elif choice == "4":
            yolo_pick_image(cfg); save_config(cfg)
        elif choice == "5":
            yolo_set_params(cfg); save_config(cfg)
        elif choice == "6":
            yolo_toggle_save(cfg); save_config(cfg)
        elif choice == "7":
            yolo_show_config(cfg)
        elif choice == "0" or choice.lower() in ("q", "back"):
            save_config(cfg)
            return
        else:
            print("[!] Invalid option.")


# --------------------------------------------------------------------------- #
# Top-level path selection (VLM vs YOLO)
# --------------------------------------------------------------------------- #
PATH_MENU = """
╔════════════════════════════════════════════════╗
║              PoC — choose a path               ║
╠════════════════════════════════════════════════╣
║   1) VLM  (Ollama vision-language model)        ║
║   2) YOLO (Ultralytics detector, in-process)    ║
║                                                ║
║   0) Exit                                      ║
╚════════════════════════════════════════════════╝"""


def main():
    cfg = load_config()

    while True:
        print(PATH_MENU)
        choice = ask("Path: ")
        if choice == "1":
            vlm_menu(cfg)
        elif choice == "2":
            yolo_menu(cfg)
        elif choice == "0" or choice.lower() in ("q", "exit"):
            save_config(cfg)
            print("Config saved. Bye!")
            break
        else:
            print("[!] Invalid option.")


if __name__ == "__main__":
    main()
