#!/usr/bin/env python3
"""
menu.py — Interactive menu for the VLM PoC.

Run THIS file (IDE Play button or `python3 menu.py`) and choose everything from
a menu. There are two ways to analyze:

  1) ANALYZE (smoke test): one image, prints the reasoning live + JSON.
  2) BENCHMARK: opens a submenu where you choose WHICH images, WHICH models, how
     many runs, which prompt and which context, and runs everything with a
     progress bar and a timing report (per image, total, average, P50/P95) + % JSON.

Choices are saved to config.json, so next time it starts with whatever you used
last. To use it WITHOUT the menu (command line with flags), see README.md.

Requirements:  pip install requests
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
    load_config,
    model_supports_thinking,
    save_config,
)
from smoke_test import run_smoke
from benchmark import run_benchmark


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
    print("  [Smoke test]")
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
│             VLM PoC — Main menu                │
├────────────────────────────────────────────────┤
│  ANALYZE                                       │
│   1) Smoke test (1 image, live reasoning)      │
│   2) Benchmark (models × prompts, submenu)     │
│                                                │
│  CONFIGURE SMOKE TEST (saved to config)        │
│   3) Model                                     │
│   4) Image                                     │
│   5) Detection mode (industrial / all)         │
│   6) Prompt variant                            │
│   7) Reasoning think (ON/OFF)                  │
│   8) max_tokens / num_ctx                      │
│   9) Show current config                       │
│                                                │
│   0) Exit                                      │
└────────────────────────────────────────────────┘"""


def main():
    cfg = load_config()
    show_config(cfg)

    while True:
        print(MENU)
        choice = ask("Option: ")

        if choice == "1":
            save_config(cfg)
            run_smoke(cfg["image"], cfg["model"], scope=cfg["scope"],
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
        elif choice == "0" or choice.lower() in ("q", "exit"):
            save_config(cfg)
            print("Config saved. Bye!")
            break
        else:
            print("[!] Invalid option.")


if __name__ == "__main__":
    main()
