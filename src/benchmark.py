#!/usr/bin/env python3
"""
benchmark.py — Latency benchmark (P50/P95), valid-JSON rate and prompt A/B testing.

Runs the SAME set of images, N times per image, sweeping the product
MODELS × PROMPT VARIANTS. It serves two purposes at once:

  1) Compare MODELS (qwen3-vl:8b vs :4b vs qwen2.5vl:7b) with lab data and
     decide the primary VLM for the PoC (F1.8 criteria: precision, P95 latency,
     valid-JSON rate for VLM->VLA).
  2) Compare PROMPT VARIANTS (v1_original vs v2_antiloop, etc.) to choose
     which one to keep active. This replaces the old 05_prompt_test.py: now
     different prompts are tested just like different models, all together.

What it shows while running:
  - a live PROGRESS BAR (which model/prompt/image is running, % complete, ETA);
And when it finishes:
  - time per image, total time, average time per call, P50/P95;
  - valid-JSON rate (the VLM->VLA contract), length truncations and average objects;
  - a verdict with the best combination when you compare more than one.

Results are saved to results/ (not loose in the project root).

Don't want to type flags? Run `python3 menu.py` -> option 2 (Benchmark),
which opens a submenu to choose images, models, prompts, runs and context.

Requirements:  pip install requests
Usage:
    python3 src/benchmark.py fotos/clean --runs 3
    python3 src/benchmark.py fotos/clean --models qwen3-vl:8b qwen3-vl:4b
    python3 src/benchmark.py fotos/clean --variants v1_original v2_antiloop
    python3 src/benchmark.py fotos/clean --images 1.jpeg 14.jpeg 16.jpeg
    python3 src/benchmark.py fotos/clean --scope all
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

from vlm_common import (
    DEFAULT_VARIANT,
    OLLAMA_HOST,
    PROMPT_VARIANTS,
    SCOPES,
    encode_image,
    fmt_secs,
    image_size,
    list_images,
    load_config,
    model_supports_thinking,
    progress_bar,
    query_vlm,
    results_path,
)


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def dedup(seq):
    """Remove duplicates while preserving order of appearance."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def as_list(v, fallback):
    """Normalize a value (scalar or list) to a non-empty, deduplicated list.

    This lets the sweep dimensions (max_tokens, num_ctx, think) accept either a
    single value or a list. If it ends up empty, use `fallback` (a list).
    """
    if v is None:
        items = []
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [v]
    items = dedup(items)
    return items or list(fallback)


def parse_bool(s):
    """Convert 'true/false/on/off/1/0/yes/no/si' (or a bool) to bool. None if not understood."""
    if isinstance(s, bool):
        return s
    t = str(s).strip().lower()
    if t in ("true", "on", "1", "yes", "si", "sí", "y"):
        return True
    if t in ("false", "off", "0", "no", "n"):
        return False
    return None


def combo_label(c):
    """Compact, unique label for a sweep combination."""
    return (f"{c['model']} [{c['variant']}] "
            f"ctx{c['num_ctx']} max{c['max_tokens']} think{'1' if c['think'] else '0'}")


def _stats(latencies, valid, trunc, objs, errors, total):
    """Build the metrics dict for a combination (model × variant)."""
    return {
        "p50": pctl(latencies, 50),
        "p95": pctl(latencies, 95),
        "mean": statistics.mean(latencies) if latencies else float("nan"),
        "min": min(latencies) if latencies else float("nan"),
        "max": max(latencies) if latencies else float("nan"),
        "total_time": sum(latencies),
        "json_rate": (valid / total * 100) if total else 0.0,
        "truncated": trunc,
        "avg_objects": statistics.mean(objs) if objs else 0.0,
        "errors": errors,
        "n": total,
    }


def run_benchmark(images, models, runs=3, scope="industrial", max_tokens=4096,
                  think=True, url=OLLAMA_HOST, num_ctx=8192, variants=None,
                  out=None):
    """Run the benchmark sweeping the product models × prompts × max_tokens × num_ctx × think.

    `images`     : list of concrete paths (already chosen). If you have a folder,
                   expand it first with vlm_common.list_images().
    `models`     : list of Ollama models to compare.
    `variants`   : list of prompt variants (keys of PROMPT_VARIANTS for the
                   scope). Empty/None -> the scope's default variant.
    `max_tokens` : scalar or LIST of output caps to compare.
    `num_ctx`    : scalar or LIST of context windows to compare.
    `think`      : scalar or LIST of bools to compare (useful when you get a
                   model that DOES respect the flag; see model_supports_thinking()).

    Each combination is one row in the report. Prints times + JSON% + truncations +
    objects per combination and a final verdict. Returns the results dict
    (also saves it to `out`, inside results/ if no path is passed).
    """
    images = [p for p in images if os.path.exists(p)]
    if not images:
        print("[ERROR] No valid images to run.", file=sys.stderr)
        return None
    if not models:
        print("[ERROR] No models to run.", file=sys.stderr)
        return None

    # Normalize variants: validate against the scope and fall back to the default if needed.
    valid_variants = list(PROMPT_VARIANTS[scope])
    variants = dedup([v for v in (variants or []) if v in valid_variants])
    if not variants:
        variants = [DEFAULT_VARIANT[scope]]

    # The other three sweep dimensions accept a scalar or a list.
    max_tokens_list = as_list(max_tokens, [4096])
    num_ctx_list = as_list(num_ctx, [8192])
    think_list = dedup([bool(t) for t in as_list(think, [True])])

    # Cartesian product of ALL dimensions (just like models/prompts/images).
    combos = [
        {"model": m, "variant": v, "max_tokens": mt, "num_ctx": nc, "think": th}
        for m in models
        for v in variants
        for mt in max_tokens_list
        for nc in num_ctx_list
        for th in think_list
    ]
    total_calls = len(images) * runs * len(combos)

    print("=" * 72)
    print(" BENCHMARK VLM")
    print("=" * 72)
    print(f"  Images    : {len(images)}  ->  {', '.join(os.path.basename(i) for i in images)}")
    print(f"  Models    : {len(models)}  ->  {', '.join(models)}")
    print(f"  Prompts   : {len(variants)}  ->  {', '.join(variants)}")
    print(f"  max_tokens: {', '.join(str(x) for x in max_tokens_list)}")
    print(f"  num_ctx   : {', '.join(str(x) for x in num_ctx_list)}")
    print(f"  think     : {', '.join('ON' if t else 'OFF' for t in think_list)}")
    print(f"  Runs/img  : {runs}     Mode: {scope}")
    # The `think` flag only applies to models with the 'thinking' capability. We
    # state the reality: qwen3-vl always reasons (ignores think=OFF on 0.30.6); qwen2.5vl never.
    thinkers = [m for m in models if model_supports_thinking(m, url)]
    nonthinkers = [m for m in models if m not in thinkers]
    if thinkers:
        print(f"  Reasoning : {', '.join(thinkers)}  (always; the think=OFF flag is ignored)")
    if nonthinkers:
        print(f"  No reasoning : {', '.join(nonthinkers)}  (no 'thinking' capability; truly OFF)")
    print(f"  Total calls: {total_calls}  ({len(combos)} combination(s) "
          f"model×prompt×max_tokens×num_ctx×think)")
    print("=" * 72 + "\n")

    encoded = {p: encode_image(p) for p in images}
    sizes = {p: image_size(p) for p in images}
    results = {}

    done = 0
    bench_t0 = time.perf_counter()

    for combo in combos:
        model, variant = combo["model"], combo["variant"]
        label = combo_label(combo)
        print(f"=== Model: {model}  |  Prompt: {variant}  |  "
              f"ctx={combo['num_ctx']} max_tokens={combo['max_tokens']} "
              f"think={'ON' if combo['think'] else 'OFF'} ===")
        latencies, valid, errors, total, trunc, objs = [], 0, 0, 0, 0, []
        thought = 0  # how many times the model actually reasoned
        per_image = {}  # img -> list of latencies
        for img in images:
            name = os.path.basename(img)
            per_image.setdefault(name, [])
            for r in range(1, runs + 1):
                total += 1
                done += 1
                # Bar BEFORE the call (so you see what's being processed).
                avg_so_far = statistics.mean(latencies) if latencies else 0.0
                eta = avg_so_far * (total_calls - done + 1)
                progress_bar(done - 1, total_calls,
                             suffix=f"{label} | {name} run {r}/{runs} | "
                                    f"avg {fmt_secs(avg_so_far)} | ETA {fmt_secs(eta)}")
                try:
                    res = query_vlm(encoded[img], model, scope=scope,
                                    max_tokens=combo["max_tokens"], think=combo["think"],
                                    url=url, num_ctx=combo["num_ctx"], size=sizes[img],
                                    variant=variant)
                    latencies.append(res["elapsed"])
                    per_image[name].append(res["elapsed"])
                    if res["ok"]:
                        valid += 1
                        if isinstance(res["parsed"], dict):
                            objs.append(len(res["parsed"].get("objects", [])))
                    if res["finish_reason"] == "length":
                        trunc += 1
                    if res.get("did_think"):
                        thought += 1
                except requests.RequestException as e:
                    errors += 1
                    progress_bar(done, total_calls, suffix=f"ERROR on {name}")
                    print(f"\n  [!] {name}: ERROR {e}")
                    continue
            # Refresh the bar when each image finishes.
            avg_so_far = statistics.mean(latencies) if latencies else 0.0
            eta = avg_so_far * (total_calls - done)
            progress_bar(done, total_calls,
                         suffix=f"{label} | done {name} | avg {fmt_secs(avg_so_far)} | "
                                f"ETA {fmt_secs(eta)}")
        progress_bar(done, total_calls, suffix=f"{label} complete")

        # Per-image table for this combination (average time per image).
        print(f"\n  Time per image ({label}):")
        for name, lats in per_image.items():
            if lats:
                print(f"    {name:28s} avg {fmt_secs(statistics.mean(lats)):>7s}  "
                      f"(min {fmt_secs(min(lats))} / max {fmt_secs(max(lats))})")
            else:
                print(f"    {name:28s} no data (errors)")

        stats = _stats(latencies, valid, trunc, objs, errors, total)
        stats["model"] = model
        stats["variant"] = variant
        stats["max_tokens"] = combo["max_tokens"]
        stats["num_ctx"] = combo["num_ctx"]
        stats["think"] = combo["think"]
        stats["thinking_supported"] = model_supports_thinking(model, url)
        stats["thought"] = thought  # how many runs actually reasoned
        results[label] = stats
        print()

    bench_wall = time.perf_counter() - bench_t0

    # ---------------- Comparison summary + times ----------------
    header = (f"{'Model':16s} {'Prompt':13s} {'ctx':>6s} {'maxtok':>6s} {'thk':>3s} "
              f"{'P50':>7s} {'P95':>7s} {'Mean':>7s} {'Min':>7s} {'Max':>7s} "
              f"{'Total':>8s} {'JSON%':>7s} {'Trunc':>6s} {'~obj':>5s} {'Errs':>5s}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results.values():
        print(f"{r['model']:16s} {r['variant']:13s} {r['num_ctx']:>6d} {r['max_tokens']:>6d} "
              f"{('ON' if r['think'] else 'OFF'):>3s} {fmt_secs(r['p50']):>7s} "
              f"{fmt_secs(r['p95']):>7s} {fmt_secs(r['mean']):>7s} {fmt_secs(r['min']):>7s} "
              f"{fmt_secs(r['max']):>7s} {fmt_secs(r['total_time']):>8s} {r['json_rate']:6.1f}% "
              f"{r['truncated']:6d} {r['avg_objects']:5.1f} {r['errors']:5d}")
    print("=" * len(header))
    print(f"Total benchmark time (wall clock): {fmt_secs(bench_wall)}  "
          f"|  {total_calls} calls")
    print("Doc target (F1.8): P95 < 1.5 s on-prem, high JSON rate for VLM->VLA.")

    # Verdict: best combination (fastest among those producing reliable JSON).
    if len(results) > 1:
        usable = {k: r for k, r in results.items()
                  if r["json_rate"] >= 50 and r["mean"] == r["mean"]}
        pool = usable or {k: r for k, r in results.items() if r["mean"] == r["mean"]}
        if pool:
            best = min(pool, key=lambda k: pool[k]["mean"])
            tag = "fastest with reliable JSON" if usable else "fastest (watch the JSON%!)"
            r = results[best]
            print(f"-> Best combination: {best} ({tag}).")
            print(f"   To pin it in config.json: \"model\"={r['model']!r}, "
                  f"\"variant\"={r['variant']!r}, \"max_tokens\"={r['max_tokens']}, "
                  f"\"num_ctx\"={r['num_ctx']}, \"think\"={str(r['think']).lower()}.")

    payload = {"config": {"runs": runs, "scope": scope, "variants": variants,
                          "models": models, "max_tokens": max_tokens_list,
                          "num_ctx": num_ctx_list, "think": think_list,
                          "images": [os.path.basename(i) for i in images],
                          "total_wall_s": round(bench_wall, 2)},
               "results": results}
    out = out or results_path("benchmark_resultados.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out}")
    return results


def main():
    cfg = load_config()  # defaults come from config.json (benchmark_* keys)
    ap = argparse.ArgumentParser(description="VLM latency/JSON benchmark and prompt A/B testing.")
    ap.add_argument("folder", nargs="?", default=cfg["folder"],
                    help="Folder with images. Default: the one in config.json")
    ap.add_argument("--images", nargs="+", default=None,
                    help="Names of specific images inside the folder "
                         "(e.g. 1.jpeg 14.jpeg). Default: ALL.")
    ap.add_argument("--models", nargs="+", default=cfg["benchmark_models"],
                    help="Models to compare")
    ap.add_argument("--variants", nargs="+",
                    default=cfg.get("benchmark_variants") or [cfg.get("benchmark_variant")],
                    help="Prompt variants to compare (e.g. v1_original v2_antiloop). "
                         "Default: those in config.json. See keys in src/vlm_common.py.")
    ap.add_argument("--runs", type=int, default=cfg["benchmark_runs"],
                    help="Repetitions per image")
    ap.add_argument("--scope", choices=list(SCOPES),
                    default=cfg.get("benchmark_scope", cfg["scope"]),
                    help="Detection mode: industrial | all")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", nargs="+", type=int,
                    default=cfg.get("benchmark_max_tokens", [4096]),
                    help="One or SEVERAL OUTPUT token caps / num_predict to compare "
                         "(e.g. --max-tokens 4096 8192)")
    ap.add_argument("--num-ctx", nargs="+", type=int,
                    default=cfg.get("benchmark_num_ctx", [8192]),
                    help="One or SEVERAL context windows to compare (e.g. --num-ctx 8192 16384)")
    ap.add_argument("--think", nargs="+", default=cfg.get("benchmark_think", [True]),
                    help="One or SEVERAL reasoning values to compare: true/false "
                         "(e.g. --think true false). Only applies to models with the 'thinking' capability.")
    args = ap.parse_args()

    # Validate variants against the chosen scope.
    valid_variants = list(PROMPT_VARIANTS[args.scope])
    unknown = [v for v in (args.variants or []) if v and v not in valid_variants]
    if unknown:
        print(f"[ERROR] Unknown variant(s) for scope '{args.scope}': {unknown}",
              file=sys.stderr)
        print(f"        Available: {valid_variants}", file=sys.stderr)
        sys.exit(1)

    # Parse the think values (true/false/...). None = not understood.
    think_vals = [parse_bool(t) for t in (args.think if isinstance(args.think, list) else [args.think])]
    if any(t is None for t in think_vals):
        print(f"[ERROR] Invalid --think value(s): {args.think}. Use true/false.",
              file=sys.stderr)
        sys.exit(1)

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

    res = run_benchmark(images, args.models, runs=args.runs, scope=args.scope,
                        max_tokens=args.max_tokens, think=think_vals, url=args.url,
                        num_ctx=args.num_ctx, variants=args.variants)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
