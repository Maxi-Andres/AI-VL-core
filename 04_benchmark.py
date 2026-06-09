#!/usr/bin/env python3
"""
04_benchmark.py — Benchmark de latencia (P50/P95) y tasa de JSON válido.

Corre el mismo prompt sobre todas las imágenes de una carpeta, N veces por
imagen, contra uno o varios modelos. Sirve para comparar qwen3-vl:8b vs
qwen3-vl:4b vs qwen2.5vl:7b con datos reales del lab y decidir el VLM primario
del PoC (criterios F1.8: precisión, latencia P95, tasa de JSON válido VLM->VLA).

¿No querés escribir flags? Corré `python3 menu.py` (menú interactivo).

Requisitos:  pip install requests
Uso:
    python3 04_benchmark.py fotosClean --runs 3
    python3 04_benchmark.py fotosClean --models qwen3-vl:8b qwen3-vl:4b
    python3 04_benchmark.py fotosClean --scope todo
"""
import argparse
import glob
import json
import os
import statistics
import sys

import requests

from vlm_common import (
    IMG_EXTS,
    OLLAMA_HOST,
    SCOPES,
    encode_image,
    image_size,
    load_config,
    query_vlm,
)


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_benchmark(folder, models, runs=3, scope="industrial", max_tokens=8192,
                  think=True, url=OLLAMA_HOST, num_ctx=16384, variant=None,
                  out="benchmark_resultados.json"):
    """Corre el benchmark e imprime la tabla comparativa. Devuelve el dict de resultados."""
    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(os.path.join(folder, ext)))
    if not images:
        print(f"[ERROR] No hay imágenes en {folder}", file=sys.stderr)
        return None
    print(f"[..] {len(images)} imágenes, {runs} runs c/u, {len(models)} modelo(s), "
          f"modo: {scope}.\n")

    encoded = {p: encode_image(p) for p in images}
    sizes = {p: image_size(p) for p in images}
    results = {}

    for model in models:
        print(f"=== Modelo: {model} ===")
        latencies, valid, errors, total = [], 0, 0, 0
        for img in images:
            for _ in range(runs):
                total += 1
                try:
                    res = query_vlm(encoded[img], model, scope=scope,
                                    max_tokens=max_tokens, think=think, url=url,
                                    num_ctx=num_ctx, size=sizes[img], variant=variant)
                    latencies.append(res["elapsed"])
                    valid += 1 if res["ok"] else 0
                    flag = "json-ok" if res["ok"] else "json-FALLA"
                    print(f"  {os.path.basename(img):30s} {res['elapsed']:6.2f}s  {flag}")
                except requests.RequestException as e:
                    errors += 1
                    print(f"  {os.path.basename(img):30s}  ERROR: {e}")
        results[model] = {
            "p50": pctl(latencies, 50),
            "p95": pctl(latencies, 95),
            "mean": statistics.mean(latencies) if latencies else float("nan"),
            "json_rate": (valid / total * 100) if total else 0.0,
            "errors": errors,
            "n": total,
        }
        print()

    # Resumen comparativo
    print("=" * 72)
    print(f"{'Modelo':20s} {'P50(s)':>8s} {'P95(s)':>8s} {'Media(s)':>9s} "
          f"{'JSON%':>7s} {'Errs':>6s}")
    print("-" * 72)
    for model, r in results.items():
        print(f"{model:20s} {r['p50']:8.2f} {r['p95']:8.2f} {r['mean']:9.2f} "
              f"{r['json_rate']:6.1f}% {r['errors']:6d}")
    print("=" * 72)
    print("Objetivo del doc (F1.8): P95 < 1.5 s on-prem, tasa JSON alta para VLM->VLA.")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Guardado en {out}")
    return results


def main():
    cfg = load_config()  # los defaults salen de config.json
    ap = argparse.ArgumentParser(description="Benchmark de latencia y JSON del VLM.")
    ap.add_argument("folder", nargs="?", default=cfg["folder"],
                    help="Carpeta con imágenes. Default: el de config.json")
    ap.add_argument("--models", nargs="+", default=cfg["benchmark_models"],
                    help="Modelos a comparar")
    ap.add_argument("--runs", type=int, default=cfg["benchmark_runs"],
                    help="Repeticiones por imagen")
    ap.add_argument("--scope", choices=list(SCOPES), default=cfg["scope"],
                    help="Modo de detección: industrial | todo")
    ap.add_argument("--variant", default=cfg.get("variant"),
                    help="Variante de prompt (ej. v1_original, v2_antiloop). Default: config.json")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", type=int, default=cfg["max_tokens"],
                    help="Tope de tokens de SALIDA / num_predict (incluye razonamiento)")
    ap.add_argument("--num-ctx", type=int, default=cfg.get("num_ctx", 16384),
                    help="Ventana de contexto (entrada+salida); la que ves en `ollama ps`")
    ap.add_argument("--think", dest="think", action="store_true", default=cfg["think"],
                    help="Razonamiento del modelo (default ON)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="Pedir think=false")
    args = ap.parse_args()

    res = run_benchmark(args.folder, args.models, runs=args.runs, scope=args.scope,
                        max_tokens=args.max_tokens, think=args.think, url=args.url,
                        num_ctx=args.num_ctx, variant=args.variant)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
