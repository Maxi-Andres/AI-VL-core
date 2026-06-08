#!/usr/bin/env python3
"""
04_benchmark.py — Benchmark de latencia (P50/P95) y tasa de JSON válido.

Corre el mismo prompt industrial sobre todas las imágenes de una carpeta,
N veces por imagen, contra uno o varios modelos. Sirve para comparar
qwen3-vl:8b vs qwen3-vl:4b vs qwen2.5vl:7b con datos reales del lab y
decidir el VLM primario del PoC (criterios F1.8: precisión, latencia P95,
tasa de JSON válido para el contrato VLM->VLA).

Requisitos:  pip install requests
Uso:
    python3 04_benchmark.py ./imagenes_lab --runs 3
    python3 04_benchmark.py ./imagenes_lab --models qwen3-vl:8b qwen3-vl:4b
"""
import argparse
import base64
import glob
import json
import os
import re
import statistics
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

SYSTEM_PROMPT = (
    "Sos un asistente de inspección industrial. Respondés SIEMPRE en JSON válido, "
    "sin texto extra ni markdown."
)
USER_PROMPT = (
    "Identificá manómetros, válvulas y EPP en la imagen. Devolvé "
    '{"objetos":[{"tipo":..., "bbox":[x_min,y_min,x_max,y_max], '
    '"lectura":..., "confianza":...}]} con coordenadas normalizadas 0-1.'
)

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_valid_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                json.loads(m.group(0))
                return True
            except json.JSONDecodeError:
                return False
    return False


def query(model, img_b64, url):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 768,
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    content = r.json()["choices"][0]["message"]["content"]
    return elapsed, is_valid_json(content)


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Carpeta con imágenes del lab")
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"])
    ap.add_argument("--runs", type=int, default=3, help="Repeticiones por imagen")
    ap.add_argument("--url", default=OLLAMA_URL)
    args = ap.parse_args()

    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(os.path.join(args.folder, ext)))
    if not images:
        print(f"[ERROR] No hay imágenes en {args.folder}", file=sys.stderr)
        sys.exit(1)
    print(f"[..] {len(images)} imágenes, {args.runs} runs c/u, "
          f"{len(args.models)} modelo(s).\n")

    encoded = {p: encode_image(p) for p in images}
    results = {}

    for model in args.models:
        print(f"=== Modelo: {model} ===")
        latencies, valid, errors = [], 0, 0
        total = 0
        for img in images:
            for _ in range(args.runs):
                total += 1
                try:
                    lat, ok = query(model, encoded[img], args.url)
                    latencies.append(lat)
                    valid += 1 if ok else 0
                    flag = "json-ok" if ok else "json-FALLA"
                    print(f"  {os.path.basename(img):30s} {lat:6.2f}s  {flag}")
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

    with open("benchmark_resultados.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] Guardado en benchmark_resultados.json")


if __name__ == "__main__":
    main()
