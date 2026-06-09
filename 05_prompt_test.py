#!/usr/bin/env python3
"""
05_prompt_test.py — Comparador A/B de VARIANTES DE PROMPT.

Corre la MISMA imagen (o carpeta) con distintas variantes de prompt y mide cuál
es más rápida y más confiable. Sirve para decidir qué prompt dejar activo
(DEFAULT_VARIANT en vlm_common.py / clave "variant" en config.json).

Todo lo demás se mantiene constante entre variantes (modelo, max_tokens,
num_ctx, think) para que la comparación sea justa: la única variable es el prompt.

¿No querés escribir flags? Corré `python3 menu.py` (opción de comparar prompts).

Requisitos:  pip install requests
Uso:
    python3 05_prompt_test.py                       # imagen y scope de config.json, todas las variantes
    python3 05_prompt_test.py fotosClean/16.jpeg --runs 3
    python3 05_prompt_test.py fotosClean/16.jpeg --variants v1_original v2_antiloop
    python3 05_prompt_test.py --folder fotosClean --runs 1     # promedia sobre toda la carpeta
    python3 05_prompt_test.py --list                # lista las variantes disponibles
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
    PROMPT_VARIANTS,
    SCOPES,
    encode_image,
    image_size,
    load_config,
    query_vlm,
)


def list_variants():
    print("Variantes de prompt disponibles (por scope):\n")
    for scope, variants in PROMPT_VARIANTS.items():
        print(f"  {scope}:")
        for name in variants:
            print(f"    - {name}")
    print("\nProbá: python3 05_prompt_test.py <imagen> --variants <a> <b>")


def collect_images(image, folder):
    """Devuelve la lista de imágenes a usar: la carpeta si se pidió, si no la imagen."""
    if folder:
        imgs = []
        for ext in IMG_EXTS:
            imgs.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(imgs)
    return [image]


def run_prompt_test(images, scope, variants, model, runs=3, max_tokens=8192,
                    think=True, url=OLLAMA_HOST, num_ctx=16384,
                    out="prompt_test_resultados.json"):
    """Compara variantes de prompt sobre las imágenes dadas. Devuelve el dict de resultados."""
    # Pre-encode una sola vez (no es parte de lo que medimos).
    encoded = {p: encode_image(p) for p in images}
    sizes = {p: image_size(p) for p in images}

    print(f"[..] {len(images)} imagen(es) x {runs} run(s) x {len(variants)} variante(s) "
          f"| scope: {scope} | modelo: {model}")
    print(f"     (constantes: max_tokens={max_tokens}, num_ctx={num_ctx}, think={think})\n")

    results = {}
    for variant in variants:
        print(f"=== Prompt: {variant} ===")
        lat, valid, trunc, objs, out_toks, total = [], 0, 0, [], [], 0
        for img in images:
            for _ in range(runs):
                total += 1
                try:
                    res = query_vlm(encoded[img], model, scope=scope,
                                    max_tokens=max_tokens, think=think, url=url,
                                    num_ctx=num_ctx, size=sizes[img], variant=variant)
                except requests.RequestException as e:
                    print(f"  {os.path.basename(img):28s}  ERROR: {e}")
                    continue
                lat.append(res["elapsed"])
                if res["out_tokens"] is not None:
                    out_toks.append(res["out_tokens"])
                if res["ok"]:
                    valid += 1
                    n = len(res["parsed"].get("objetos", [])) if isinstance(res["parsed"], dict) else 0
                    objs.append(n)
                    flag = f"json-ok ({n} obj)"
                else:
                    flag = "json-FALLA"
                if res["finish_reason"] == "length":
                    trunc += 1
                    flag += " [CORTADO x length]"
                print(f"  {os.path.basename(img):28s} {res['elapsed']:6.2f}s  {flag}")
        results[variant] = {
            "mean_s": statistics.mean(lat) if lat else float("nan"),
            "p50_s": statistics.median(lat) if lat else float("nan"),
            "json_rate": (valid / total * 100) if total else 0.0,
            "truncadas": trunc,
            "avg_objetos": (statistics.mean(objs) if objs else 0.0),
            "avg_out_tokens": (statistics.mean(out_toks) if out_toks else 0.0),
            "n": total,
        }
        print()

    # Tabla comparativa
    print("=" * 84)
    print(f"{'Prompt':16s} {'Media(s)':>9s} {'P50(s)':>8s} {'JSON%':>7s} "
          f"{'Cortes':>7s} {'~obj':>6s} {'~tok_out':>9s}")
    print("-" * 84)
    for v, r in results.items():
        print(f"{v:16s} {r['mean_s']:9.2f} {r['p50_s']:8.2f} {r['json_rate']:6.1f}% "
              f"{r['truncadas']:7d} {r['avg_objetos']:6.1f} {r['avg_out_tokens']:9.0f}")
    print("=" * 84)

    # Veredicto rápido: más rápida entre las que tienen buena tasa de JSON.
    usable = {v: r for v, r in results.items() if r["json_rate"] >= 50 and r["mean_s"] == r["mean_s"]}
    pool = usable or {v: r for v, r in results.items() if r["mean_s"] == r["mean_s"]}
    if pool:
        best = min(pool, key=lambda v: pool[v]["mean_s"])
        tag = "más rápida con JSON confiable" if usable else "más rápida (¡ojo con el JSON%!)"
        print(f"-> Mejor candidata: '{best}' ({tag}).")
        print(f"   Para dejarla activa: poné \"variant\": \"{best}\" en config.json "
              f"(o cambiá DEFAULT_VARIANT en vlm_common.py).")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Guardado en {out}")
    return results


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Comparador A/B de variantes de prompt.")
    ap.add_argument("image", nargs="?", default=cfg["image"],
                    help="Imagen a usar. Default: la de config.json")
    ap.add_argument("--folder", default=None,
                    help="En vez de una imagen, promediar sobre todas las de esta carpeta.")
    ap.add_argument("--scope", choices=list(SCOPES), default=cfg["scope"],
                    help="Modo de detección: industrial | todo")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Variantes a comparar. Default: TODAS las del scope.")
    ap.add_argument("--model", default=cfg["model"])
    ap.add_argument("--runs", type=int, default=2,
                    help="Repeticiones por imagen y variante (default 2).")
    ap.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    ap.add_argument("--num-ctx", type=int, default=cfg.get("num_ctx", 16384))
    ap.add_argument("--think", dest="think", action="store_true", default=cfg["think"])
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--list", action="store_true", help="Lista las variantes y sale.")
    args = ap.parse_args()

    if args.list:
        list_variants()
        return

    variants = args.variants or list(PROMPT_VARIANTS[args.scope])
    unknown = [v for v in variants if v not in PROMPT_VARIANTS[args.scope]]
    if unknown:
        print(f"[ERROR] Variante(s) desconocida(s) para scope '{args.scope}': {unknown}",
              file=sys.stderr)
        print(f"        Disponibles: {list(PROMPT_VARIANTS[args.scope])}", file=sys.stderr)
        sys.exit(1)

    images = collect_images(args.image, args.folder)
    if not images or (not args.folder and not os.path.exists(args.image)):
        print(f"[ERROR] No encuentro imágenes ({args.folder or args.image}).", file=sys.stderr)
        sys.exit(1)

    run_prompt_test(images, args.scope, variants, args.model, runs=args.runs,
                    max_tokens=args.max_tokens, think=args.think, url=args.url,
                    num_ctx=args.num_ctx)


if __name__ == "__main__":
    main()
