#!/usr/bin/env python3
"""
benchmark.py — Benchmark de latencia (P50/P95), tasa de JSON válido y A/B de prompts.

Corre el MISMO conjunto de imágenes, N veces por imagen, barriendo el producto
MODELOS × VARIANTES DE PROMPT. Sirve para dos cosas a la vez:

  1) Comparar MODELOS (qwen3-vl:8b vs :4b vs qwen2.5vl:7b) con datos del lab y
     decidir el VLM primario del PoC (criterios F1.8: precisión, latencia P95,
     tasa de JSON válido VLM->VLA).
  2) Comparar VARIANTES DE PROMPT (v1_original vs v2_antiloop, etc.) para elegir
     cuál dejar activa. Esto reemplaza al viejo 05_prompt_test.py: ahora se
     prueban distintas prompts igual que se prueban distintos modelos, todo junto.

Lo que muestra mientras corre:
  - una BARRA DE PROGRESO en vivo (qué modelo/prompt/imagen va, % completado, ETA);
Y al terminar:
  - tiempo por imagen, tiempo total, tiempo promedio por llamada, P50/P95;
  - tasa de JSON válido (el contrato VLM->VLA), cortes por longitud y objetos promedio;
  - un veredicto con la mejor combinación cuando comparás más de una.

Los resultados se guardan en results/ (no sueltos en la raíz del proyecto).

¿No querés escribir flags? Corré `python3 menu.py` -> opción 2 (Benchmark),
que abre un submenú para elegir imágenes, modelos, prompts, runs y contexto.

Requisitos:  pip install requests
Uso:
    python3 src/benchmark.py fotos/clean --runs 3
    python3 src/benchmark.py fotos/clean --models qwen3-vl:8b qwen3-vl:4b
    python3 src/benchmark.py fotos/clean --variants v1_original v2_antiloop
    python3 src/benchmark.py fotos/clean --images 1.jpeg 14.jpeg 16.jpeg
    python3 src/benchmark.py fotos/clean --scope todo
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


def _stats(latencies, valid, trunc, objs, errors, total):
    """Arma el dict de métricas de una combinación (modelo × variante)."""
    return {
        "p50": pctl(latencies, 50),
        "p95": pctl(latencies, 95),
        "mean": statistics.mean(latencies) if latencies else float("nan"),
        "min": min(latencies) if latencies else float("nan"),
        "max": max(latencies) if latencies else float("nan"),
        "total_time": sum(latencies),
        "json_rate": (valid / total * 100) if total else 0.0,
        "truncadas": trunc,
        "avg_objetos": statistics.mean(objs) if objs else 0.0,
        "errors": errors,
        "n": total,
    }


def run_benchmark(images, models, runs=3, scope="industrial", max_tokens=4096,
                  think=True, url=OLLAMA_HOST, num_ctx=8192, variants=None,
                  out=None):
    """Corre el benchmark sobre una LISTA de imágenes barriendo modelos × variantes.

    `images`   : lista de rutas concretas (ya elegidas). Si tenés una carpeta,
                 expandila antes con vlm_common.list_images().
    `models`   : lista de modelos de Ollama a comparar.
    `variants` : lista de variantes de prompt a comparar (claves de PROMPT_VARIANTS
                 para el scope). Vacío/None -> la variante por defecto del scope.

    Imprime tiempos + JSON% + cortes + objetos por cada combinación y un veredicto
    final. Devuelve el dict de resultados (también lo guarda en `out`, dentro de
    results/ si no se pasa una ruta).
    """
    images = [p for p in images if os.path.exists(p)]
    if not images:
        print("[ERROR] No hay imágenes válidas para correr.", file=sys.stderr)
        return None
    if not models:
        print("[ERROR] No hay modelos para correr.", file=sys.stderr)
        return None

    # Normalizar variantes: validar contra el scope y caer al default si hace falta.
    valid_variants = list(PROMPT_VARIANTS[scope])
    variants = [v for v in (variants or []) if v in valid_variants]
    if not variants:
        variants = [DEFAULT_VARIANT[scope]]

    combos = [(m, v) for m in models for v in variants]
    total_calls = len(images) * runs * len(combos)

    print("=" * 72)
    print(" BENCHMARK VLM")
    print("=" * 72)
    print(f"  Imágenes : {len(images)}  ->  {', '.join(os.path.basename(i) for i in images)}")
    print(f"  Modelos  : {len(models)}  ->  {', '.join(models)}")
    print(f"  Prompts  : {len(variants)}  ->  {', '.join(variants)}")
    print(f"  Runs/img : {runs}     Modo: {scope}")
    print(f"  Contexto : num_ctx={num_ctx}   max_tokens(salida)={max_tokens}   think={think}")
    print(f"  Total de llamadas: {total_calls}  ({len(combos)} combinación/es modelo×prompt)")
    print("=" * 72 + "\n")

    encoded = {p: encode_image(p) for p in images}
    sizes = {p: image_size(p) for p in images}
    results = {}

    done = 0
    bench_t0 = time.perf_counter()

    for model, variant in combos:
        combo_label = f"{model} [{variant}]"
        print(f"=== Modelo: {model}  |  Prompt: {variant} ===")
        latencies, valid, errors, total, trunc, objs = [], 0, 0, 0, 0, []
        per_image = {}  # img -> lista de latencias
        for img in images:
            name = os.path.basename(img)
            per_image.setdefault(name, [])
            for r in range(1, runs + 1):
                total += 1
                done += 1
                # Barra ANTES de la llamada (así se ve qué está procesando).
                avg_so_far = statistics.mean(latencies) if latencies else 0.0
                eta = avg_so_far * (total_calls - done + 1)
                progress_bar(done - 1, total_calls,
                             suffix=f"{combo_label} | {name} run {r}/{runs} | "
                                    f"prom {fmt_secs(avg_so_far)} | ETA {fmt_secs(eta)}")
                try:
                    res = query_vlm(encoded[img], model, scope=scope,
                                    max_tokens=max_tokens, think=think, url=url,
                                    num_ctx=num_ctx, size=sizes[img], variant=variant)
                    latencies.append(res["elapsed"])
                    per_image[name].append(res["elapsed"])
                    if res["ok"]:
                        valid += 1
                        if isinstance(res["parsed"], dict):
                            objs.append(len(res["parsed"].get("objetos", [])))
                    if res["finish_reason"] == "length":
                        trunc += 1
                except requests.RequestException as e:
                    errors += 1
                    progress_bar(done, total_calls, suffix=f"ERROR en {name}")
                    print(f"\n  [!] {name}: ERROR {e}")
                    continue
            # Refresca la barra al cerrar cada imagen.
            avg_so_far = statistics.mean(latencies) if latencies else 0.0
            eta = avg_so_far * (total_calls - done)
            progress_bar(done, total_calls,
                         suffix=f"{combo_label} | hecho {name} | prom {fmt_secs(avg_so_far)} | "
                                f"ETA {fmt_secs(eta)}")
        progress_bar(done, total_calls, suffix=f"{combo_label} completo")

        # Tabla por imagen de esta combinación (tiempo medio por imagen).
        print(f"\n  Tiempo por imagen ({combo_label}):")
        for name, lats in per_image.items():
            if lats:
                print(f"    {name:28s} prom {fmt_secs(statistics.mean(lats)):>7s}  "
                      f"(min {fmt_secs(min(lats))} / max {fmt_secs(max(lats))})")
            else:
                print(f"    {name:28s} sin datos (errores)")

        stats = _stats(latencies, valid, trunc, objs, errors, total)
        stats["model"] = model
        stats["variant"] = variant
        results[combo_label] = stats
        print()

    bench_wall = time.perf_counter() - bench_t0

    # ---------------- Resumen comparativo + tiempos ----------------
    print("=" * 102)
    print(f"{'Modelo':16s} {'Prompt':14s} {'P50':>7s} {'P95':>7s} {'Media':>7s} "
          f"{'Min':>7s} {'Max':>7s} {'Total':>8s} {'JSON%':>7s} {'Cortes':>7s} "
          f"{'~obj':>6s} {'Errs':>5s}")
    print("-" * 102)
    for r in results.values():
        print(f"{r['model']:16s} {r['variant']:14s} {fmt_secs(r['p50']):>7s} "
              f"{fmt_secs(r['p95']):>7s} {fmt_secs(r['mean']):>7s} {fmt_secs(r['min']):>7s} "
              f"{fmt_secs(r['max']):>7s} {fmt_secs(r['total_time']):>8s} {r['json_rate']:6.1f}% "
              f"{r['truncadas']:7d} {r['avg_objetos']:6.1f} {r['errors']:5d}")
    print("=" * 102)
    print(f"Tiempo total del benchmark (reloj): {fmt_secs(bench_wall)}  "
          f"|  {total_calls} llamadas")
    print("Objetivo del doc (F1.8): P95 < 1.5 s on-prem, tasa JSON alta para VLM->VLA.")

    # Veredicto: mejor combinación (más rápida entre las que dan JSON confiable).
    if len(results) > 1:
        usable = {k: r for k, r in results.items()
                  if r["json_rate"] >= 50 and r["mean"] == r["mean"]}
        pool = usable or {k: r for k, r in results.items() if r["mean"] == r["mean"]}
        if pool:
            best = min(pool, key=lambda k: pool[k]["mean"])
            tag = "más rápida con JSON confiable" if usable else "más rápida (¡ojo con el JSON%!)"
            r = results[best]
            print(f"-> Mejor combinación: {best} ({tag}).")
            print(f"   Para dejar esa prompt activa: poné \"variant\": \"{r['variant']}\" "
                  f"en config.json (o cambiá DEFAULT_VARIANT en src/vlm_common.py).")

    payload = {"config": {"runs": runs, "scope": scope, "variants": variants,
                          "models": models, "max_tokens": max_tokens,
                          "num_ctx": num_ctx, "think": think,
                          "images": [os.path.basename(i) for i in images],
                          "total_wall_s": round(bench_wall, 2)},
               "results": results}
    out = out or results_path("benchmark_resultados.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Guardado en {out}")
    return results


def main():
    cfg = load_config()  # los defaults salen de config.json (claves benchmark_*)
    ap = argparse.ArgumentParser(description="Benchmark de latencia/JSON y A/B de prompts del VLM.")
    ap.add_argument("folder", nargs="?", default=cfg["folder"],
                    help="Carpeta con imágenes. Default: el de config.json")
    ap.add_argument("--images", nargs="+", default=None,
                    help="Nombres de imágenes concretas dentro de la carpeta "
                         "(ej. 1.jpeg 14.jpeg). Default: TODAS.")
    ap.add_argument("--models", nargs="+", default=cfg["benchmark_models"],
                    help="Modelos a comparar")
    ap.add_argument("--variants", nargs="+",
                    default=cfg.get("benchmark_variants") or [cfg.get("benchmark_variant")],
                    help="Variantes de prompt a comparar (ej. v1_original v2_antiloop). "
                         "Default: las de config.json. Ver claves en src/vlm_common.py.")
    ap.add_argument("--runs", type=int, default=cfg["benchmark_runs"],
                    help="Repeticiones por imagen")
    ap.add_argument("--scope", choices=list(SCOPES),
                    default=cfg.get("benchmark_scope", cfg["scope"]),
                    help="Modo de detección: industrial | todo")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", type=int,
                    default=cfg.get("benchmark_max_tokens", 4096),
                    help="Tope de tokens de SALIDA / num_predict (incluye razonamiento)")
    ap.add_argument("--num-ctx", type=int,
                    default=cfg.get("benchmark_num_ctx", 8192),
                    help="Ventana de contexto (entrada+salida); la que ves en `ollama ps`")
    ap.add_argument("--think", dest="think", action="store_true",
                    default=cfg.get("benchmark_think", cfg["think"]),
                    help="Razonamiento del modelo (default ON)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="Pedir think=false")
    args = ap.parse_args()

    # Validar variantes contra el scope elegido.
    valid_variants = list(PROMPT_VARIANTS[args.scope])
    unknown = [v for v in (args.variants or []) if v and v not in valid_variants]
    if unknown:
        print(f"[ERROR] Variante(s) desconocida(s) para scope '{args.scope}': {unknown}",
              file=sys.stderr)
        print(f"        Disponibles: {valid_variants}", file=sys.stderr)
        sys.exit(1)

    all_imgs = list_images(args.folder)
    if args.images:
        wanted = set(args.images)
        images = [p for p in all_imgs if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in images}
        if missing:
            print(f"[!] No encontré en {args.folder}: {', '.join(sorted(missing))}",
                  file=sys.stderr)
    else:
        images = all_imgs

    res = run_benchmark(images, args.models, runs=args.runs, scope=args.scope,
                        max_tokens=args.max_tokens, think=args.think, url=args.url,
                        num_ctx=args.num_ctx, variants=args.variants)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
