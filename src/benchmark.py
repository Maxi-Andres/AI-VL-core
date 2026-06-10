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
    """Quita duplicados preservando el orden de aparición."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def as_list(v, fallback):
    """Normaliza un valor (escalar o lista) a una lista no vacía y sin duplicados.

    Sirve para que las dimensiones del barrido (max_tokens, num_ctx, think) acepten
    tanto un solo valor como una lista. Si queda vacía, usa `fallback` (lista).
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
    """Convierte 'true/false/on/off/1/0/yes/no/si' (o un bool) a bool. None si no entiende."""
    if isinstance(s, bool):
        return s
    t = str(s).strip().lower()
    if t in ("true", "on", "1", "yes", "si", "sí", "y"):
        return True
    if t in ("false", "off", "0", "no", "n"):
        return False
    return None


def combo_label(c):
    """Etiqueta compacta y única de una combinación del barrido."""
    return (f"{c['model']} [{c['variant']}] "
            f"ctx{c['num_ctx']} max{c['max_tokens']} think{'1' if c['think'] else '0'}")


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
    """Corre el benchmark barriendo el producto modelos × prompts × max_tokens × num_ctx × think.

    `images`     : lista de rutas concretas (ya elegidas). Si tenés una carpeta,
                   expandila antes con vlm_common.list_images().
    `models`     : lista de modelos de Ollama a comparar.
    `variants`   : lista de variantes de prompt (claves de PROMPT_VARIANTS para el
                   scope). Vacío/None -> la variante por defecto del scope.
    `max_tokens` : escalar o LISTA de topes de salida a comparar.
    `num_ctx`    : escalar o LISTA de ventanas de contexto a comparar.
    `think`      : escalar o LISTA de bools a comparar (sirve cuando consigas un
                   modelo que SÍ respete el flag; ver model_supports_thinking()).

    Cada combinación es una fila del reporte. Imprime tiempos + JSON% + cortes +
    objetos por combinación y un veredicto final. Devuelve el dict de resultados
    (también lo guarda en `out`, dentro de results/ si no se pasa una ruta).
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
    variants = dedup([v for v in (variants or []) if v in valid_variants])
    if not variants:
        variants = [DEFAULT_VARIANT[scope]]

    # Las otras tres dimensiones del barrido aceptan escalar o lista.
    max_tokens_list = as_list(max_tokens, [4096])
    num_ctx_list = as_list(num_ctx, [8192])
    think_list = dedup([bool(t) for t in as_list(think, [True])])

    # Producto cartesiano de TODAS las dimensiones (igual que modelos/prompts/fotos).
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
    print(f"  Imágenes  : {len(images)}  ->  {', '.join(os.path.basename(i) for i in images)}")
    print(f"  Modelos   : {len(models)}  ->  {', '.join(models)}")
    print(f"  Prompts   : {len(variants)}  ->  {', '.join(variants)}")
    print(f"  max_tokens: {', '.join(str(x) for x in max_tokens_list)}")
    print(f"  num_ctx   : {', '.join(str(x) for x in num_ctx_list)}")
    print(f"  think     : {', '.join('ON' if t else 'OFF' for t in think_list)}")
    print(f"  Runs/img  : {runs}     Modo: {scope}")
    # El flag `think` solo aplica a modelos con capability 'thinking'. Avisamos la
    # realidad: qwen3-vl siempre razona (ignora think=OFF en 0.30.6); qwen2.5vl nunca.
    thinkers = [m for m in models if model_supports_thinking(m, url)]
    nonthinkers = [m for m in models if m not in thinkers]
    if thinkers:
        print(f"  Razonan   : {', '.join(thinkers)}  (siempre; el flag think=OFF se ignora)")
    if nonthinkers:
        print(f"  No razonan : {', '.join(nonthinkers)}  (sin capability 'thinking'; OFF real)")
    print(f"  Total de llamadas: {total_calls}  ({len(combos)} combinación/es "
          f"modelo×prompt×max_tokens×num_ctx×think)")
    print("=" * 72 + "\n")

    encoded = {p: encode_image(p) for p in images}
    sizes = {p: image_size(p) for p in images}
    results = {}

    done = 0
    bench_t0 = time.perf_counter()

    for combo in combos:
        model, variant = combo["model"], combo["variant"]
        label = combo_label(combo)
        print(f"=== Modelo: {model}  |  Prompt: {variant}  |  "
              f"ctx={combo['num_ctx']} max_tokens={combo['max_tokens']} "
              f"think={'ON' if combo['think'] else 'OFF'} ===")
        latencies, valid, errors, total, trunc, objs = [], 0, 0, 0, 0, []
        thought = 0  # cuántas veces el modelo razonó de verdad
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
                             suffix=f"{label} | {name} run {r}/{runs} | "
                                    f"prom {fmt_secs(avg_so_far)} | ETA {fmt_secs(eta)}")
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
                            objs.append(len(res["parsed"].get("objetos", [])))
                    if res["finish_reason"] == "length":
                        trunc += 1
                    if res.get("did_think"):
                        thought += 1
                except requests.RequestException as e:
                    errors += 1
                    progress_bar(done, total_calls, suffix=f"ERROR en {name}")
                    print(f"\n  [!] {name}: ERROR {e}")
                    continue
            # Refresca la barra al cerrar cada imagen.
            avg_so_far = statistics.mean(latencies) if latencies else 0.0
            eta = avg_so_far * (total_calls - done)
            progress_bar(done, total_calls,
                         suffix=f"{label} | hecho {name} | prom {fmt_secs(avg_so_far)} | "
                                f"ETA {fmt_secs(eta)}")
        progress_bar(done, total_calls, suffix=f"{label} completo")

        # Tabla por imagen de esta combinación (tiempo medio por imagen).
        print(f"\n  Tiempo por imagen ({label}):")
        for name, lats in per_image.items():
            if lats:
                print(f"    {name:28s} prom {fmt_secs(statistics.mean(lats)):>7s}  "
                      f"(min {fmt_secs(min(lats))} / max {fmt_secs(max(lats))})")
            else:
                print(f"    {name:28s} sin datos (errores)")

        stats = _stats(latencies, valid, trunc, objs, errors, total)
        stats["model"] = model
        stats["variant"] = variant
        stats["max_tokens"] = combo["max_tokens"]
        stats["num_ctx"] = combo["num_ctx"]
        stats["think"] = combo["think"]
        stats["thinking_supported"] = model_supports_thinking(model, url)
        stats["thought"] = thought  # cuántas corridas razonaron de verdad
        results[label] = stats
        print()

    bench_wall = time.perf_counter() - bench_t0

    # ---------------- Resumen comparativo + tiempos ----------------
    header = (f"{'Modelo':16s} {'Prompt':13s} {'ctx':>6s} {'maxtok':>6s} {'thk':>3s} "
              f"{'P50':>7s} {'P95':>7s} {'Media':>7s} {'Min':>7s} {'Max':>7s} "
              f"{'Total':>8s} {'JSON%':>7s} {'Cortes':>6s} {'~obj':>5s} {'Errs':>5s}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results.values():
        print(f"{r['model']:16s} {r['variant']:13s} {r['num_ctx']:>6d} {r['max_tokens']:>6d} "
              f"{('ON' if r['think'] else 'OFF'):>3s} {fmt_secs(r['p50']):>7s} "
              f"{fmt_secs(r['p95']):>7s} {fmt_secs(r['mean']):>7s} {fmt_secs(r['min']):>7s} "
              f"{fmt_secs(r['max']):>7s} {fmt_secs(r['total_time']):>8s} {r['json_rate']:6.1f}% "
              f"{r['truncadas']:6d} {r['avg_objetos']:5.1f} {r['errors']:5d}")
    print("=" * len(header))
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
            print(f"   Para fijarla en config.json: \"model\"={r['model']!r}, "
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
    ap.add_argument("--max-tokens", nargs="+", type=int,
                    default=cfg.get("benchmark_max_tokens", [4096]),
                    help="Uno o VARIOS topes de tokens de SALIDA / num_predict a comparar "
                         "(ej. --max-tokens 4096 8192)")
    ap.add_argument("--num-ctx", nargs="+", type=int,
                    default=cfg.get("benchmark_num_ctx", [8192]),
                    help="Una o VARIAS ventanas de contexto a comparar (ej. --num-ctx 8192 16384)")
    ap.add_argument("--think", nargs="+", default=cfg.get("benchmark_think", [True]),
                    help="Uno o VARIOS valores de razonamiento a comparar: true/false "
                         "(ej. --think true false). Solo aplica a modelos con capability 'thinking'.")
    args = ap.parse_args()

    # Validar variantes contra el scope elegido.
    valid_variants = list(PROMPT_VARIANTS[args.scope])
    unknown = [v for v in (args.variants or []) if v and v not in valid_variants]
    if unknown:
        print(f"[ERROR] Variante(s) desconocida(s) para scope '{args.scope}': {unknown}",
              file=sys.stderr)
        print(f"        Disponibles: {valid_variants}", file=sys.stderr)
        sys.exit(1)

    # Parsear los valores de think (true/false/...). None = no se entendió.
    think_vals = [parse_bool(t) for t in (args.think if isinstance(args.think, list) else [args.think])]
    if any(t is None for t in think_vals):
        print(f"[ERROR] Valor(es) de --think inválido(s): {args.think}. Usá true/false.",
              file=sys.stderr)
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
                        max_tokens=args.max_tokens, think=think_vals, url=args.url,
                        num_ctx=args.num_ctx, variants=args.variants)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
