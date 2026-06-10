#!/usr/bin/env python3
"""
menu.py — Menú interactivo del PoC de VLM.

Corré ESTE archivo (botón Play del IDE o `python3 menu.py`) y elegí todo desde
un menú. Hay dos formas de analizar:

  1) ANALIZAR (smoke test): una imagen, imprime el razonamiento en vivo + JSON.
  2) BENCHMARK: abre un submenú donde elegís QUÉ imágenes, QUÉ modelos, cuántas
     runs, qué prompt y qué contexto, y corre todo con barra de progreso y
     reporte de tiempos (por imagen, total, promedio, P50/P95) + % de JSON.

Las opciones se guardan en config.json, así la próxima vez arranca con lo último
que usaste. Para usarlo SIN menú (línea de comandos con flags), mirá el README.md.

Requisitos:  pip install requests
"""
import os
import sys

# El código vive en src/. Lo agregamos al path para poder importarlo tanto si
# corrés `python3 menu.py` desde la raíz como desde el IDE (botón Play).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import requests

from vlm_common import (
    PROMPT_VARIANTS,
    SCOPES,
    list_images,
    load_config,
    save_config,
)
from smoke_test import run_smoke
from benchmark import run_benchmark


# --------------------------------------------------------------------------- #
# Helpers de UI (entrada por consola)
# --------------------------------------------------------------------------- #
def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nChau!")
        raise SystemExit(0)


def list_ollama_models(url):
    """Lista los modelos instalados consultando /api/tags. [] si falla."""
    host = url.split("/v1")[0]
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, KeyError, ValueError):
        return []


def choose_from_list(title, options, current):
    """Menú numerado (selección simple). Enter vacío = mantener el actual."""
    print(f"\n{title}  (actual: {current})")
    for i, opt in enumerate(options, 1):
        marca = " <-- actual" if opt == current else ""
        print(f"  {i}) {opt}{marca}")
    print("  0) escribir un valor a mano")
    sel = ask("Elegí número (Enter = dejar actual): ")
    if sel == "":
        return current
    if sel == "0":
        return ask("Valor: ") or current
    if sel.isdigit() and 1 <= int(sel) <= len(options):
        return options[int(sel) - 1]
    print("[!] Opción inválida, dejo el actual.")
    return current


def choose_multi(title, options, current):
    """Selección MÚLTIPLE. Devuelve una lista (subconjunto de `options`).

    Acepta: '1,3,5'  |  rangos '1-4'  |  combinado '1-3,7'  |  'all'  |  'none'.
    Enter vacío = dejar la selección actual.
    """
    print(f"\n{title}")
    cur = set(current)
    for i, opt in enumerate(options, 1):
        marca = " *" if opt in cur else ""
        print(f"  {i}) {opt}{marca}")
    print("  (lo marcado con * es lo elegido ahora)")
    sel = ask("Elegí (ej: 1,3,5  ó  1-4,7  ó  'all'  ó  'none'; Enter = dejar): ")
    if sel == "":
        return current
    low = sel.lower()
    if low in ("all", "todo", "todos", "todas", "*"):
        return list(options)
    if low in ("none", "ninguna", "ninguno", "0"):
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
    # dedup preservando orden
    seen, out = set(), []
    for o in chosen:
        if o not in seen:
            seen.add(o)
            out.append(o)
    if not out:
        print("[!] No entendí la selección, dejo lo actual.")
        return current
    return out


# --------------------------------------------------------------------------- #
# Acciones de CONFIG (smoke test)
# --------------------------------------------------------------------------- #
def pick_model(cfg):
    models = list_ollama_models(cfg["url"])
    if not models:
        print("[!] No pude listar modelos de Ollama (¿está corriendo?). Escribilo a mano.")
        cfg["model"] = ask(f"Modelo (actual: {cfg['model']}): ") or cfg["model"]
    else:
        cfg["model"] = choose_from_list("Modelo de Ollama", models, cfg["model"])


def pick_image(cfg):
    folder = ask(f"Carpeta de imágenes (Enter = {cfg['folder']}): ") or cfg["folder"]
    cfg["folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No hay imágenes en {folder}. Escribí la ruta a mano.")
        cfg["image"] = ask(f"Imagen (actual: {cfg['image']}): ") or cfg["image"]
    else:
        cfg["image"] = choose_from_list("Imagen", imgs, cfg["image"])


def pick_scope(cfg, key="scope"):
    keys = list(SCOPES)
    labels = [f"{k} — {SCOPES[k]['label']}" for k in keys]
    current = cfg.get(key, keys[0])
    current_label = f"{current} — {SCOPES[current]['label']}"
    chosen = choose_from_list("Modo de detección", labels, current_label)
    cfg[key] = chosen.split(" — ")[0].strip()
    if cfg[key] not in SCOPES:
        cfg[key] = keys[0]


def pick_variant(cfg, scope_key="scope", variant_key="variant"):
    """Elige la variante de prompt activa para el scope dado."""
    scope = cfg[scope_key]
    variants = list(PROMPT_VARIANTS[scope])
    current = cfg.get(variant_key) if cfg.get(variant_key) in variants else variants[0]
    chosen = choose_from_list(f"Variante de prompt (scope: {scope})", variants, current)
    cfg[variant_key] = chosen


def explain_ctx():
    print("\n  ── max_tokens vs num_ctx ─────────────────────────────────────────")
    print("  • num_ctx    = ventana de contexto COMPLETA: entrada (system+user+")
    print("                 imagen) + salida (razonamiento+respuesta). Es el número")
    print("                 que ves en `ollama ps`. Más num_ctx = la imagen entra a")
    print("                 más resolución (más detalle) pero el prefill es MÁS LENTO.")
    print("  • max_tokens = tope de lo que el modelo GENERA (razonamiento+respuesta),")
    print("                 o sea num_predict. Si lo alcanza, corta -> finish_reason")
    print("                 'length' y el JSON puede venir vacío.")
    print("  • Se combinan: salida real = min(max_tokens, num_ctx − tokens_entrada).")
    print("                 Si num_ctx es muy chico, la imagen le come lugar a la")
    print("                 respuesta y se traba; si es muy grande, tarda más.")
    print("  ──────────────────────────────────────────────────────────────────")


def set_ctx(cfg, mt_key="max_tokens", nc_key="num_ctx"):
    explain_ctx()
    v = ask(f"max_tokens (actual: {cfg[mt_key]}, Enter = dejar): ")
    if v.isdigit():
        cfg[mt_key] = int(v)
    c = ask(f"num_ctx (actual: {cfg[nc_key]}, Enter = dejar): ")
    if c.isdigit():
        cfg[nc_key] = int(c)


def toggle_think(cfg, key="think"):
    cfg[key] = not cfg[key]
    if cfg[key]:
        print("-> Razonamiento (think): ON — se imprime en vivo lo que piensa el modelo.")
    else:
        print("-> Razonamiento (think): OFF — OJO: en qwen3-vl think=false NO lo apaga "
              "del todo, sólo acorta el razonamiento.")


def show_config(cfg):
    print("\n" + "=" * 52)
    print(" CONFIG ACTUAL (config.json)")
    print("=" * 52)
    print("  [Smoke test]")
    print(f"    Modelo           : {cfg['model']}")
    print(f"    Imagen           : {cfg['image']}")
    print(f"    Modo detección   : {cfg['scope']} ({SCOPES[cfg['scope']]['label']})")
    print(f"    Variante prompt  : {cfg.get('variant', '(default)')}")
    print(f"    think            : {'ON' if cfg['think'] else 'OFF'}")
    print(f"    max_tokens/num_ctx: {cfg['max_tokens']} / {cfg['num_ctx']}")
    print("  [Benchmark]")
    n_img = cfg.get("benchmark_images") or "TODAS"
    print(f"    Carpeta          : {cfg['folder']}")
    print(f"    Imágenes         : {n_img}")
    print(f"    Modelos          : {', '.join(cfg['benchmark_models'])}")
    print(f"    Runs/imagen      : {cfg['benchmark_runs']}")
    variants = cfg.get("benchmark_variants") or [cfg.get("benchmark_variant")]
    print(f"    Modo / prompts   : {cfg.get('benchmark_scope')} / {', '.join(v for v in variants if v)}")
    print(f"    think            : {'ON' if cfg.get('benchmark_think') else 'OFF'}")
    print(f"    max_tokens/num_ctx: {cfg.get('benchmark_max_tokens')} / {cfg.get('benchmark_num_ctx')}")
    print(f"  URL Ollama         : {cfg['url']}")
    print("=" * 52)


# --------------------------------------------------------------------------- #
# Submenú de BENCHMARK
# --------------------------------------------------------------------------- #
def bench_pick_images(cfg):
    folder = ask(f"Carpeta de imágenes (Enter = {cfg['folder']}): ") or cfg["folder"]
    cfg["folder"] = folder
    imgs = list_images(folder)
    if not imgs:
        print(f"[!] No hay imágenes en {folder}.")
        return
    names = [os.path.basename(p) for p in imgs]
    current = cfg.get("benchmark_images") or names  # [] significaba TODAS
    chosen = choose_multi(f"Imágenes del benchmark en '{folder}'", names, current)
    # Si eligió todas, guardamos [] (= TODAS, se adapta si agregás fotos).
    cfg["benchmark_images"] = [] if set(chosen) == set(names) else chosen
    sel = cfg["benchmark_images"] or names
    print(f"-> {len(sel)} imagen(es) seleccionada(s).")


def bench_pick_models(cfg):
    models = list_ollama_models(cfg["url"])
    if not models:
        print("[!] No pude listar modelos de Ollama. Escribilos separados por coma.")
        raw = ask(f"Modelos (Enter = {', '.join(cfg['benchmark_models'])}): ")
        if raw:
            cfg["benchmark_models"] = [m.strip() for m in raw.split(",") if m.strip()]
        return
    chosen = choose_multi("Modelos a comparar", models, cfg["benchmark_models"])
    if chosen:
        cfg["benchmark_models"] = chosen


def bench_set_runs(cfg):
    runs = ask(f"Runs por imagen (actual: {cfg['benchmark_runs']}): ")
    if runs.isdigit() and int(runs) >= 1:
        cfg["benchmark_runs"] = int(runs)


def bench_pick_variants(cfg):
    """Selección MÚLTIPLE de variantes de prompt a comparar en el benchmark.

    Comparar varias prompts en una corrida reemplaza al viejo 05_prompt_test.py.
    """
    scope = cfg.get("benchmark_scope", "industrial")
    options = list(PROMPT_VARIANTS[scope])
    current = [v for v in (cfg.get("benchmark_variants") or []) if v in options]
    if not current:
        current = options[:1]
    chosen = choose_multi(f"Variantes de prompt a comparar (scope: {scope})", options, current)
    cfg["benchmark_variants"] = chosen or current


def bench_run(cfg):
    """Resuelve las imágenes elegidas y dispara el benchmark."""
    all_imgs = list_images(cfg["folder"])
    if not all_imgs:
        print(f"[!] No hay imágenes en {cfg['folder']}.")
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
│              BENCHMARK — configurar            │
├────────────────────────────────────────────────┤
│   1) Elegir imágenes (cuáles / cuántas)        │
│   2) Elegir modelos                            │
│   3) Runs por imagen                           │
│   4) Modo de detección (industrial / todo)     │
│   5) Prompts a comparar (1 o varias)           │
│   6) max_tokens / num_ctx (contexto)           │
│   7) Razonamiento think (ON/OFF)               │
│                                                │
│   8) ▶ CORRER BENCHMARK                        │
│   0) Volver al menú principal                  │
└────────────────────────────────────────────────┘"""


def benchmark_menu(cfg):
    while True:
        # Resumen rápido de qué se va a correr.
        all_imgs = list_images(cfg["folder"])
        n_img = len(cfg.get("benchmark_images") or all_imgs)
        runs = cfg["benchmark_runs"]
        variants = cfg.get("benchmark_variants") or []
        n_models = len(cfg["benchmark_models"])
        n_calls = n_img * runs * n_models * max(len(variants), 1)
        print(f"\n  >> {n_img} img × {runs} runs × {n_models} modelo(s) × {len(variants)} prompt(s) "
              f"= {n_calls} llamadas | num_ctx={cfg.get('benchmark_num_ctx')} "
              f"max_tokens={cfg.get('benchmark_max_tokens')} | prompts={', '.join(variants)}")
        print(BENCH_MENU)
        choice = ask("Opción: ")
        if choice == "1":
            bench_pick_images(cfg); save_config(cfg)
        elif choice == "2":
            bench_pick_models(cfg); save_config(cfg)
        elif choice == "3":
            bench_set_runs(cfg); save_config(cfg)
        elif choice == "4":
            pick_scope(cfg, key="benchmark_scope")
            # Si las variantes ya no existen para el nuevo scope, resetear.
            scope = cfg["benchmark_scope"]
            valid = list(PROMPT_VARIANTS[scope])
            kept = [v for v in (cfg.get("benchmark_variants") or []) if v in valid]
            cfg["benchmark_variants"] = kept or valid[:1]
            save_config(cfg)
        elif choice == "5":
            bench_pick_variants(cfg)
            save_config(cfg)
        elif choice == "6":
            set_ctx(cfg, mt_key="benchmark_max_tokens", nc_key="benchmark_num_ctx")
            save_config(cfg)
        elif choice == "7":
            toggle_think(cfg, key="benchmark_think"); save_config(cfg)
        elif choice == "8":
            save_config(cfg)
            bench_run(cfg)
        elif choice == "0" or choice.lower() in ("q", "volver", "back"):
            return
        else:
            print("[!] Opción inválida.")


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
MENU = """
┌────────────────────────────────────────────────┐
│           VLM PoC — Menú principal             │
├────────────────────────────────────────────────┤
│  ANALIZAR                                      │
│   1) Smoke test (1 imagen, razonamiento vivo)  │
│   2) Benchmark (modelos × prompts, submenú)    │
│                                                │
│  CONFIGURAR SMOKE TEST (se guarda en config)   │
│   3) Modelo                                    │
│   4) Imagen                                    │
│   5) Modo de detección (industrial / todo)     │
│   6) Variante de prompt                        │
│   7) Razonamiento think (ON/OFF)               │
│   8) max_tokens / num_ctx                      │
│   9) Ver config actual                         │
│                                                │
│   0) Salir                                     │
└────────────────────────────────────────────────┘"""


def main():
    cfg = load_config()
    show_config(cfg)

    while True:
        print(MENU)
        choice = ask("Opción: ")

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
        elif choice == "0" or choice.lower() in ("q", "salir", "exit"):
            save_config(cfg)
            print("Config guardada. Chau!")
            break
        else:
            print("[!] Opción inválida.")


if __name__ == "__main__":
    main()
