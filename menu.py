#!/usr/bin/env python3
"""
menu.py — Menú interactivo del PoC de VLM.

Corré ESTE archivo (botón Play del IDE o `python3 menu.py`) y elegí todo desde
un menú: modelo, imagen, modo de detección, etc. Las opciones se guardan en
config.json, así la próxima vez ya arranca con lo último que usaste.

Para usarlo SIN menú (línea de comandos con flags), mirá el README.md.

Requisitos:  pip install requests
"""
import glob
import os
import re

import requests

from vlm_common import (
    IMG_EXTS,
    PROMPT_VARIANTS,
    SCOPES,
    load_config,
    save_config,
)
import importlib

# Importamos los runners de los scripts numerados (nombre no-identificador).
run_smoke = importlib.import_module("03_smoke_test").run_smoke
run_benchmark = importlib.import_module("04_benchmark").run_benchmark
run_prompt_test = importlib.import_module("05_prompt_test").run_prompt_test


# --------------------------------------------------------------------------- #
# Helpers de UI (entrada por consola)
# --------------------------------------------------------------------------- #
def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nChau!")
        raise SystemExit(0)


def natural_key(path):
    """Ordena 1,2,...,10 en vez de 1,10,2 (orden 'natural')."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return (int(nums[0]) if nums else 0, os.path.basename(path))


def list_images(folder):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(imgs, key=natural_key)


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
    """Menú numerado. Enter vacío = mantener el valor actual. Devuelve la opción."""
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


# --------------------------------------------------------------------------- #
# Acciones del menú
# --------------------------------------------------------------------------- #
def pick_model(cfg):
    models = list_ollama_models(cfg["url"])
    if not models:
        print("[!] No pude listar modelos de Ollama (¿está corriendo?). "
              "Escribilo a mano.")
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


def pick_scope(cfg):
    keys = list(SCOPES)
    labels = [f"{k} — {SCOPES[k]['label']}" for k in keys]
    current_label = f"{cfg['scope']} — {SCOPES[cfg['scope']]['label']}"
    chosen = choose_from_list("Modo de detección", labels, current_label)
    cfg["scope"] = chosen.split(" — ")[0].strip()
    if cfg["scope"] not in SCOPES:
        cfg["scope"] = keys[0]


def pick_variant(cfg):
    """Elige la variante de prompt activa para el scope actual."""
    scope = cfg["scope"]
    variants = list(PROMPT_VARIANTS[scope])
    current = cfg.get("variant") if cfg.get("variant") in variants else variants[0]
    chosen = choose_from_list(f"Variante de prompt (scope: {scope})", variants, current)
    cfg["variant"] = chosen


def compare_prompts(cfg):
    """Corre el comparador A/B de prompts sobre la imagen actual."""
    scope = cfg["scope"]
    runs = ask(f"Runs por variante (Enter = 2): ")
    runs = int(runs) if runs.isdigit() else 2
    run_prompt_test([cfg["image"]], scope, list(PROMPT_VARIANTS[scope]),
                    cfg["model"], runs=runs, max_tokens=cfg["max_tokens"],
                    think=cfg["think"], url=cfg["url"], num_ctx=cfg["num_ctx"])


def toggle_think(cfg):
    cfg["think"] = not cfg["think"]
    if cfg["think"]:
        print("-> Razonamiento (think): ON — se imprime en vivo lo que piensa el modelo.")
    else:
        print("-> Razonamiento (think): OFF — OJO: en qwen3-vl think=false NO lo apaga "
              "del todo, sólo acorta el razonamiento. Igual conviene dejar margen de tokens.")


def set_max_tokens(cfg):
    print("  max_tokens = tope de tokens de SALIDA (num_predict). Incluye el razonamiento.")
    print("  num_ctx    = ventana total entrada+salida (la que muestra `ollama ps`).")
    print("               Subirla da más resolución a la imagen y más lugar para razonar.")
    v = ask(f"max_tokens (actual: {cfg['max_tokens']}, Enter = dejar): ")
    if v.isdigit():
        cfg["max_tokens"] = int(v)
    c = ask(f"num_ctx (actual: {cfg['num_ctx']}, Enter = dejar): ")
    if c.isdigit():
        cfg["num_ctx"] = int(c)


def benchmark_settings(cfg):
    runs = ask(f"Runs por imagen (actual: {cfg['benchmark_runs']}): ")
    if runs.isdigit():
        cfg["benchmark_runs"] = int(runs)
    models = list_ollama_models(cfg["url"])
    if models:
        print("\nModelos disponibles:", ", ".join(models))
    raw = ask(f"Modelos a comparar separados por coma\n"
              f"  (Enter = {', '.join(cfg['benchmark_models'])}): ")
    if raw:
        cfg["benchmark_models"] = [m.strip() for m in raw.split(",") if m.strip()]


def show_config(cfg):
    print("\n" + "=" * 50)
    print(" CONFIG ACTUAL (config.json)")
    print("=" * 50)
    print(f"  Modelo            : {cfg['model']}")
    print(f"  Imagen (smoke)    : {cfg['image']}")
    print(f"  Carpeta (bench)   : {cfg['folder']}")
    print(f"  Modo detección    : {cfg['scope']} ({SCOPES[cfg['scope']]['label']})")
    print(f"  Variante prompt   : {cfg.get('variant', '(default)')}")
    print(f"  Razonamiento think: {'ON' if cfg['think'] else 'OFF'}")
    print(f"  max_tokens (salida): {cfg['max_tokens']}")
    print(f"  num_ctx (ventana) : {cfg['num_ctx']}")
    print(f"  URL Ollama        : {cfg['url']}")
    print(f"  Bench: runs       : {cfg['benchmark_runs']}")
    print(f"  Bench: modelos    : {', '.join(cfg['benchmark_models'])}")
    print("=" * 50)


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
MENU = """
┌────────────────────────────────────────────────┐
│           VLM PoC — Menú principal             │
├────────────────────────────────────────────────┤
│  ANALIZAR                                      │
│   1) Smoke test (1 imagen)                     │
│   2) Benchmark (carpeta, P50/P95, JSON%)       │
│   3) Comparar prompts (A/B: velocidad/JSON)    │
│                                                │
│  CONFIGURAR (se guarda en config.json)         │
│   4) Modelo                                    │
│   5) Imagen                                    │
│   6) Modo de detección (industrial / todo)     │
│   7) Variante de prompt                        │
│   8) Razonamiento think (ON/OFF)               │
│   9) max_tokens / num_ctx                      │
│  10) Ajustes del benchmark (runs / modelos)    │
│  11) Ver config actual                         │
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
            save_config(cfg)
            run_benchmark(cfg["folder"], cfg["benchmark_models"],
                          runs=cfg["benchmark_runs"], scope=cfg["scope"],
                          max_tokens=cfg["max_tokens"], think=cfg["think"],
                          url=cfg["url"], num_ctx=cfg["num_ctx"],
                          variant=cfg.get("variant"))
        elif choice == "3":
            save_config(cfg)
            compare_prompts(cfg)
        elif choice == "4":
            pick_model(cfg); save_config(cfg)
        elif choice == "5":
            pick_image(cfg); save_config(cfg)
        elif choice == "6":
            pick_scope(cfg); save_config(cfg)
        elif choice == "7":
            pick_variant(cfg); save_config(cfg)
        elif choice == "8":
            toggle_think(cfg); save_config(cfg)
        elif choice == "9":
            set_max_tokens(cfg); save_config(cfg)
        elif choice == "10":
            benchmark_settings(cfg); save_config(cfg)
        elif choice == "11":
            show_config(cfg)
        elif choice == "0" or choice.lower() in ("q", "salir", "exit"):
            save_config(cfg)
            print("Config guardada. Chau!")
            break
        else:
            print("[!] Opción inválida.")


if __name__ == "__main__":
    main()
