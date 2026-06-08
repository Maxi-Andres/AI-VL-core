#!/usr/bin/env python3
"""
03_smoke_test.py — Validación rápida del VLM sobre una imagen del lab.

Manda una imagen al endpoint OpenAI-compatible de Ollama (el mismo que va a
consumir el Silk AI Proxy Gateway / F1.9) y pide identificar objetos
devolviendo JSON con bounding boxes.

¿No querés escribir flags? Corré `python3 menu.py` (menú interactivo).

Requisitos:  pip install requests
Uso:
    python3 03_smoke_test.py fotosClean/2.jpeg
    python3 03_smoke_test.py fotosClean/2.jpeg --model qwen3-vl:8b
    python3 03_smoke_test.py fotosClean/2.jpeg --scope todo --think
"""
import argparse
import sys

import requests

from vlm_common import (
    OLLAMA_URL,
    SCOPES,
    encode_image,
    load_config,
    query_vlm,
    render_result,
)


def run_smoke(image, model, scope="industrial", max_tokens=4096,
              think=False, url=OLLAMA_URL):
    """Corre el smoke test sobre una imagen e imprime el resultado."""
    try:
        img_b64 = encode_image(image)
    except FileNotFoundError:
        print(f"[ERROR] No encuentro la imagen: {image}", file=sys.stderr)
        return False

    print(f"[..] Enviando '{image}' al modelo '{model}' (modo: {scope}) ...")
    try:
        res = query_vlm(img_b64, model, scope=scope, max_tokens=max_tokens,
                        think=think, url=url, timeout=120)
    except requests.RequestException as e:
        print(f"[ERROR] Falló la request a Ollama: {e}", file=sys.stderr)
        print("        ¿Está corriendo el servicio? curl http://localhost:11434/api/version")
        return False

    render_result(model, res)
    return True


def main():
    cfg = load_config()  # los defaults salen de config.json
    ap = argparse.ArgumentParser(description="Smoke test del VLM sobre una imagen.")
    ap.add_argument("image", nargs="?", default=cfg["image"],
                    help="Ruta a la imagen (jpg/png). Default: el de config.json")
    ap.add_argument("--model", default=cfg["model"],
                    help="Modelo de Ollama (4b entra 100%% en 8GB de VRAM; "
                         "el 8b se parte CPU/GPU y es ~3x más lento)")
    ap.add_argument("--scope", choices=list(SCOPES), default=cfg["scope"],
                    help="Modo de detección: industrial (solo instrumentos) | todo (cualquier objeto)")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", type=int, default=cfg["max_tokens"],
                    help="Tope de tokens de salida (incluye razonamiento)")
    ap.add_argument("--think", action="store_true", default=cfg["think"],
                    help="Permitir razonamiento del modelo (más lento; por defecto desactivado)")
    args = ap.parse_args()

    ok = run_smoke(args.image, args.model, scope=args.scope,
                   max_tokens=args.max_tokens, think=args.think, url=args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
