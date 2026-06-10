#!/usr/bin/env python3
"""
03_smoke_test.py — Validación rápida del VLM sobre una imagen del lab.

Manda una imagen al endpoint NATIVO de Ollama (/api/chat, vía vlm_common) y pide
identificar objetos devolviendo JSON con bounding boxes. El contrato JSON es el
que va a consumir el Silk AI Proxy Gateway / F1.9.

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
    OLLAMA_HOST,
    PROMPT_VARIANTS,
    SCOPES,
    encode_image,
    image_size,
    load_config,
    query_vlm,
    render_result,
)


def run_smoke(image, model, scope="industrial", max_tokens=8192,
              think=True, url=OLLAMA_HOST, num_ctx=16384, variant=None):
    """Corre el smoke test sobre una imagen e imprime el resultado.

    Con think=True imprime EN VIVO lo que el modelo va razonando (verbose).
    `variant` elige la variante de prompt (ver PROMPT_VARIANTS); None = default del scope.
    """
    try:
        img_b64 = encode_image(image)
    except FileNotFoundError:
        print(f"[ERROR] No encuentro la imagen: {image}", file=sys.stderr)
        return False

    print(f"[..] Enviando '{image}' al modelo '{model}' "
          f"(modo: {scope}, prompt: {variant or 'default'}) ...")
    try:
        res = query_vlm(img_b64, model, scope=scope, max_tokens=max_tokens,
                        think=think, url=url, timeout=300, num_ctx=num_ctx,
                        verbose=True, size=image_size(image), variant=variant)
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
    ap.add_argument("--variant", default=cfg.get("variant"),
                    help="Variante de prompt (ej. v1_original, v2_antiloop). "
                         "Default: el de config.json. Ver: python3 05_prompt_test.py --list")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", type=int, default=cfg["max_tokens"],
                    help="Tope de tokens de SALIDA / num_predict (incluye razonamiento)")
    ap.add_argument("--num-ctx", type=int, default=cfg.get("num_ctx", 16384),
                    help="Ventana de contexto (entrada+salida); la que ves en `ollama ps`")
    ap.add_argument("--think", dest="think", action="store_true", default=cfg["think"],
                    help="Razonamiento del modelo + impresión en vivo (default ON)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="Pedir think=false (en qwen3-vl no lo apaga del todo, sólo lo acorta)")
    args = ap.parse_args()

    ok = run_smoke(args.image, args.model, scope=args.scope,
                   max_tokens=args.max_tokens, think=args.think, url=args.url,
                   num_ctx=args.num_ctx, variant=args.variant)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
