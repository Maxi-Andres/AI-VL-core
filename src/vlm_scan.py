#!/usr/bin/env python3
"""
vlm_scan.py — Quick VLM check on a single image (one "scan").

Sends an image to Ollama's NATIVE endpoint (/api/chat, via vlm_common) and asks
it to identify objects, returning JSON with bounding boxes. The JSON contract is
the one consumed by the Silk AI Proxy Gateway / F1.9.

This is the VLM counterpart of yolo_scan.py (same idea, but the VLM reasons over
a prompt instead of running a YOLO detector).

Don't want to type flags? Run `python3 menu.py` (interactive menu).

Requirements:  pip install requests
Usage:
    python3 src/vlm_scan.py fotos/clean/2.jpeg
    python3 src/vlm_scan.py fotos/clean/2.jpeg --model qwen3-vl:8b
    python3 src/vlm_scan.py fotos/clean/2.jpeg --scope all --think
"""
import argparse
import sys

import requests

from vlm_common import (
    OLLAMA_HOST,
    SCOPES,
    encode_image,
    image_size,
    load_config,
    query_vlm,
    render_result,
)


def run_vlm_scan(image, model, scope="industrial", max_tokens=8192,
                 think=True, url=OLLAMA_HOST, num_ctx=16384, variant=None,
                 prompt=None):
    """Run a VLM scan on a single image and print the result.

    With think=True it prints the model's reasoning LIVE (verbose).
    `variant` selects the prompt variant (see PROMPT_VARIANTS); None = scope default.
    `prompt` (free-form question about the image) overrides scope/variant: the
    model answers in plain text instead of the structured {objects:[...]} JSON.
    """
    try:
        img_b64 = encode_image(image)
    except FileNotFoundError:
        print(f"[ERROR] Image not found: {image}", file=sys.stderr)
        return False

    label = "free prompt" if prompt else f"mode: {scope}, prompt: {variant or 'default'}"
    print(f"[..] Sending '{image}' to model '{model}' ({label}) ...")
    try:
        res = query_vlm(img_b64, model, scope=scope, max_tokens=max_tokens,
                        think=think, url=url, timeout=300, num_ctx=num_ctx,
                        verbose=True, size=image_size(image), variant=variant,
                        prompt=prompt)
    except requests.RequestException as e:
        print(f"[ERROR] Request to Ollama failed: {e}", file=sys.stderr)
        print("        Is the service running? curl http://localhost:11434/api/version")
        return False

    if prompt:
        # Free-form answer: print the plain-text content, not the JSON contract.
        print("\n========== ANSWER ==========")
        print(f"Model:       {model}")
        print(f"E2E latency: {res['elapsed']:.2f} s")
        print("-------------------------------")
        print(res["content"].strip() or "[!] The model returned an empty answer.")
        print("===============================")
    else:
        render_result(model, res)
    return True


def main():
    cfg = load_config()  # defaults come from config.json
    ap = argparse.ArgumentParser(description="VLM scan on a single image.")
    ap.add_argument("image", nargs="?", default=cfg["image"],
                    help="Path to the image (jpg/png). Default: the one in config.json")
    ap.add_argument("--model", default=cfg["model"],
                    help="Ollama model (4b fits 100%% in 8GB of VRAM; "
                         "the 8b splits CPU/GPU and is ~3x slower)")
    ap.add_argument("--scope", choices=list(SCOPES), default=cfg["scope"],
                    help="Detection mode: industrial (instruments only) | all (any object)")
    ap.add_argument("--variant", default=cfg.get("variant"),
                    help="Prompt variant (e.g. v1_original, v2_antiloop). "
                         "Default: the one in config.json. Variants are compared "
                         "with python3 src/vlm_benchmark.py --variants ...")
    ap.add_argument("--prompt", default=None,
                    help="Free-form question about the image (overrides "
                         "--scope/--variant; the model answers in plain text)")
    ap.add_argument("--url", default=cfg["url"])
    ap.add_argument("--max-tokens", type=int, default=cfg["max_tokens"],
                    help="OUTPUT token cap / num_predict (includes reasoning)")
    ap.add_argument("--num-ctx", type=int, default=cfg.get("num_ctx", 16384),
                    help="Context window (input+output); the one you see in `ollama ps`")
    ap.add_argument("--think", dest="think", action="store_true", default=cfg["think"],
                    help="Model reasoning + live printing (default ON)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="Request think=false (on qwen3-vl it doesn't fully disable it, only shortens it)")
    args = ap.parse_args()

    ok = run_vlm_scan(args.image, args.model, scope=args.scope,
                      max_tokens=args.max_tokens, think=args.think, url=args.url,
                      num_ctx=args.num_ctx, variant=args.variant,
                      prompt=args.prompt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
