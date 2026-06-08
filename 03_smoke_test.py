#!/usr/bin/env python3
"""
03_smoke_test.py — Validación rápida del VLM sobre una imagen del lab.

Manda una imagen al endpoint OpenAI-compatible de Ollama (el mismo que va a
consumir el Silk AI Proxy Gateway / F1.9) y pide identificar instrumentos
industriales devolviendo JSON con bounding boxes.

Requisitos:  pip install requests
Uso:
    python3 03_smoke_test.py imagen_planta.jpg
    python3 03_smoke_test.py imagen_planta.jpg --model qwen3-vl:4b
"""
import argparse
import base64
import json
import re
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

# Prompt industrial en español rioplatense. Pedimos SOLO JSON para poder parsear.
SYSTEM_PROMPT = (
    "Sos un asistente de inspección industrial para minería y oil & gas. "
    "Analizás imágenes de planta y respondés SIEMPRE en JSON válido, sin texto extra, "
    "sin markdown, sin explicaciones."
)

USER_PROMPT = (
    "Identificá manómetros, válvulas y EPP (elementos de protección personal) visibles "
    "en la imagen. Para cada objeto devolvé un item con: "
    '"tipo" (manometro|valvula|epp|otro), '
    '"bbox" como [x_min, y_min, x_max, y_max] normalizado entre 0 y 1, '
    '"lectura" (string con el valor del instrumento si es legible, o null), '
    '"confianza" (0 a 1). '
    'Respondé con un objeto JSON: {"objetos": [ ... ]}.'
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(text: str):
    """Intenta parsear JSON aunque el modelo agregue ruido, fences o razonamiento."""
    text = text.strip()
    # quitar bloques de razonamiento <think>...</think> (modelos thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # quitar fences ```json ... ```
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        # buscar el primer { ... } balanceado de forma simple
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), True
            except json.JSONDecodeError:
                pass
    return text, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Ruta a la imagen del lab (jpg/png)")
    ap.add_argument("--model", default="qwen3-vl:8b", help="Modelo de Ollama")
    ap.add_argument("--url", default=OLLAMA_URL)
    args = ap.parse_args()

    try:
        img_b64 = encode_image(args.image)
    except FileNotFoundError:
        print(f"[ERROR] No encuentro la imagen: {args.image}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 768,
        # Modo JSON: fuerza al modelo a devolver SOLO JSON válido.
        "response_format": {"type": "json_object"},
    }

    print(f"[..] Enviando '{args.image}' al modelo '{args.model}' ...")
    t0 = time.perf_counter()
    try:
        r = requests.post(args.url, json=payload, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Falló la request a Ollama: {e}", file=sys.stderr)
        print("        ¿Está corriendo el servicio? curl http://localhost:11434/api/version")
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    parsed, ok = extract_json(content)
    print("\n========== RESULTADO ==========")
    print(f"Modelo:        {args.model}")
    print(f"Latencia E2E:  {elapsed:.2f} s")
    print(f"JSON válido:   {'SÍ' if ok else 'NO'}")
    print("-------------------------------")
    if ok:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        n = len(parsed.get("objetos", [])) if isinstance(parsed, dict) else 0
        print(f"\n[OK] Objetos detectados: {n}")
    else:
        print("[!] No devolvió JSON parseable. Respuesta cruda:")
        print(content)
    print("===============================")


if __name__ == "__main__":
    main()
