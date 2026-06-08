#!/usr/bin/env python3
"""
vlm_common.py — Núcleo compartido por el menú, el smoke test y el benchmark.

Acá viven, en un solo lugar para no duplicar:
  - Los MODOS de detección (prompts) -> SCOPES
  - El cliente del VLM contra el endpoint OpenAI-compatible de Ollama
  - El parseo robusto de JSON (tolera razonamiento, fences, ruido)
  - La config persistente en config.json

No se ejecuta solo; lo importan menu.py / 03_smoke_test.py / 04_benchmark.py.
"""
import base64
import json
import os
import re
import time

import requests

# Endpoint OpenAI-compatible de Ollama (el mismo que consume el Silk AI Proxy
# Gateway / F1.9). Se puede pisar por config.json o por --url.
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

# Extensiones de imagen que reconoce el benchmark al barrer una carpeta.
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Archivo de configuración persistente (al lado de este módulo).
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# --------------------------------------------------------------------------- #
# MODOS DE DETECCIÓN (scope)
# --------------------------------------------------------------------------- #
# "industrial": SOLO instrumentos/equipos industriales, con taxonomía cerrada.
# "todo":       CUALQUIER objeto visible (industrial o no), categoría libre.
SCOPES = {
    "industrial": {
        "label": "Solo instrumentos industriales",
        "system": (
            "Sos un asistente de inspección industrial para minería y oil & gas. "
            "Analizás imágenes de planta e identificás cualquier instrumento o equipo. "
            "Respondés SIEMPRE en JSON válido, sin texto extra, sin markdown, sin explicaciones."
        ),
        "user": (
            "Identificá TODOS los instrumentos, equipos u objetos industriales visibles "
            "en la imagen (no solo manómetros). Para cada objeto devolvé un item con: "
            '"tipo" (manometro|termometro|valvula|sensor|epp|otro), '
            '"descripcion" (string corto de qué es, p. ej. "termómetro bimetálico", '
            '"termopozo", "manómetro de presión"), '
            '"bbox" como [x_min, y_min, x_max, y_max] normalizado entre 0 y 1, '
            '"lectura" (string con el valor del instrumento si es legible, o null), '
            '"confianza" (0 a 1). '
            "Si el objeto no encaja en las categorías nombradas usá \"otro\" pero igual "
            "devolvelo con su descripción. Solo devolvé la lista vacía si NO hay ningún "
            "objeto de interés en la imagen. "
            'Respondé con un objeto JSON: {"objetos": [ ... ]}.'
        ),
    },
    "todo": {
        "label": "Cualquier objeto (uso general)",
        "system": (
            "Sos un asistente de visión que identifica cualquier objeto en una imagen. "
            "Respondés SIEMPRE en JSON válido, sin texto extra, sin markdown, sin explicaciones."
        ),
        "user": (
            "Identificá TODOS los objetos visibles en la imagen, sean industriales o no. "
            "Devolvé SIEMPRE un objeto JSON con la clave \"objetos\" (una lista), aunque "
            "haya un solo objeto. Cada item de la lista tiene: "
            '"tipo" (categoría libre en una palabra, p. ej. "persona", "herramienta", '
            '"vehiculo", "animal", "manometro"), '
            '"descripcion" (string corto de qué es), '
            '"bbox" como [x_min, y_min, x_max, y_max] normalizado entre 0 y 1, '
            '"lectura" (texto legible en el objeto si lo hay, o null), '
            '"confianza" (0 a 1). '
            "La lista solo va vacía si la imagen no tiene NINGÚN objeto. "
            'Ejemplo de formato exacto: '
            '{"objetos":[{"tipo":"manometro","descripcion":"...","bbox":[0.1,0.1,0.9,0.9],'
            '"lectura":null,"confianza":0.9}]}'
        ),
    },
}

# --------------------------------------------------------------------------- #
# CONFIG persistente
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "model": "qwen3-vl:4b",            # 4b entra 100% en 8GB de VRAM; el 8b se parte CPU/GPU
    "image": "fotosClean/2.jpeg",      # imagen para el smoke test
    "folder": "fotosClean",            # carpeta para el benchmark
    "scope": "industrial",             # modo de detección (industrial | todo)
    "max_tokens": 4096,                # tope de tokens de salida
    "think": False,                    # razonamiento del modelo (lento); apagado por defecto
    "url": OLLAMA_URL,                 # endpoint de Ollama
    "benchmark_models": ["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"],
    "benchmark_runs": 3,               # repeticiones por imagen en el benchmark
}


def load_config():
    """Devuelve la config mergeada con los defaults (crea el archivo si falta)."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # config corrupta -> caemos a defaults
    else:
        save_config(cfg)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Cliente VLM
# --------------------------------------------------------------------------- #
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(text):
    """Parsea JSON aunque el modelo agregue ruido, fences o razonamiento.

    Devuelve (objeto_o_texto, ok_bool).
    """
    text = (text or "").strip()
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


def query_vlm(img_b64, model, scope="industrial", max_tokens=4096,
              think=False, url=OLLAMA_URL, timeout=180):
    """Manda una imagen al VLM y devuelve un dict con la respuesta + diagnóstico.

    qwen3-vl es un modelo "thinking": si dejamos el razonamiento activo gasta
    tokens y latencia pensando (y a veces se corta dejando content=""). Lo
    apagamos con el flag think:false de Ollama (el switch real; no hace falta
    ensuciar el prompt con /no_think).

    Levanta requests.RequestException si falla la red/el server.
    """
    sp = SCOPES[scope]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sp["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sp["user"]},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},  # fuerza JSON válido
        "think": think,                              # apaga el razonamiento si False
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0

    choice = r.json()["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    parsed, ok = extract_json(content)
    return {
        "elapsed": elapsed,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "reasoning": reasoning,
        "parsed": parsed,
        "ok": ok,
    }


def render_result(model, res):
    """Imprime el resultado de un query_vlm de forma legible (smoke test / menú)."""
    print("\n========== RESULTADO ==========")
    print(f"Modelo:        {model}")
    print(f"Latencia E2E:  {res['elapsed']:.2f} s")
    print(f"finish_reason: {res['finish_reason']}")
    print(f"JSON válido:   {'SÍ' if res['ok'] else 'NO'}")
    print("-------------------------------")
    if res["ok"]:
        parsed = res["parsed"]
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        n = len(parsed.get("objetos", [])) if isinstance(parsed, dict) else 0
        print(f"\n[OK] Objetos detectados: {n}")
    else:
        print("[!] No devolvió JSON parseable.")
        if not res["content"].strip():
            print("[!] content vino VACÍO.")
            if res["finish_reason"] == "length":
                print("    -> Cortado por longitud. Subí max_tokens.")
            if res["reasoning"]:
                print("    -> El modelo gastó tokens en razonamiento. "
                      "Desactivá 'think' o subí max_tokens.")
                print(f"    reasoning_content (primeros 500 chars):\n{res['reasoning'][:500]}")
        else:
            print("Respuesta cruda:")
            print(res["content"])
    print("===============================")
