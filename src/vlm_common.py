#!/usr/bin/env python3
"""
vlm_common.py — Núcleo compartido por el menú, el smoke test y el benchmark.

Acá viven, en un solo lugar para no duplicar:
  - Los MODOS de detección (prompts) -> SCOPES
  - El cliente del VLM contra el endpoint NATIVO de Ollama (/api/chat)
  - El parseo robusto de JSON + normalización de bounding boxes
  - La config persistente en config.json

No se ejecuta solo; lo importan menu.py / 03_smoke_test.py / 04_benchmark.py.

¿Por qué el endpoint nativo y no el OpenAI-compatible?
---------------------------------------------------------------------------
qwen3-vl es un modelo "thinking" (razona antes de responder). El endpoint
OpenAI-compatible (/v1/chat/completions) IGNORA el flag `think` y, peor,
mezcla el razonamiento con la respuesta: si el modelo gasta todo el presupuesto
de tokens pensando, `content` vuelve VACÍO y no hay JSON.
El endpoint nativo (/api/chat) separa el razonamiento (campo `thinking`) de la
respuesta (campo `content`), así que el JSON sale limpio y además podemos
imprimir en vivo lo que el modelo va pensando.
"""
import base64
import json
import os
import re
import struct
import time

import requests

# Host base de Ollama. Se le cuelgan /api/chat y /api/tags.
# Se puede pisar por config.json o por --url (acepta también URLs viejas con /v1).
OLLAMA_HOST = "http://localhost:11434"
# Compatibilidad hacia atrás: algunos scripts/imports usan OLLAMA_URL.
OLLAMA_URL = OLLAMA_HOST

# Extensiones de imagen que reconoce el benchmark al barrer una carpeta.
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Archivo de configuración persistente (al lado de este módulo).
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def host_of(url):
    """Normaliza cualquier URL de Ollama a su host base.

    Acepta el host pelado o URLs con sufijos (incluida la vieja /v1/chat/completions)
    para no romper config.json existentes.
    """
    if not url:
        return OLLAMA_HOST
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1",
                   "/api/chat", "/api/generate"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


# --------------------------------------------------------------------------- #
# MODOS DE DETECCIÓN (scope) + VARIANTES DE PROMPT (intercambiables / A-B test)
# --------------------------------------------------------------------------- #
# Hay dos scopes:
#   "industrial": instrumentos/equipos de industria (oil & gas, minería).
#   "todo":       CUALQUIER objeto visible (industrial o no), categoría libre.
#
# Cada scope tiene VARIANTES de prompt en PROMPT_VARIANTS, para poder comparar
# cuál es más rápida/mejor (ver 05_prompt_test.py). Los prompts están en INGLÉS
# (el modelo razona en inglés; se busca menos overhead). Las KEYS y los valores
# de `tipo` del JSON quedan en español porque son el contrato VLM->VLA.
#
# La variante ACTIVA por defecto (la que usan menú/smoke/benchmark) la define
# DEFAULT_VARIANT. Para "intercambiar" el prompt, cambiá DEFAULT_VARIANT acá,
# o pasá --variant en los scripts, o elegilo en el menú.
SCOPE_LABELS = {
    "industrial": "Industrial instruments (oil & gas / mining)",
    "todo": "Any object (general use)",
}

# --- Industrial: variante v1 = el prompt ORIGINAL (corto, más rápido) --------
_INDUSTRIAL_V1 = {
    "system": (
        "You are an expert assistant in industrial inspection for mining and "
        "oil & gas. You recognize ANY plant instrument or equipment. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Identify ALL the industrial instruments, equipment or objects visible "
        "in the image. You must be able to recognize ANY industrial instrument, "
        "do not limit yourself to a list. As a REFERENCE (not exhaustive), "
        "typical instruments by family:\n"
        "- presion: pressure gauge, pressure transmitter, pressure switch, vacuum gauge.\n"
        "- temperatura: bimetallic thermometer, thermocouple, RTD/Pt100, pyrometer, thermowell.\n"
        "- caudal: electromagnetic/turbine/ultrasonic flow meter, Coriolis, rotameter, orifice plate.\n"
        "- nivel: ultrasonic, radar, hydrostatic transmitter, float, capacitive.\n"
        "- electrica: power analyzer, wattmeter, multimeter, CT, VT.\n"
        "- analisis: O2/CO/CO2 analyzer, pH meter, conductivity meter, turbidimeter, chromatograph.\n"
        "- control: control valve, positioner, actuator, VFD, PID controller.\n"
        "- vibracion: accelerometer, vibration sensor, proximity probe (eddy current).\n"
        "- otro: encoder/tachometer, load cell, PPE, anything that does not fit above.\n"
        "Do not overthink the category: pick the closest family and move on.\n"
        "For each object return an item with:\n"
        '  "tipo": one of (presion|temperatura|caudal|nivel|electrica|analisis|control|vibracion|valvula|epp|otro),\n'
        '  "descripcion": what it is exactly (e.g. "electromagnetic flow meter CONTATEC"),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "lectura": the value shown by the instrument if legible, or null,\n'
        '  "confianza": 0 to 1.\n'
        'Return ONLY: {"objetos": [ ... ]}. '
        "Empty list only if there is NO object of interest."
    ),
}

# --- Industrial: variante v2 = anti-loop (más larga, frena la deliberación) ---
_INDUSTRIAL_V2 = {
    "system": (
        "You are an expert assistant in industrial inspection for mining and "
        "oil & gas. You recognize ANY plant instrument, equipment or object. "
        "Reason in FEW steps and WITHOUT repeating yourself: as soon as you "
        "recognize an object, go straight to the JSON. Do not re-evaluate the "
        "category or revisit what you already thought. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Identify ALL the industrial instruments, equipment or objects visible "
        "in the image. Both measurement devices and plant equipment count "
        "(transformers, valves, motors, etc.). Recognize ANY industrial element, "
        "do not limit yourself to a list.\n\n"
        "IMPORTANT RULE — do not get stuck on the category:\n"
        "  - Identify the object at a glance and pick the closest family. Do NOT debate.\n"
        "  - If you hesitate between two families, pick any one and put the detail in 'descripcion'.\n"
        "  - If it fits none, use 'otro'. Never repeat the same reasoning.\n\n"
        "Families (tipo) with REFERENCE examples (not a closed list):\n"
        "- presion: pressure gauge, pressure transmitter, pressure switch, vacuum gauge.\n"
        "- temperatura: bimetallic thermometer, thermocouple, RTD/Pt100, pyrometer, thermowell.\n"
        "- caudal: electromagnetic/turbine/ultrasonic flow meter, Coriolis, rotameter, orifice plate.\n"
        "- nivel: ultrasonic, radar, hydrostatic transmitter, float, capacitive.\n"
        "- electrica: transformer, insulator/bushing, disconnector, breaker, busbar, switchgear, power analyzer, multimeter, CT, VT.\n"
        "- analisis: O2/CO/CO2 analyzer, pH meter, conductivity meter, turbidimeter, chromatograph.\n"
        "- control: control valve, positioner, actuator, VFD, PID controller.\n"
        "- vibracion: accelerometer, vibration sensor, proximity probe (eddy current).\n"
        "- valvula: manual, ball, gate, butterfly, safety valve.\n"
        "- epp: helmet, gloves, harness, eye/ear protection.\n"
        "- otro: encoder/tachometer, load cell, anything that does not fit above.\n\n"
        "For each object return an item with:\n"
        '  "tipo": one of (presion|temperatura|caudal|nivel|electrica|analisis|control|vibracion|valvula|epp|otro),\n'
        '  "descripcion": what it is exactly (e.g. "high-voltage bushing" or "electromagnetic flow meter CONTATEC"),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "lectura": the value shown by the instrument if legible, or null,\n'
        '  "confianza": 0 to 1.\n'
        'Return ONLY: {"objetos": [ ... ]}. '
        "Empty list only if there is NO object of interest."
    ),
}

# --- Todo: variante única -----------------------------------------------------
_TODO_DEFAULT = {
    "system": (
        "You are a vision assistant that identifies any object in an image. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Identify ALL the objects visible in the image, industrial or not. "
        "For each object return an item with:\n"
        '  "tipo": free one-word category (e.g. "persona", "herramienta", "vehiculo", "manometro"),\n'
        '  "descripcion": what it is (short string),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "lectura": legible text on the object if any, or null,\n'
        '  "confianza": 0 to 1.\n'
        'Return ONLY: {"objetos": [ ... ]}. '
        "Empty list only if the image has NO object at all."
    ),
}

# Registro de variantes por scope. Agregá las que quieras y testealas con 05_prompt_test.py.
PROMPT_VARIANTS = {
    "industrial": {
        "v1_original": _INDUSTRIAL_V1,
        "v2_antiloop": _INDUSTRIAL_V2,
    },
    "todo": {
        "default": _TODO_DEFAULT,
    },
}

# Variante ACTIVA por defecto en cada scope (la que usan menú/smoke/benchmark).
# Volvimos a v1_original en industrial: es más corta y por eso más rápida.
DEFAULT_VARIANT = {
    "industrial": "v1_original",
    "todo": "default",
}


def get_prompt(scope, variant=None):
    """Devuelve el {system, user} de un scope/variante, con fallback al default.

    Si `variant` es None o no existe para ese scope, usa DEFAULT_VARIANT[scope].
    """
    variants = PROMPT_VARIANTS[scope]
    if variant and variant in variants:
        return variants[variant]
    return variants[DEFAULT_VARIANT[scope]]


# SCOPES: vista "armada" que usan los entry points para label + prompt activo.
# Mantiene la interfaz vieja (SCOPES[scope]["system"]/["user"]/["label"]).
SCOPES = {
    scope: {"label": SCOPE_LABELS[scope], **get_prompt(scope)}
    for scope in PROMPT_VARIANTS
}

# --------------------------------------------------------------------------- #
# CONFIG persistente
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "model": "qwen3-vl:4b",            # 4b entra 100% en 8GB de VRAM; el 8b se parte CPU/GPU
    "image": "fotosClean/1.jpeg",      # imagen para el smoke test
    "folder": "fotosClean",            # carpeta para el benchmark
    "scope": "industrial",             # modo de detección (industrial | todo)
    "variant": "v1_original",          # variante de prompt activa (ver PROMPT_VARIANTS); None = default del scope
    "max_tokens": 8192,                # tope de tokens de SALIDA (num_predict; incluye el razonamiento)
    "num_ctx": 16384,                  # ventana de contexto (entrada+salida); la que muestra `ollama ps`
    "think": True,                     # razonamiento del modelo (en qwen3-vl no se puede apagar de verdad; mejor verlo)
    "url": OLLAMA_HOST,                # host de Ollama
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
# Utilidades de imagen
# --------------------------------------------------------------------------- #
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_size(path):
    """Devuelve (ancho, alto) leyendo el header del archivo. None si no se puede.

    Soporta JPEG y PNG (sin dependencias extra). qwen3-vl devuelve los bbox en
    píxeles absolutos del archivo, así que necesitamos las dimensiones para
    normalizarlos a 0..1.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if head == b"\xff\xd8":  # JPEG
                f.seek(0)
                data = f.read()
                i = 2
                while i < len(data) - 9:
                    if data[i] != 0xFF:
                        i += 1
                        continue
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                  0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        h, w = struct.unpack(">HH", data[i + 5:i + 9])
                        return w, h
                    seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                    i += 2 + seg
            elif head == b"\x89P":  # PNG
                f.seek(16)
                w, h = struct.unpack(">II", f.read(8))
                return w, h
    except (OSError, struct.error):
        pass
    return None


def normalize_bboxes(parsed, size):
    """Si los bbox vienen en píxeles (algún valor > 1), los pasa a 0..1 in-place.

    `size` es (ancho, alto). Si es None o el bbox ya está normalizado, no toca nada.
    """
    if not isinstance(parsed, dict) or not size:
        return parsed
    w, h = size
    if not w or not h:
        return parsed
    for obj in parsed.get("objetos", []):
        bb = obj.get("bbox")
        if (isinstance(bb, list) and len(bb) == 4
                and all(isinstance(v, (int, float)) for v in bb)
                and any(v > 1.0 for v in bb)):
            obj["bbox"] = [round(bb[0] / w, 4), round(bb[1] / h, 4),
                           round(bb[2] / w, 4), round(bb[3] / h, 4)]
    return parsed


# --------------------------------------------------------------------------- #
# Parseo de JSON
# --------------------------------------------------------------------------- #
def extract_json(text):
    """Parsea JSON aunque el modelo agregue ruido, fences o razonamiento.

    Devuelve (objeto_o_texto, ok_bool).
    """
    text = (text or "").strip()
    # quitar bloques de razonamiento <think>...</think> por las dudas
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # quitar fences ```json ... ```
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), True
            except json.JSONDecodeError:
                pass
    return text, False


# --------------------------------------------------------------------------- #
# Cliente VLM (endpoint nativo /api/chat, con streaming)
# --------------------------------------------------------------------------- #
def query_vlm(img_b64, model, scope="industrial", max_tokens=8192,
              think=True, url=OLLAMA_HOST, timeout=300, num_ctx=16384,
              verbose=False, size=None, variant=None):
    """Manda una imagen al VLM y devuelve un dict con la respuesta + diagnóstico.

    Usa el endpoint nativo /api/chat con streaming:
      - separa `thinking` (razonamiento) de `content` (la respuesta JSON);
      - si verbose=True, imprime EN VIVO lo que el modelo va pensando;
      - fuerza JSON con format:"json";
      - num_ctx = ventana total (la que ves en `ollama ps`); subirla da más
        resolución a la imagen y más lugar para razonar + responder;
      - max_tokens (num_predict) = tope de tokens de SALIDA (incluye el thinking).

    `size` (ancho, alto) se usa para normalizar los bbox a 0..1.
    Levanta requests.RequestException si falla la red/el server.
    """
    sp = get_prompt(scope, variant)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sp["system"]},
            {"role": "user", "content": sp["user"], "images": [img_b64]},
        ],
        "stream": True,
        "think": think,
        "format": "json",  # fuerza JSON válido en content
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    chat_url = host_of(url) + "/api/chat"

    think_buf, content_buf = [], []
    done_reason = None
    in_tok = out_tok = None
    printed_header = False

    t0 = time.perf_counter()
    with requests.post(chat_url, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message", {})
            piece = msg.get("thinking")
            if piece:
                think_buf.append(piece)
                if verbose:
                    if not printed_header:
                        print("\n\033[2m💭 pensando: ", end="", flush=True)
                        printed_header = True
                    print(piece, end="", flush=True)
            if msg.get("content"):
                content_buf.append(msg["content"])
            if chunk.get("done"):
                done_reason = chunk.get("done_reason")
                in_tok = chunk.get("prompt_eval_count")
                out_tok = chunk.get("eval_count")
    if verbose and printed_header:
        print("\033[0m", flush=True)  # cierra el "dim" y baja de línea
    elapsed = time.perf_counter() - t0

    content = "".join(content_buf)
    reasoning = "".join(think_buf)
    parsed, ok = extract_json(content)
    if ok:
        normalize_bboxes(parsed, size)
    return {
        "elapsed": elapsed,
        "content": content,
        "finish_reason": done_reason,
        "reasoning": reasoning,
        "parsed": parsed,
        "ok": ok,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }


def render_result(model, res):
    """Imprime el resultado de un query_vlm de forma legible (smoke test / menú)."""
    print("\n========== RESULTADO ==========")
    print(f"Modelo:        {model}")
    print(f"Latencia E2E:  {res['elapsed']:.2f} s")
    print(f"finish_reason: {res['finish_reason']}")
    if res.get("in_tokens") is not None:
        print(f"Tokens:        entrada={res['in_tokens']}  "
              f"salida={res['out_tokens']}  (razonamiento≈{len(res['reasoning'])} chars)")
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
                print("    -> Se cortó por longitud: el razonamiento se comió el presupuesto.")
                print("    -> Subí max_tokens (num_predict) y/o num_ctx.")
            if res["reasoning"]:
                print(f"    reasoning (primeros 500 chars):\n{res['reasoning'][:500]}")
        else:
            print("Respuesta cruda:")
            print(res["content"])
    print("===============================")
