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
# MODOS DE DETECCIÓN (scope)
# --------------------------------------------------------------------------- #
# "industrial": instrumentos/equipos de industria (oil & gas, minería). Categoría
#               general + descripción libre, SIN limitarse a una lista cerrada.
# "todo":       CUALQUIER objeto visible (industrial o no), categoría libre.
SCOPES = {
    "industrial": {
        "label": "Instrumentos industriales (oil & gas / minería)",
        "system": (
            "Sos un asistente experto en inspección industrial para minería y "
            "oil & gas. Reconocés CUALQUIER instrumento, equipo u objeto de planta. "
            "Razoná en POCOS pasos y SIN repetirte: en cuanto reconocés un objeto, "
            "pasá directo al JSON. No re-evalúes la categoría ni vuelvas sobre lo "
            "que ya pensaste. "
            "Respondés ÚNICAMENTE con un objeto JSON válido: sin markdown, sin "
            "fences, sin texto antes ni después."
        ),
        "user": (
            "Identificá TODOS los instrumentos, equipos u objetos industriales "
            "visibles en la imagen. Cuentan tanto los aparatos de medición como los "
            "equipos de planta (transformadores, válvulas, motores, etc.). Reconocé "
            "CUALQUIER elemento de industria, no te limites a una lista.\n\n"
            "REGLA IMPORTANTE — no te trabes en la categoría:\n"
            "  - Identificá el objeto de un vistazo y elegí la familia más cercana. NO debatas.\n"
            "  - Si dudás entre dos familias, elegí una cualquiera y poné el detalle en 'descripcion'.\n"
            "  - Si no encaja en ninguna, usá 'otro'. Nunca repitas el mismo razonamiento.\n\n"
            "Familias (tipo) con ejemplos de REFERENCIA (no es lista cerrada):\n"
            "- presion: manómetro, transmisor de presión, presostato, vacuómetro.\n"
            "- temperatura: termómetro bimetálico, termopar, RTD/Pt100, pirómetro, termopozo.\n"
            "- caudal: caudalímetro electromagnético/turbina/ultrasónico, Coriolis, rotámetro, placa orificio.\n"
            "- nivel: ultrasónico, radar, transmisor hidrostático, flotador, capacitivo.\n"
            "- electrica: transformador, aislador/bushing, seccionador, interruptor, barra, celda, analizador de red, multímetro, TC, TP.\n"
            "- analisis: analizador O2/CO/CO2, pHmetro, conductímetro, turbidímetro, cromatógrafo.\n"
            "- control: válvula de control, posicionador, servomotor, VFD, controlador PID.\n"
            "- vibracion: acelerómetro, sensor de vibración, sonda de proximidad (eddy current).\n"
            "- valvula: válvula manual, esférica, compuerta, mariposa, de seguridad.\n"
            "- epp: casco, guantes, arnés, protección visual/auditiva.\n"
            "- otro: encoder/taquímetro, celda de carga, lo que no encaje arriba.\n\n"
            "Para cada objeto devolvé un item con:\n"
            '  "tipo": una de (presion|temperatura|caudal|nivel|electrica|analisis|control|vibracion|valvula|epp|otro),\n'
            '  "descripcion": qué es exactamente (p. ej. "bushing de alta tensión" o "caudalímetro electromagnético CONTATEC"),\n'
            '  "bbox": [x_min, y_min, x_max, y_max] normalizado entre 0 y 1,\n'
            '  "lectura": el valor que muestra el instrumento si es legible, o null,\n'
            '  "confianza": 0 a 1.\n'
            'Devolvé SOLO: {"objetos": [ ... ]}. '
            "Lista vacía únicamente si NO hay ningún objeto de interés."
        ),
    },
    "todo": {
        "label": "Cualquier objeto (uso general)",
        "system": (
            "Sos un asistente de visión que identifica cualquier objeto en una "
            "imagen. Razoná en POCOS pasos y SIN repetirte: en cuanto reconocés un "
            "objeto, pasá directo al JSON. "
            "Respondés ÚNICAMENTE con un objeto JSON válido: sin markdown, "
            "sin fences, sin texto antes ni después."
        ),
        "user": (
            "Identificá TODOS los objetos visibles en la imagen, sean industriales "
            "o no. Para cada objeto devolvé un item con:\n"
            '  "tipo": categoría libre en una palabra (p. ej. "persona", "herramienta", "vehiculo", "manometro"),\n'
            '  "descripcion": qué es (string corto),\n'
            '  "bbox": [x_min, y_min, x_max, y_max] normalizado entre 0 y 1,\n'
            '  "lectura": texto legible en el objeto si lo hay, o null,\n'
            '  "confianza": 0 a 1.\n'
            'Devolvé SOLO: {"objetos": [ ... ]}. '
            "Lista vacía únicamente si la imagen no tiene NINGÚN objeto."
        ),
    },
}

# --------------------------------------------------------------------------- #
# CONFIG persistente
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "model": "qwen3-vl:4b",            # 4b entra 100% en 8GB de VRAM; el 8b se parte CPU/GPU
    "image": "fotosClean/1.jpeg",      # imagen para el smoke test
    "folder": "fotosClean",            # carpeta para el benchmark
    "scope": "industrial",             # modo de detección (industrial | todo)
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
              verbose=False, size=None):
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
    sp = SCOPES[scope]
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
