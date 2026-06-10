#!/usr/bin/env python3
"""
vlm_common.py — Core shared by the menu, the smoke test and the benchmark.

This is the single place (to avoid duplication) that holds:
  - The detection MODES (prompts) -> SCOPES
  - The VLM client against Ollama's NATIVE endpoint (/api/chat)
  - Robust JSON parsing + bounding box normalization
  - The persistent config in config.json

It does not run on its own; it is imported by menu.py / src/smoke_test.py / src/benchmark.py.

Why the native endpoint and not the OpenAI-compatible one?
---------------------------------------------------------------------------
qwen3-vl is a "thinking" model (it reasons before answering). The
OpenAI-compatible endpoint (/v1/chat/completions) IGNORES the `think` flag and,
worse, mixes the reasoning into the answer: if the model spends its entire token
budget thinking, `content` comes back EMPTY and there is no JSON.
The native endpoint (/api/chat) separates the reasoning (field `thinking`) from
the answer (field `content`), so the JSON comes out clean and we can also
print live what the model is thinking.
"""
import base64
import glob
import json
import os
import re
import struct
import time

import requests

# Ollama base host. /api/chat and /api/tags are appended to it.
# Can be overridden via config.json or --url (also accepts old URLs with /v1).
OLLAMA_HOST = "http://localhost:11434"
# Backward compatibility: some scripts/imports use OLLAMA_URL.
OLLAMA_URL = OLLAMA_HOST

# Image extensions the benchmark recognizes when scanning a folder.
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Project root = folder containing src/ (this module lives in src/).
# Everything "project-level" (config.json, results/) hangs off here, not src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Persistent configuration file (in the project root).
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

# Folder where benchmark results are saved (not loose in the root).
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def results_path(name):
    """Return the path to a file inside results/, creating the folder if missing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def host_of(url):
    """Normalize any Ollama URL down to its base host.

    Accepts the bare host or URLs with suffixes (including the old
    /v1/chat/completions) so as not to break existing config.json files.
    """
    if not url:
        return OLLAMA_HOST
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1",
                   "/api/chat", "/api/generate"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


# --------------------------------------------------------------------------- #
# Model capabilities (does it reason or not?)
# --------------------------------------------------------------------------- #
# WATCH OUT with the `think` flag: tested against Ollama 0.30.6 with qwen3-vl and
# qwen2.5vl, reasoning control is NOT a simple per-request on/off, it depends on
# the MODEL:
#   - qwen3-vl:4b / :8b  -> "thinking" capability (renderer "qwen3-vl-thinking").
#                           ALWAYS reasons; sending "think": false does NOT turn it
#                           off in this version (the renderer ignores it).
#   - qwen2.5vl:7b       -> NO "thinking" capability. Never reasons, and sending it
#                           "think": true returns HTTP 400 ("does not support
#                           thinking"). That is why we must NOT send the flag if the
#                           model does not support it (otherwise the benchmark hits 100% errors).
# In other words: the real reasoning switch is CHOOSING THE MODEL. Here we detect
# the capability via /api/show to (a) not break models without thinking and
# (b) tell the user the truth in the UI.
_CAPS_CACHE = {}


def model_capabilities(model, url=OLLAMA_HOST):
    """Return the model's capability set (via /api/show). Cached by (host, model).

    If it cannot be queried (server down, nonexistent model), returns an empty set.
    """
    key = (host_of(url), model)
    if key in _CAPS_CACHE:
        return _CAPS_CACHE[key]
    caps = set()
    try:
        r = requests.post(host_of(url) + "/api/show", json={"model": model}, timeout=10)
        r.raise_for_status()
        caps = set(r.json().get("capabilities") or [])
    except (requests.RequestException, ValueError):
        caps = set()
    _CAPS_CACHE[key] = caps
    return caps


def model_supports_thinking(model, url=OLLAMA_HOST):
    """True if the model declares the 'thinking' capability (can reason)."""
    return "thinking" in model_capabilities(model, url)


# --------------------------------------------------------------------------- #
# DETECTION MODES (scope) + PROMPT VARIANTS (swappable / A-B test)
# --------------------------------------------------------------------------- #
# There are two scopes:
#   "industrial": industrial instruments/equipment (oil & gas, mining).
#   "all":        ANY visible object (industrial or not), free category.
#
# Each scope has prompt VARIANTS in PROMPT_VARIANTS, so you can compare which is
# faster/better (with the benchmark, --variants). Everything (prompts and the JSON
# contract keys/values) is in English.
#
# The ACTIVE variant by default (the one menu/smoke/benchmark use) is set by
# DEFAULT_VARIANT. To "swap" the prompt, change DEFAULT_VARIANT here, or pass
# --variant in the scripts, or choose it in the menu.
SCOPE_LABELS = {
    "industrial": "Industrial instruments (oil & gas / mining)",
    "all": "Any object (general use)",
}

# --- Industrial: variant v1 = the ORIGINAL prompt (short, faster) ------------
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
        "- pressure: pressure gauge, pressure transmitter, pressure switch, vacuum gauge.\n"
        "- temperature: bimetallic thermometer, thermocouple, RTD/Pt100, pyrometer, thermowell.\n"
        "- flow: electromagnetic/turbine/ultrasonic flow meter, Coriolis, rotameter, orifice plate.\n"
        "- level: ultrasonic, radar, hydrostatic transmitter, float, capacitive.\n"
        "- electrical: power analyzer, wattmeter, multimeter, CT, VT.\n"
        "- analysis: O2/CO/CO2 analyzer, pH meter, conductivity meter, turbidimeter, chromatograph.\n"
        "- control: control valve, positioner, actuator, VFD, PID controller.\n"
        "- vibration: accelerometer, vibration sensor, proximity probe (eddy current).\n"
        "- other: encoder/tachometer, load cell, PPE, anything that does not fit above.\n"
        "Do not overthink the category: pick the closest family and move on.\n"
        "For each object return an item with:\n"
        '  "type": one of (pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other),\n'
        '  "description": what it is exactly (e.g. "electromagnetic flow meter CONTATEC"),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "reading": the value shown by the instrument if legible, or null,\n'
        '  "confidence": 0 to 1.\n'
        'Return ONLY: {"objects": [ ... ]}. '
        "Empty list only if there is NO object of interest."
    ),
}

# --- Industrial: variant v2 = anti-loop (longer, curbs deliberation) ---------
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
        "  - If you hesitate between two families, pick any one and put the detail in 'description'.\n"
        "  - If it fits none, use 'other'. Never repeat the same reasoning.\n\n"
        "Families (type) with REFERENCE examples (not a closed list):\n"
        "- pressure: pressure gauge, pressure transmitter, pressure switch, vacuum gauge.\n"
        "- temperature: bimetallic thermometer, thermocouple, RTD/Pt100, pyrometer, thermowell.\n"
        "- flow: electromagnetic/turbine/ultrasonic flow meter, Coriolis, rotameter, orifice plate.\n"
        "- level: ultrasonic, radar, hydrostatic transmitter, float, capacitive.\n"
        "- electrical: transformer, insulator/bushing, disconnector, breaker, busbar, switchgear, power analyzer, multimeter, CT, VT.\n"
        "- analysis: O2/CO/CO2 analyzer, pH meter, conductivity meter, turbidimeter, chromatograph.\n"
        "- control: control valve, positioner, actuator, VFD, PID controller.\n"
        "- vibration: accelerometer, vibration sensor, proximity probe (eddy current).\n"
        "- valve: manual, ball, gate, butterfly, safety valve.\n"
        "- ppe: helmet, gloves, harness, eye/ear protection.\n"
        "- other: encoder/tachometer, load cell, anything that does not fit above.\n\n"
        "For each object return an item with:\n"
        '  "type": one of (pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other),\n'
        '  "description": what it is exactly (e.g. "high-voltage bushing" or "electromagnetic flow meter CONTATEC"),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "reading": the value shown by the instrument if legible, or null,\n'
        '  "confidence": 0 to 1.\n'
        'Return ONLY: {"objects": [ ... ]}. '
        "Empty list only if there is NO object of interest."
    ),
}

# --- All: single variant ------------------------------------------------------
_ALL_DEFAULT = {
    "system": (
        "You are a vision assistant that identifies any object in an image. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Identify ALL the objects visible in the image, industrial or not. "
        "For each object return an item with:\n"
        '  "type": free one-word category (e.g. "person", "tool", "vehicle", "gauge"),\n'
        '  "description": what it is (short string),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "reading": legible text on the object if any, or null,\n'
        '  "confidence": 0 to 1.\n'
        'Return ONLY: {"objects": [ ... ]}. '
        "Empty list only if the image has NO object at all."
    ),
}

# Variant registry by scope. Add as many as you want and compare with
# `python3 src/benchmark.py --variants <a> <b>` (or from the benchmark submenu).
PROMPT_VARIANTS = {
    "industrial": {
        "v1_original": _INDUSTRIAL_V1,
        "v2_antiloop": _INDUSTRIAL_V2,
    },
    "all": {
        "default": _ALL_DEFAULT,
    },
}

# ACTIVE variant by default for each scope (the one menu/smoke/benchmark use).
# We reverted to v1_original for industrial: it is shorter and therefore faster.
DEFAULT_VARIANT = {
    "industrial": "v1_original",
    "all": "default",
}


def get_prompt(scope, variant=None):
    """Return the {system, user} for a scope/variant, falling back to the default.

    If `variant` is None or does not exist for that scope, use DEFAULT_VARIANT[scope].
    """
    variants = PROMPT_VARIANTS[scope]
    if variant and variant in variants:
        return variants[variant]
    return variants[DEFAULT_VARIANT[scope]]


# SCOPES: the "assembled" view the entry points use for label + active prompt.
# Keeps the old interface (SCOPES[scope]["system"]/["user"]/["label"]).
SCOPES = {
    scope: {"label": SCOPE_LABELS[scope], **get_prompt(scope)}
    for scope in PROMPT_VARIANTS
}

# --------------------------------------------------------------------------- #
# Persistent CONFIG
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "model": "qwen3-vl:4b",            # 4b fits 100% in 8GB of VRAM; the 8b splits CPU/GPU
    "image": "fotos/clean/1.jpeg",     # image for the smoke test
    "folder": "fotos/clean",           # folder for the benchmark
    "scope": "industrial",             # detection mode (industrial | all)
    "variant": "v1_original",          # active prompt variant (see PROMPT_VARIANTS); None = scope default
    "max_tokens": 8192,                # OUTPUT token ceiling (num_predict; includes the reasoning)
    "num_ctx": 16384,                  # context window (input+output); the one `ollama ps` shows
    "think": True,                     # model reasoning (in qwen3-vl it cannot really be turned off; better to watch it)
    "url": OLLAMA_HOST,                # Ollama host

    # --- Benchmark: has its OWN config, independent of the smoke test ---------
    # This way you can run the benchmark with a lighter (faster) context without
    # lowering the smoke test's context. All of this is edited from the benchmark
    # submenu (option 2 of the main menu).
    #
    # The benchmark sweeps the MODELS × VARIANTS product: just as you compare
    # several models, you now compare several prompt variants in the same run
    # (this replaces the old 05_prompt_test.py).
    #
    # max_tokens/num_ctx/think are LISTS: the benchmark compares ALL their values
    # (just like models/prompts/photos). Put a single value to not sweep them.
    "benchmark_models": ["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"],
    "benchmark_runs": 3,               # repetitions per image
    "benchmark_images": [],            # hand-picked images ([] = ALL in the folder)
    "benchmark_scope": "industrial",   # benchmark detection mode
    "benchmark_variants": ["v2_antiloop"],  # prompt variants to compare (see PROMPT_VARIANTS)
    "benchmark_max_tokens": [4096],    # OUTPUT ceilings to compare (list)
    "benchmark_num_ctx": [8192],       # context windows to compare (list)
    "benchmark_think": [True],         # reasoning values to compare (list of bools)
}


def load_config():
    """Return the config merged with the defaults (creates the file if missing)."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt config -> fall back to defaults
    else:
        save_config(cfg)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Image utilities
# --------------------------------------------------------------------------- #
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_size(path):
    """Return (width, height) by reading the file header. None if it cannot.

    Supports JPEG and PNG (no extra dependencies). qwen3-vl returns the bbox in
    absolute pixels of the file, so we need the dimensions to normalize them
    to 0..1.
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
    """If the bbox come in pixels (any value > 1), convert them to 0..1 in-place.

    `size` is (width, height). If it is None or the bbox is already normalized, leaves it alone.
    """
    if not isinstance(parsed, dict) or not size:
        return parsed
    w, h = size
    if not w or not h:
        return parsed
    for obj in parsed.get("objects", []):
        bb = obj.get("bbox")
        if (isinstance(bb, list) and len(bb) == 4
                and all(isinstance(v, (int, float)) for v in bb)
                and any(v > 1.0 for v in bb)):
            obj["bbox"] = [round(bb[0] / w, 4), round(bb[1] / h, 4),
                           round(bb[2] / w, 4), round(bb[3] / h, 4)]
    return parsed


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #
def extract_json(text):
    """Parse JSON even if the model adds noise, fences or reasoning.

    Returns (object_or_text, ok_bool).
    """
    text = (text or "").strip()
    # strip <think>...</think> reasoning blocks just in case
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # strip ```json ... ``` fences
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
# VLM client (native endpoint /api/chat, with streaming)
# --------------------------------------------------------------------------- #
def query_vlm(img_b64, model, scope="industrial", max_tokens=8192,
              think=True, url=OLLAMA_HOST, timeout=300, num_ctx=16384,
              verbose=False, size=None, variant=None):
    """Send an image to the VLM and return a dict with the response + diagnostics.

    Uses the native /api/chat endpoint with streaming:
      - separates `thinking` (reasoning) from `content` (the JSON answer);
      - if verbose=True, prints LIVE what the model is thinking;
      - forces JSON with format:"json";
      - num_ctx = total window (the one you see in `ollama ps`); raising it gives
        the image more resolution and more room to reason + answer;
      - max_tokens (num_predict) = OUTPUT token ceiling (includes the thinking).

    `size` (width, height) is used to normalize the bbox to 0..1.

    The `think` flag is ONLY sent if the model supports reasoning (capability
    'thinking'); sending the flag to a model that lacks it (e.g. qwen2.5vl)
    returns HTTP 400, so we omit it. See model_supports_thinking().
    Raises requests.RequestException if the network/server fails.
    """
    sp = get_prompt(scope, variant)
    supports_think = model_supports_thinking(model, url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sp["system"]},
            {"role": "user", "content": sp["user"], "images": [img_b64]},
        ],
        "stream": True,
        "format": "json",  # forces valid JSON in content
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    # Only include `think` if the model supports it (otherwise Ollama returns 400).
    if supports_think:
        payload["think"] = think
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
                        print("\n\033[2m💭 thinking: ", end="", flush=True)
                        printed_header = True
                    print(piece, end="", flush=True)
            if msg.get("content"):
                content_buf.append(msg["content"])
            if chunk.get("done"):
                done_reason = chunk.get("done_reason")
                in_tok = chunk.get("prompt_eval_count")
                out_tok = chunk.get("eval_count")
    if verbose and printed_header:
        print("\033[0m", flush=True)  # closes the "dim" and moves to a new line
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
        "think_requested": think,          # what the user requested
        "thinking_supported": supports_think,  # whether the model can reason
        "did_think": bool(reasoning),      # whether it actually reasoned
    }


# --------------------------------------------------------------------------- #
# Image listing and UI helpers (progress bar / timings)
# --------------------------------------------------------------------------- #
def natural_key(path):
    """Sort 1,2,...,10 instead of 1,10,2 ('natural' order) by file name."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return (int(nums[0]) if nums else 0, os.path.basename(path))


def list_images(folder):
    """List all supported images in a folder, in natural order."""
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(imgs, key=natural_key)


def fmt_secs(s):
    """Format seconds as '12.3s' or '2m05s' for easy reading."""
    if s != s:  # NaN
        return "  -  "
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}m{sec:02d}s"


def progress_bar(done, total, suffix="", width=24):
    """Draw/update a progress bar on a single line (carriage return).

    When done >= total it moves to a new line so it does not overwrite what comes next.
    No dependencies: just block characters and \\r.
    """
    total = max(total, 1)
    frac = min(done / total, 1.0)
    filled = int(round(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    line = f"\r  [{bar}] {frac * 100:5.1f}% ({done}/{total})"
    if suffix:
        line += f" | {suffix}"
    # Pad to cover leftovers from a longer previous line.
    print(line.ljust(100)[:120], end="", flush=True)
    if done >= total:
        print()


def describe_thinking(res):
    """Summarize IN ONE LINE the truth about this run's reasoning.

    The `think` flag is not a reliable on/off (it depends on the model and the
    Ollama version), so instead of showing what was REQUESTED we show what
    HAPPENED: whether the model can reason, whether it was asked to, and whether
    it actually reasoned.
    """
    requested = res.get("think_requested")
    supported = res.get("thinking_supported")
    did = res.get("did_think")
    if not supported:
        return "this model does NOT reason (no 'thinking' capability) — truly OFF, the flag does not apply"
    if did and not requested:
        return ("you requested OFF but it reasoned anyway — this model/Ollama version "
                "ignores think=false (to avoid reasoning, use a model without 'thinking')")
    if did and requested:
        return "ON (reasoned)"
    if not did and requested:
        return "ON requested, but it did not reason on this run"
    return "OFF (did not reason)"


def render_result(model, res):
    """Print a query_vlm result in a readable way (smoke test / menu)."""
    print("\n========== RESULT ==========")
    print(f"Model:         {model}")
    print(f"E2E latency:   {res['elapsed']:.2f} s")
    print(f"finish_reason: {res['finish_reason']}")
    if res.get("in_tokens") is not None:
        print(f"Tokens:        input={res['in_tokens']}  "
              f"output={res['out_tokens']}  (reasoning≈{len(res['reasoning'])} chars)")
    print(f"Reasoning:     {describe_thinking(res)}")
    print(f"Valid JSON:    {'YES' if res['ok'] else 'NO'}")
    print("-------------------------------")
    if res["ok"]:
        parsed = res["parsed"]
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        n = len(parsed.get("objects", [])) if isinstance(parsed, dict) else 0
        print(f"\n[OK] Objects detected: {n}")
    else:
        print("[!] Did not return parseable JSON.")
        if not res["content"].strip():
            print("[!] content came back EMPTY.")
            if res["finish_reason"] == "length":
                print("    -> Cut off by length: the reasoning ate the budget.")
                print("    -> Raise max_tokens (num_predict) and/or num_ctx.")
            if res["reasoning"]:
                print(f"    reasoning (first 500 chars):\n{res['reasoning'][:500]}")
        else:
            print("Raw response:")
            print(res["content"])
    print("===============================")
