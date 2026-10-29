#!/usr/bin/env python3
"""
vlm_common.py — Core shared by the menu, the smoke test and the benchmark.

This is the single place (to avoid duplication) that holds:
  - The detection MODES (prompts) -> SCOPES
  - The VLM client: Ollama's NATIVE endpoint (/api/chat), or llama-server (see below)
  - Robust JSON parsing + bounding box normalization
  - The persistent config in config.json

It does not run on its own; it is imported by menu.py / src/vlm_smoke_test.py / src/vlm_benchmark.py (and indirectly by the yolo_* scripts via shared helpers).

Why the native endpoint and not the OpenAI-compatible one?
---------------------------------------------------------------------------
qwen3-vl is a "thinking" model (it reasons before answering). The
OpenAI-compatible endpoint (/v1/chat/completions) IGNORES the `think` flag and,
worse, mixes the reasoning into the answer: if the model spends its entire token
budget thinking, `content` comes back EMPTY and there is no JSON.
The native endpoint (/api/chat) separates the reasoning (field `thinking`) from
the answer (field `content`), so the JSON comes out clean and we can also
print live what the model is thinking.

The llama-server backend (vlm_backend = "llamacpp", since 2026-10-16)
---------------------------------------------------------------------------
Ollama 0.30.6 runs the vision encoder of qwen3-vl ON THE CPU: it starts its
llama-server with --no-mmproj-offload and --image-min-tokens 1024, whatever the
model or the free VRAM ("CLIP using CPU backend" in its log). Measured on the RTX
5060: 6.4 s just to look at a 768 px photo, GPU at 1-2 % meanwhile — most of the
~10 s a spoken question took. The same model under llama-server with the encoder on
the GPU: 0.5 s in total. So the VLM can be served by a llama-server that
linux/run.sh starts with the app; set vlm_backend to "llamacpp" (config.json or the
VLM_BACKEND env) and point llamacpp_url at it.

That server speaks the OpenAI-compatible /v1/chat/completions. The caveat above
does not bite there: llama-server returns any reasoning in its own
`reasoning_content` field, separate from `content`, and the models we serve with it
are Instruct checkpoints that do not reason anyway. The callers keep building
Ollama-style payloads; _openai_payload() translates them, so every entry point (the
detection scan, the free-prompt stream, the command interpreter) works on both.
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

# llama-server, when vlm_backend is "llamacpp" (see the module docstring).
LLAMACPP_HOST = "http://127.0.0.1:8081"


def vlm_backend():
    """Which server answers VLM requests: "ollama" (default) or "llamacpp".

    VLM_BACKEND in the environment wins over config.json's vlm_backend, so the
    launcher can fall back to Ollama when it could not start llama-server.
    """
    b = os.environ.get("VLM_BACKEND") or load_config().get("vlm_backend") or "ollama"
    return b.strip().lower()

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
# Benchmark helpers (shared by vlm_benchmark and yolo_benchmark)
# --------------------------------------------------------------------------- #
def pctl(values, p):
    """Linear-interpolated percentile p (0..100). NaN on an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def dedup(seq):
    """Remove duplicates while preserving order of appearance."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def as_list(v, fallback):
    """Normalize a value (scalar/list/None) to a non-empty, deduplicated list.

    Lets the sweep dimensions accept either a single value or a list; if it ends
    up empty, use `fallback` (a list).
    """
    if v is None:
        items = []
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [v]
    items = dedup(items)
    return items or list(fallback)


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
    if vlm_backend() == "llamacpp":
        return set()               # no /api/show there; we serve Instruct checkpoints
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

# --- Industrial: variant v3 = minimal (fewest input tokens -> faster prefill) -
# Same JSON contract, but as terse as possible: shorter prompt = fewer input
# tokens = faster prefill, which dominates latency on a non-reasoning model like
# qwen3-vl:4b-instruct. The family list is collapsed to a single enum line.
_INDUSTRIAL_V3 = {
    "system": (
        "You are an industrial inspection assistant (mining, oil & gas). "
        "Detect any plant instrument, equipment or object. "
        "Respond with ONLY a valid JSON object: no markdown, no extra text."
    ),
    "user": (
        "Detect every industrial instrument, equipment or object in the image. "
        'Return ONLY {"objects": [ ... ]}, where each item has: '
        '"type" (one of pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other), '
        '"description" (short), '
        '"bbox" ([x_min, y_min, x_max, y_max] normalized 0-1), '
        '"reading" (the instrument value if legible, else null), '
        '"confidence" (0-1). '
        "Empty list only if there is nothing of interest."
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

# --- All: variant = people only -----------------------------------------------
# Detect ONLY people. Same JSON contract; "type" is always "person".
_ALL_PEOPLE = {
    "system": (
        "You are a vision assistant that detects people in an image. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Detect ALL the people (humans) visible in the image. Count partially "
        "occluded people and people in the background; ignore everything that is "
        "not a person.\n"
        "For each person return an item with:\n"
        '  "type": always "person",\n'
        '  "description": short distinguishing detail (e.g. "worker with hard hat", '
        '"person sitting", "child"), or "person" if none stands out,\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "reading": text legible on the person (e.g. badge/jersey number) or null,\n'
        '  "confidence": 0 to 1.\n'
        'Return ONLY: {"objects": [ ... ]}. '
        "Empty list only if there is NO person in the image."
    ),
}

# --- All: variant = graspable objects (humanoid robot) ------------------------
# Detect ONLY small, light objects a humanoid robot hand could pick up.
_ALL_GRASPABLE = {
    "system": (
        "You are a vision assistant for a humanoid robot. You detect objects the "
        "robot could pick up and carry with one hand. "
        "You respond ONLY with a valid JSON object: no markdown, no fences, "
        "no text before or after."
    ),
    "user": (
        "Detect ALL the objects in the image that a humanoid robot hand could "
        "GRASP and lift: small and light enough to be picked up with one hand "
        "(roughly hand-sized, up to ~2 kg).\n"
        "INCLUDE: cups, mugs, bottles, cans, tools (wrench, screwdriver, hammer), "
        "small boxes, books, phones, remotes, pens, fruit, hand tools, small parts.\n"
        "EXCLUDE: anything too big, heavy or fixed to grasp — furniture, "
        "appliances, vehicles, walls, doors, large machinery, people, animals, "
        "and the floor/ground.\n"
        "For each graspable object return an item with:\n"
        '  "type": free one-word category (e.g. "cup", "bottle", "tool", "box"),\n'
        '  "description": what it is (short string),\n'
        '  "bbox": [x_min, y_min, x_max, y_max] normalized between 0 and 1,\n'
        '  "reading": legible text on the object if any, or null,\n'
        '  "confidence": 0 to 1.\n'
        'Return ONLY: {"objects": [ ... ]}. '
        "Empty list only if there is NO graspable object."
    ),
}

# Variant registry by scope. Add as many as you want and compare with
# `python3 src/vlm_benchmark.py --variants <a> <b>` (or from the benchmark submenu).
PROMPT_VARIANTS = {
    "industrial": {
        "v1_original": _INDUSTRIAL_V1,
        "v2_antiloop": _INDUSTRIAL_V2,
        "v3_minimal": _INDUSTRIAL_V3,
    },
    "all": {
        "default": _ALL_DEFAULT,
        "people": _ALL_PEOPLE,
        "graspable": _ALL_GRASPABLE,
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
    "vlm_backend": "ollama",           # "ollama" or "llamacpp" (llama-server, vision on the GPU)
    "llamacpp_url": LLAMACPP_HOST,     # where linux/run.sh starts llama-server
    # The model linux/vlm-server.sh serves (paths relative to the repo; linux/install.sh
    # downloads them) and how many tokens a photo may shrink to. Picked by
    # src/vlm_server_benchmark.py; 1024 is Ollama's floor, which reads small print that
    # the model's own (~430 tokens for a 768 px frame) misses.
    "llamacpp_model": "vlm_models/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "llamacpp_mmproj": "vlm_models/mmproj-Qwen3VL-4B-Instruct-F16.gguf",
    "llamacpp_alias": "qwen3vl-4b-q4",
    "llamacpp_repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",   # where install.sh gets both files
    "llamacpp_image_min_tokens": 1024,
    # Who the spoken assistant is, by robot id, appended to every free question (persona_for).
    "assistant_robot": "g1",
    # Each robot: an identity that always goes, and facts that go only when the question
    # mentions one of their keywords (persona_for). Keywords are matched without accents.
    "assistant_persona": {
        "g1": {
            # "not the cartoon character": the model knows Futurama's Bender, and the name alone
            # brought him in ("¡Ay, qué mierda!", "¡Ay, qué miedo!"). This form answered calm
            # and short and kept 17.84 kW 6/6; adding "Unitree G1" here lost the name ("Soy
            # Unitree G1") and read 17.84 right 2/6 — so the model is a fact, asked for.
            "identity": "You are Bender, a calm and helpful robot, not the cartoon character. "
                        "The image is what your own camera sees in front of you.",
            "facts": [
                # "¿Quién sos?" got "Soy un asistente de inteligencia artificial" with the name
                # in the identity (2026-10-28): asked for, it is said again as a fact.
                {"about": ["quien sos", "quien eres", "quien es usted", "como te llamas",
                           "tu nombre", "llamas", "who are you", "your name"],
                 "fact": "your name is Bender"},
                {"about": ["que sos", "que eres", "que robot", "modelo", "unitree", "g1",
                           "humanoide", "what are you"],
                 "fact": "you are a Unitree G1 humanoid robot"},
                {"about": ["hora", "dia", "fecha", "hoy", "time", "date", "today"],
                 "fact": "right now it is {now}"},
                # "made by Maxi" in the identity came back as "Maxi, la empresa que fabrica
                # los robots Unitree": who Maxi is has to be said, and only when asked.
                {"about": ["cre", "hizo", "fabric", "program", "disen", "construy", "dueno",
                           "papa", "padre", "maxi", "made", "built", "creator"],
                 "fact": "your creator is Maxi, an engineer at Silk Technologies who "
                         "designed and programmed you"},
                {"about": ["trabaj", "empresa", "silk", "work"],
                 "fact": "you work at Silk Technologies"},
                {"about": ["amigo", "amiga", "friend", "mateo"],
                 "fact": "your best friend is Mateo"},
                {"about": ["hermano", "hermana", "familia", "perro", "huesos", "go2",
                           "brother", "family"],
                 "fact": "your brother is Huesos, a Unitree Go2 robot dog"},
                {"about": ["medis", "mides", "altura", "alto", "mide", "tall", "height"],
                 "fact": "you are 130 cm tall"},
            ],
        },
        "go2": {
            "identity": "You are Huesos, a calm and helpful robot dog. "
                        "The image is what your own camera sees in front of you.",
            "facts": [
                # "¿Quién sos?" got "Soy un asistente de inteligencia artificial" with the name
                # in the identity (2026-10-28): asked for, it is said again as a fact.
                {"about": ["quien sos", "quien eres", "quien es usted", "como te llamas",
                           "tu nombre", "llamas", "who are you", "your name"],
                 "fact": "your name is Huesos"},
                {"about": ["que sos", "que eres", "que robot", "modelo", "unitree", "go2",
                           "what are you"],
                 "fact": "you are a Unitree Go2 robot dog"},
                {"about": ["hora", "dia", "fecha", "hoy", "time", "date", "today"],
                 "fact": "right now it is {now}"},
                {"about": ["cre", "hizo", "fabric", "program", "disen", "construy", "dueno",
                           "papa", "padre", "maxi", "made", "built", "creator"],
                 "fact": "your creator is Maxi, an engineer at Silk Technologies who "
                         "programmed you"},
                {"about": ["trabaj", "empresa", "silk", "work"],
                 "fact": "you work at Silk Technologies"},
                {"about": ["hermano", "hermana", "familia", "bender", "g1", "brother",
                           "family"],
                 "fact": "your brother is Bender, a Unitree G1 humanoid robot"},
            ],
        },
    },

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
    """Return (width, height). None if it cannot be determined.

    Fast path reads the JPEG/PNG header with no extra dependencies; for the other
    formats in IMG_EXTS (bmp/webp) — or any header we fail to parse — it falls back
    to PIL. qwen3-vl returns the bbox in absolute pixels of the file, so we need the
    dimensions to normalize them to 0..1 (a missing size leaves bboxes un-normalized).
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
    # Fallback for bmp/webp and any header we couldn't parse. PIL is already a
    # dependency (used by service.py); lazy-import so the fast path needs nothing.
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
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
    objects = parsed.get("objects")
    # Valid JSON is not the contract: a model has answered "objects": ["person", ...]
    # (qwen3-vl 4B Q8, 2026-10-16), which crashed here with a 500 on /vlm.
    for obj in objects if isinstance(objects, list) else []:
        if not isinstance(obj, dict):
            continue
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
# llama-server (OpenAI-compatible) translation — see the module docstring
# --------------------------------------------------------------------------- #
# JSON with at most one space between tokens: no newlines, no indentation. A plain
# json_object response format let the model pretty-print, and the indentation cost so
# many tokens that a 30-object detection Ollama returned in 1652 tokens did not fit in
# 2000 (2026-10-16). Ollama's own format:"json" is compact the same way.
_COMPACT_JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ( string ":" ws value ( "," ws string ":" ws value )* )? "}"
array  ::= "[" ( value ( "," ws value )* )? "]"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
ws     ::= " "?
"""


def _openai_payload(payload):
    """An Ollama /api/chat payload as a llama-server /v1/chat/completions one."""
    messages = []
    for m in payload.get("messages", []):
        images = m.get("images") or []
        if images:
            content = [{"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64}} for b64 in images]
            content.append({"type": "text", "text": m.get("content", "")})
        else:
            content = m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    opts = payload.get("options", {})
    out = {
        "model": payload.get("model", ""),   # informational: llama-server serves one model
        "messages": messages,
        "stream": True,
        # The last chunk then says how many tokens the question took: the conversation meter.
        "stream_options": {"include_usage": True},
        "temperature": opts.get("temperature", 0.1),
        "max_tokens": opts.get("num_predict", 512),
        "cache_prompt": True,
        # Ollama's sampling defaults, which llama-server does not apply by itself. Without
        # the repeat penalty a long object list looped until max_tokens (2026-10-16): the
        # same detection Ollama finished in 1652 tokens ran past 2000 here.
        "repeat_penalty": opts.get("repeat_penalty", 1.1),
        "top_k": opts.get("top_k", 40),
        "top_p": opts.get("top_p", 0.9),
    }
    if payload.get("format") == "json":
        out["grammar"] = _COMPACT_JSON_GBNF
    return out


def _openai_chunks(payload, url, timeout):
    """POST to llama-server and yield each streamed chunk (a dict) until [DONE]."""
    with requests.post(host_of(url) + "/v1/chat/completions", json=_openai_payload(payload),
                       stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _stream_chat_llamacpp(payload, url, timeout):
    """stream_chat's contract, served by llama-server."""
    content, reasoning = [], []
    done_reason = in_tok = out_tok = None
    for chunk in _openai_chunks(payload, url, timeout):
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if choice.get("finish_reason"):
            done_reason = choice["finish_reason"]
        timings = chunk.get("timings") or {}
        usage = chunk.get("usage") or {}
        in_tok = usage.get("prompt_tokens", timings.get("prompt_n", in_tok))
        out_tok = usage.get("completion_tokens", timings.get("predicted_n", out_tok))
    return "".join(content), "".join(reasoning), done_reason, in_tok, out_tok


_WINDOW = {}


def llamacpp_window(url=LLAMACPP_HOST):
    """The llama-server's context window in tokens (its -c), or None when it does not say.
    Asked once per server: it is fixed when the server starts."""
    base = host_of(url)
    if base not in _WINDOW:
        try:
            r = requests.get(base + "/props", timeout=3)
            r.raise_for_status()
            _WINDOW[base] = int(r.json()["default_generation_settings"]["n_ctx"])
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return None
    return _WINDOW[base]


# The last chunk of a streamed answer: how much of the window it took (service /vlm/stream).
# Record separators, like search_common.SEARCHING_MARK; the page reads and drops it.
USAGE_MARK = "\x1eusage "


def usage_mark(usage):
    return f"{USAGE_MARK}{json.dumps(usage, separators=(',', ':'))}\x1e"


def list_llamacpp_models(url=LLAMACPP_HOST):
    """The model(s) a llama-server serves (its --alias), or [] when it is down."""
    try:
        r = requests.get(host_of(url) + "/v1/models", timeout=3)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except (requests.RequestException, ValueError, KeyError):
        return []


# --------------------------------------------------------------------------- #
# VLM client (native endpoint /api/chat, with streaming)
# --------------------------------------------------------------------------- #
def stream_chat(payload, url=OLLAMA_HOST, timeout=300, verbose=False):
    """POST an Ollama /api/chat request (stream=True) and consume the stream.

    Returns (content, reasoning, done_reason, in_tokens, out_tokens):
      - content:  the answer text (the JSON, when format:"json" was requested);
      - reasoning: the `thinking` field, concatenated (empty if the model did not
        reason or does not support it);
      - done_reason / in_tokens / out_tokens: taken from the final chunk.
    With verbose=True it prints the reasoning live (dim) as it streams.

    Shared by query_vlm (detection) and the command interpreter
    (command_common.interpret) so the streaming/parse loop lives in ONE place.
    Raises requests.RequestException if the network/server fails.
    """
    if vlm_backend() == "llamacpp":
        return _stream_chat_llamacpp(payload, url, timeout)
    think_buf, content_buf = [], []
    done_reason = in_tok = out_tok = None
    printed_header = False
    chat_url = host_of(url) + "/api/chat"

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

    return "".join(content_buf), "".join(think_buf), done_reason, in_tok, out_tok


# The instruction for free questions (the spoken assistant, "Ask anything"). Plain on purpose:
# rewording it has broken reading before — "Read brands exactly…" made a nameplate read
# "Fluke" 3 times out of 3 (2026-10-16). The robot's identity is APPENDED to it, never
# merged in (see persona_for).
FREE_PROMPT_SYSTEM = ("You are a helpful visual assistant. Answer the user's question about "
                      "the image clearly and concisely, in plain text.")


def _plain(text):
    """Lower case without accents, for matching a question against fact keywords."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", (text or "").lower())
                   if unicodedata.category(c) != "Mn")


_DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"]


def _now_text(now=None, english=False):
    """This machine's local time, in words the answer can repeat: "13:19 on lunes 19 de
    octubre de 2026". The model has no clock; asked the time, it now says "Es 13:19, lunes 19
    de octubre de 2026" (2026-10-19). In English for a question in English: it answered "What
    time is it?" with "14:23 on miércoles 28 de octubre" (2026-10-28)."""
    import calendar  # English names whatever the locale: it is never set in this process
    import datetime
    now = now or datetime.datetime.now()
    if english:
        return (f"{now:%H:%M} on {calendar.day_name[now.weekday()]} {now.day} "
                f"{calendar.month_name[now.month]} {now.year}")
    return f"{now:%H:%M} on {_DAYS[now.weekday()]} {now.day} de {_MONTHS[now.month - 1]} de {now.year}"


# Function words, the most frequent of each language: enough to tell which one a question is in.
_ES_WORDS = {"que", "el", "la", "los", "las", "de", "es", "un", "una", "en", "y", "hay", "por",
             "como", "cual", "cuanto", "donde", "quien", "sos", "eres", "estas", "hoy"}
_EN_WORDS = {"what", "the", "is", "are", "a", "an", "it", "of", "in", "and", "there", "how",
             "which", "where", "who", "you", "your", "today", "this", "do", "does"}


def asked_in_english(question):
    """Whether a question is in English rather than Spanish (Spanish when it cannot tell)."""
    if re.search(r"[¿¡ñáéíóú]", question or "", re.IGNORECASE):
        return False
    words = re.findall(r"[a-z']+", (question or "").lower())
    return sum(w in _EN_WORDS for w in words) > sum(w in _ES_WORDS for w in words)


# What the robot knows about itself (the gateway sends it: robot_watch.state), said only when
# the question asks about it — like the persona's facts, by what it asks, in both languages.
# Only questions about ITSELF: "¿qué temperatura marca el termómetro?" is a reading, so the
# temperature goes for "¿estás caliente?", never for "temperatura" alone.
STATE_ASKS = {
    "battery": ["bateria", "carga", "cargando", "cargado", "enchufado", "energia", "battery",
                "charg"],
    "heat": ["estas caliente", "te calent", "tenes calor", "tienes calor", "tu temperatura",
             "are you hot", "your temperature", "overheat"],
    "link": ["conexion", "conectado", "senal", "latencia", "wifi", "connection", "connected",
             "signal", "latency"],
    "status": ["como estas", "como andas", "como te sentis", "como te sientes", "tu estado",
               "how are you", "your status"],
}


def _ago(seconds):
    if seconds is None or seconds < 60:
        return ""
    minutes = round(seconds / 60)
    return f" (read {minutes} min ago)" if minutes < 120 else f" (read {round(minutes / 60)} h ago)"


def state_facts(state, question):
    """The robot's own state as facts, for what `question` asks ("" questions: none)."""
    asked = _plain(question)
    if not state or not asked:
        return []
    wants = {k for k, words in STATE_ASKS.items()
             if any(re.search(r"\b" + re.escape(w), asked) for w in words)}
    if "status" in wants:
        wants |= {"battery", "link"}
    facts = []
    b = state.get("battery") or {}
    if wants & {"battery", "heat"}:
        if isinstance(b.get("percent"), (int, float)):
            charging = {True: ", charging", False: ", not charging"}.get(b.get("charging"), "")
            fact = f"your battery is at {b['percent']:.0f} %{charging}{_ago(b.get('age_s'))}"
            if "heat" in wants and isinstance(b.get("temp_c"), (int, float)):
                # Said as the battery's, or it answered "la temperatura del motor es 33 °C".
                fact += (f"; your battery pack (not a motor) is at {b['temp_c']:.0f} °C; the "
                         "temperature of your motors is not known here")
            facts.append(fact)
        else:
            facts.append("your battery level is not known right now")
    if "link" in wants:
        if state.get("online") is False:
            facts.append("the app cannot reach you right now")
        elif state.get("online"):
            how = f" over {state['link']}" if state.get("link") else ""
            rtt = f", {state['latency_ms']} ms round trip" if state.get("latency_ms") is not None else ""
            facts.append(f"you are connected{how}{rtt}")
    return facts


def asks_about_itself(robot, question, state=None, cfg=None):
    """Whether one of the robot's facts or its state answers `question` ("¿quién te creó?",
    "¿cuánta batería tenés?") — then it is talk, not a reading."""
    if state_facts(state, question):
        return True
    cfg = cfg if cfg is not None else load_config()
    entry = (cfg.get("assistant_persona") or {}).get(
        (robot or cfg.get("assistant_robot") or "").strip().lower()) or {}
    asked = _plain(question)
    return bool(asked and isinstance(entry, dict) and any(
        re.search(r"\b" + re.escape(_plain(k)), asked)
        for f in entry.get("facts") or [] for k in f.get("about") or []))


def persona_for(robot=None, cfg=None, question=None, state=None):
    """Who the assistant is, for `robot` (config.json `assistant_persona`, by robot id), or ""
    when that robot has none. A missing robot is the configured default (`assistant_robot`),
    so the identity holds however the page asks.

    An entry is {"identity": "...", "facts": [{"about": [keywords], "fact": "..."}]} (a plain
    string is an identity with no facts). The identity always goes; a fact only when the
    question mentions one of its keywords — and only that fact.

    Measured 2026-10-19 on the live G1 camera and the reading set:
      - the short identity made it answer "Sí, estoy viendo tu mano…" (it said "parece que te
        están viendo") and "Soy Bender… fabricado por Maxi" (it said "Soy Qwen"), and still
        read 17.84 kW 6/6 at 1500 px (5/6 at 1152; 6/6 with no identity);
      - with every fact on every question it recited them all ("¿Quién te creó?" -> creator,
        work, best friend, brother) and the reading wobbled — and no wording ("only the one
        asked", "each only when that exact thing is asked") made the 4B model stop. So the
        choice of facts is made HERE, by keyword, not left to the model;
      - a longer identity read 17.84 as 1784; "speak of yourself as I and of the person as
        you" broke both again. Keep the identity short.
    """
    cfg = cfg if cfg is not None else load_config()
    personas = cfg.get("assistant_persona") or {}
    key = (robot or cfg.get("assistant_robot") or "").strip().lower()
    entry = personas.get(key) or ""
    if isinstance(entry, str):
        return entry.strip()
    identity = str(entry.get("identity") or "").strip()
    asked = _plain(question)
    facts = [str(f.get("fact") or "").strip() for f in entry.get("facts") or []
             if asked and any(re.search(r"\b" + re.escape(_plain(k)), asked)
                              for k in f.get("about") or [])]
    facts = [f.replace("{now}", _now_text(english=asked_in_english(question))) for f in facts if f]
    facts += state_facts(state, question)
    if not facts:
        return identity
    return f"{identity} About yourself, since it was asked: {'; '.join(facts)}."


# How it talks — only to what is not about the frame (a greeting, thanks, a comment, a question
# about itself or general knowledge). Without it the persona greeted ("¡Hola! 🤖"), acted
# ("*paws at the camera*"), ended everything with "¿Te gustaría saber algo más? 😊", answered a
# passing "¡Qué bueno es!" with the floor of the frame, and copied its own earlier answer into the
# next four (2026-10-28). Measured with src/vlm_chat_check.py: 46 % of answers clean -> 81 %. But
# on EVERY question it read worse (vlm_reading_check 93 % -> 79 %: "16" glands, "STECCK"), so a
# question about the frame keeps the reading prompt as measured: talks_about_the_frame decides.
# "A real robot with a body": "¿Qué hacés?" got "Solo soy una herramienta de inteligencia
# artificial", and a football question "no tengo cuerpo físico" (2026-10-28). "Greet back":
# with a flat "no greetings", "Hola" got a dry "¿Qué necesitas?". "Never say you did": a
# request the gateway did not recognise reached the model, which answered "Listo: he capturado
# la imagen… Lo guardo en mi memoria" — nothing was kept (2026-10-28).
CHAT_STYLE = (
    "Reply in the language of the message. You are a real robot with a body and a camera, not "
    "a program: never say you are an AI or have no body. Talk plainly, as yourself: do not "
    "open with a greeting unless you were greeted, and then greet back in a few words; no "
    "emojis, "
    "no actions between asterisks, no offers of more help, and do not end with a question "
    "unless you did not understand. If the message is not a question (a comment, thanks, a "
    "greeting, a word out of context), answer it in a few words, as a person would, without "
    "describing the image. Never repeat an earlier answer. You cannot take, save or store "
    "pictures or notes yourself: never say you did.")
# Whether a message needs the frame (LOOK) or is talk (CHAT), asked to the VLM server with the
# text alone (~27 ms). Measured 2026-10-28 on the 57 reading questions and the chat check's 74
# messages: 126/131, every reading question LOOK. The examples are none of those.
LOOKS = (
    "Classify a message said to a robot. CHAT: a greeting, thanks, praise, a comment, small "
    "talk, how it feels, a question about the robot itself (its name, who made it, where it "
    "works), the time or date, general knowledge, or words that make no sense. LOOK: a question "
    "or request about what is in front of the robot's camera: what is there, what it reads or "
    "shows, its brand, model, value or state. One word: CHAT or LOOK.")
LOOKS_EXAMPLES = [
    ("Buenas tardes", "CHAT"), ("¿Qué hay sobre la mesa?", "LOOK"),
    ("Excelente, te pasaste", "CHAT"), ("¿De qué marca es la heladera?", "LOOK"),
    ("¿Cuántos años tenés?", "CHAT"), ("¿Cuánto marca?", "LOOK"), ("Ajá, ok", "CHAT"),
    ("How far is the moon?", "CHAT"), ("Is the door open?", "LOOK"),
]


def talks_about_the_frame(question, url, timeout=2.0):
    """True when answering needs the frame (or it cannot be told): the reading prompt stays.
    False for talk, which gets CHAT_STYLE. Pointing at the view decides without the model."""
    from search_common import about_the_image
    if not question or about_the_image(question):
        return True
    shots = []
    for text, label in LOOKS_EXAMPLES:
        shots += [{"role": "user", "content": text}, {"role": "assistant", "content": label}]
    try:
        r = requests.post(f"{host_of(url)}/v1/chat/completions", timeout=timeout, json={
            "model": "default", "temperature": 0, "max_tokens": 3,
            "messages": [{"role": "system", "content": LOOKS}, *shots,
                         {"role": "user", "content": question}]})
        r.raise_for_status()
        return not r.json()["choices"][0]["message"]["content"].strip().upper().startswith("CHAT")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return True


# Words too common to tie a question to a note ("¿dónde está la llave?" ties by "llave").
_NOTE_STOP = {"esta", "este", "esto", "donde", "cuando", "como", "cual", "quien", "tiene", "tengo",
              "para", "pero", "porque", "sobre", "with", "where", "what", "when", "which", "there",
              "that", "this", "have", "your", "from", "about"}


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9]{4,}", _plain(text)) if w not in _NOTE_STOP}


def notes_about(notes, question):
    """The notes that share a word with `question`: the ones a question about the frame may
    still need ("¿dónde está la llave del tablero?" was answered from the picture, invented)."""
    asked = _content_words(question)
    return [n for n in notes or [] if asked & _content_words(n)]


def remembered(notes):
    """The notes the robot was asked to keep, as one sentence ("" without any)."""
    notes = [" ".join(str(n).split()) for n in notes or [] if str(n).strip()]
    if not notes:
        return ""
    return ("You were asked to remember these, use them when they answer the message: "
            + "; ".join(notes) + ".")


def free_prompt_system(robot=None, question=None, search=None, earlier=(), image=None,
                       vlm_url=None, model="default", chat=False, robot_state=None,
                       robot_notes=None, history=None):
    persona = persona_for(robot, question=question, state=robot_state)
    system = f"{FREE_PROMPT_SYSTEM} {persona}" if persona else FREE_PROMPT_SYSTEM
    # Web results, only when the question needs the web (search_common decides, or the caller
    # already did: `search`). `earlier`: the previous questions, for a "buscalo" follow-up;
    # `image`, `vlm_url`, `model`: for a question about something in view, whose query the model
    # writes looking at the frame. `chat`: not about the frame (talks_about_the_frame), so
    # CHAT_STYLE goes too.
    from search_common import search_fact
    found = search_fact(question, decided=search, earlier=earlier, image=image, vlm_url=vlm_url,
                        model=model, history=history)
    if found:
        return f"{system} {found}"
    # Talk gets every note. A question about the frame only those that share a word with it:
    # on a reading every sentence more is a risk (93 % -> 79 %), and "¿cuántos kVA marca?"
    # shares nothing with "la llave está en el cajón".
    if chat:
        notes = remembered(robot_notes)
        return f"{system} {CHAT_STYLE} {notes}" if notes else f"{system} {CHAT_STYLE}"
    notes = remembered(notes_about(robot_notes, question))
    return f"{system} {notes}" if notes else system


def query_vlm(img_b64, model, scope="industrial", max_tokens=8192,
              think=True, url=OLLAMA_HOST, timeout=300, num_ctx=16384,
              verbose=False, size=None, variant=None, prompt=None, robot=None):
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
    # A free-form `prompt` (ask anything about the image) overrides the canned
    # scope/variant prompts: use a neutral system message and DON'T force JSON, so
    # the model answers in plain text instead of the {objects:[...]} contract.
    if prompt:
        system_msg = free_prompt_system(robot, question=prompt)
        user_msg = prompt
        force_json = False
    else:
        sp = get_prompt(scope, variant)
        system_msg = sp["system"]
        user_msg = sp["user"]
        force_json = True

    supports_think = model_supports_thinking(model, url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg, "images": [img_b64]},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    if force_json:
        payload["format"] = "json"  # forces valid JSON in content
    # Only include `think` if the model supports it (otherwise Ollama returns 400).
    if supports_think:
        payload["think"] = think

    t0 = time.perf_counter()
    content, reasoning, done_reason, in_tok, out_tok = stream_chat(
        payload, url=url, timeout=timeout, verbose=verbose)
    elapsed = time.perf_counter() - t0

    if force_json:
        parsed, ok = extract_json(content)
        if ok:
            normalize_bboxes(parsed, size)
    else:
        parsed, ok = None, False  # free-form answer lives in `content`
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


# The conversation so far, as text, rides on each question (2026-10-28). Only text: an old
# frame costs ~1000-1300 tokens, a question with its answer ~100-300, and the server's window
# is 4096 (vlm-server.sh VLM_CTX) for the frame, the system message, the turns and the answer.
# The newest turns that fit go; older ones drop off. On llama-server the room is what the
# window leaves (history_budget); elsewhere, with no window to read, HISTORY_CHARS. More, and
# bigger windows, wait for a bigger card (.claude/roadmap/PLATAFORMA.md, "El asistente como un
# LLM de verdad").
HISTORY_CHARS = 5000
# Measured on the 4B's tokenizer, 2026-10-28: 3.6 characters a token in Spanish, 4.3 in
# English; the lower one so the estimate never overflows.
CHARS_PER_TOKEN = 3.5
FRAME_TOKENS = 1100       # a frame whose size cannot be read
ANSWER_TOKENS = 512       # kept for the answer: most are 50-300
TURN_OVERHEAD_CHARS = 40  # the chat template's markers around a question and its answer


def frame_tokens(img_b64):
    """What the frame costs the 4B: one token per 32x32 block, never under the server's
    --image-min-tokens (config llamacpp_image_min_tokens), plus the markers around it.
    Measured 2026-10-28: 1600x1200 took 1911 (1900 blocks), 615x410 took 1091 (the floor)."""
    try:
        import io

        from PIL import Image
        w, h = Image.open(io.BytesIO(base64.b64decode(img_b64))).size
    except Exception:   # any frame it cannot read: the usual size
        return FRAME_TOKENS
    floor = int(load_config().get("llamacpp_image_min_tokens") or 0)
    return max(floor, -(-w // 32) * -(-h // 32)) + 64


def history_budget(window, system_msg, prompt, frame):
    """Characters of earlier turns that fit `window` tokens next to the system message, the
    question, the frame (`frame` tokens, 0 without it) and the answer. A fixed 5000 forgot the first questions with 1.3k
    tokens still free (2026-10-28); this forgets only when the window is full."""
    if not window:
        # The web results make the system message long; the conversation gives way to them.
        return max(1500, HISTORY_CHARS - max(0, len(system_msg) - 1000))
    room = (window - ANSWER_TOKENS - frame
            - (len(system_msg) + len(prompt)) / CHARS_PER_TOKEN)
    return max(0, int(room * CHARS_PER_TOKEN))


def history_messages(history, budget=HISTORY_CHARS):
    """[{"question", "answer"}] oldest first -> chat messages for the newest turns that fit
    `budget` characters, in order. Turns without an answer are skipped."""
    turns = []
    used = 0
    for turn in reversed(history or []):
        q = str((turn or {}).get("question") or "").strip()
        a = str((turn or {}).get("answer") or "").strip()
        if not q or not a:
            continue
        cost = len(q) + len(a) + TURN_OVERHEAD_CHARS
        if used + cost > budget:
            break
        used += cost
        turns.append((q, a))
    messages = []
    for q, a in reversed(turns):
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return messages


def query_vlm_stream(img_b64, model, prompt, max_tokens=8192, num_ctx=16384,
                     url=OLLAMA_HOST, robot=None, history=None, search=None, usage=None,
                     robot_state=None, robot_notes=None):
    """Stream a FREE-PROMPT answer token by token (a generator of text pieces).

    This is the streaming sibling of query_vlm for the "ask anything" case only:
    it uses the same neutral system message and does NOT force JSON, so the answer
    is plain text that can be spoken sentence by sentence as it arrives. It yields
    only the `content` pieces (the model's answer); any `thinking` is ignored (it
    goes to a separate field and is not meant to be read aloud). The caller (the
    service) forwards each piece to the browser as it is produced.

    For the fastest time-to-first-word use an instruct model (no thinking), so the
    answer starts immediately instead of after a reasoning block.

    `history` is the conversation so far ([{"question", "answer"}], oldest first): its newest
    turns go before the question, as text (history_messages), so "¿y la de la izquierda?"
    knows what it follows. The image is only the current frame.

    `search`: whether this question needs the web, when the caller decided it already (None:
    decided here: only an explicit ask). A searched question that does not point at what is in view goes
    WITHOUT the frame: with it, the model kept answering "la imagen es de una habitación, no
    tiene información sobre partidos" before the results (2026-10-28).

    `usage`, a dict, is filled when the answer ends: prompt_tokens, completion_tokens, window
    (the server's context, llama-server only), turns_sent of turns_total — what the page shows
    so the person can tell when to start a new conversation.
    """
    from search_common import about_the_image, wants_search
    if search is None:
        search = wants_search(prompt)
    asked_before = [t.get("question") or "" for t in history or []]
    # Talk or a question about the frame: only llama-server is asked (the CLI stays as it was).
    # A question about itself (one of its facts or its state answers it) is talk without asking:
    # "¿Cuánta batería tenés?" was taken for the frame and answered "¡Hola!… 😊" (2026-10-28).
    chat = (not search and vlm_backend() == "llamacpp"
            and (asks_about_itself(robot, prompt, robot_state)
                 or not talks_about_the_frame(prompt, url)))
    system_msg = free_prompt_system(robot, question=prompt, search=search, earlier=asked_before,
                                    image=img_b64, vlm_url=url, model=model, chat=chat,
                                    robot_state=robot_state, robot_notes=robot_notes,
                                    history=history)
    question = {"role": "user", "content": prompt}
    if not search or about_the_image(prompt):
        question["images"] = [img_b64]
    window = llamacpp_window(url) if vlm_backend() == "llamacpp" else None
    frame = frame_tokens(img_b64) if "images" in question else 0
    earlier = history_messages(history, budget=history_budget(window, system_msg, prompt, frame))
    supports_think = model_supports_thinking(model, url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            # Web results make the system message long; the conversation gives way to them
            # (history_budget), or the frame, the results, the turns and the answer would not
            # fit the window.
            *earlier,
            question,
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    if supports_think:
        payload["think"] = False  # ask for no reasoning; content should come first
    if usage is not None:
        usage["turns_sent"] = len(earlier) // 2
        usage["turns_total"] = sum(1 for t in history or []
                                   if str((t or {}).get("question") or "").strip()
                                   and str((t or {}).get("answer") or "").strip())
    if vlm_backend() == "llamacpp":
        for chunk in _openai_chunks(payload, url, 600):
            if usage is not None and chunk.get("usage"):
                usage["prompt_tokens"] = chunk["usage"].get("prompt_tokens")
                usage["completion_tokens"] = chunk["usage"].get("completion_tokens")
                usage["window"] = window
                usage["answer_room"] = ANSWER_TOKENS   # never given to the conversation
            piece = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if piece:
                yield piece
        return
    chat_url = host_of(url) + "/api/chat"

    with requests.post(chat_url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = chunk.get("message", {}).get("content")
            if piece:
                yield piece


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
    m, sec = divmod(round(s), 60)
    return f"{m}m{sec:02d}s"


def progress_bar(done, total, suffix="", width=24):
    """Draw/update a progress bar on a single line (carriage return).

    When done >= total it moves to a new line so it does not overwrite what comes next.
    No dependencies: just block characters and \\r.
    """
    total = max(total, 1)
    frac = min(done / total, 1.0)
    filled = round(width * frac)
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
