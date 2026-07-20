#!/usr/bin/env python3
"""
command_common.py — voice/text command interpreter for the Unitree G1.

This is the "command interpreter" from ROBOT_CONTROL.md (Phase 1): it turns a
spoken command (already transcribed to text by /transcribe) into a structured
**skill JSON** that a downstream executor maps to Unitree SDK calls. It reuses the
Ollama client in vlm_common (same /api/chat streaming path) — it does NOT talk to
the robot and does NOT move anything; it only decides WHAT should happen.

    speech --/transcribe--> text --interpret()--> { "skill": ..., "params": ... }
                                                        |
                                        (Phase 2) skill executor --> unitree_sdk2

Scope of THIS module: every action the SDK already ships (locomotion, posture/FSM,
gestures, and the arm preset actions). Vision-guided skills (grab/place) are NOT
here yet — they need perception-3D (Phase 3+) and are deliberately left out so the
interpreter never claims a capability the stack cannot execute.

The SKILLS catalog below is the SINGLE SOURCE OF TRUTH: the model prompt is built
from it and the model's output is validated against it. Add a skill in one place
and both the prompt and the validation pick it up. The executor-facing numbers
(speed presets, arm action IDs) live here too so Phase 2 has one place to read.
"""
import json
import time

import vlm_common
from vlm_common import extract_json, model_supports_thinking, OLLAMA_HOST, stream_chat


# --------------------------------------------------------------------------- #
# Executor-facing constants (read by the Phase 2 skill executor, not by the LLM)
# --------------------------------------------------------------------------- #
# Categorical speeds -> concrete Unitree LocoClient.Move() velocities. The
# interpreter only emits the category ("slow|normal|fast"); the executor turns it
# into (vx, vyaw). Kept conservative on purpose — the G1 falls, so start gentle.
#   vx   = forward/back linear speed  [m/s]  (also used for strafing vy)
#   vyaw = turn rate                  [rad/s]
SPEED_PRESETS = {
    "slow":   {"vx": 0.2, "vyaw": 0.3},
    "normal": {"vx": 0.4, "vyaw": 0.6},
    "fast":   {"vx": 0.7, "vyaw": 1.0},
}
DEFAULT_SPEED = "slow"

# Default bounded-step duration (seconds) when the command does not say how long
# and is not "continuous". The executor issues Move() for this long, then stops.
DEFAULT_STEP_S = 2.0

# Arm preset actions -> SDK action IDs (G1ArmActionClient.ExecuteAction(id), see
# unitree_sdk2 g1_arm_action_client.hpp `action_map`). The interpreter emits the
# NAME; the executor resolves the ID here.
ARM_ACTION_IDS = {
    "release_arm": 99,   # return arms to rest / release a held pose
    "two_hand_kiss": 11,
    "left_kiss": 12,
    "right_kiss": 12,
    "hands_up": 15,
    "clap": 17,
    "high_five": 18,
    "hug": 19,
    "heart": 20,
    "right_heart": 21,
    "reject": 22,
    "right_hand_up": 23,
    "x_ray": 24,
    "face_wave": 25,
    "high_wave": 26,
    "shake_hand": 27,
}


# --------------------------------------------------------------------------- #
# Skill catalog — SINGLE SOURCE OF TRUTH (prompt + validation are built from it)
# --------------------------------------------------------------------------- #
# Each skill: a one-line description (goes into the prompt) and a `params` spec
# mapping param name -> {"values"/"type", "default"}. `params` empty = no params.
# `examples` are English canonical utterances shown to the model (the code stays
# English-only per repo convention); the model is told commands usually arrive in
# Spanish (Rioplatense) and must handle either language.
SKILLS = {
    # --- Locomotion (LocoClient.Move / StopMove) --------------------------- #
    "walk": {
        "desc": "Walk / move the body in a straight direction.",
        "params": {
            "direction": {"values": ["forward", "backward", "left", "right"],
                          "default": "forward"},
            "speed": {"values": ["slow", "normal", "fast"], "default": DEFAULT_SPEED},
            "duration_s": {"type": "number|null",
                           "desc": "seconds to move; null = one short step", "default": None},
            "continuous": {"type": "bool",
                           "desc": "true = keep going until 'stop'", "default": False},
        },
        "examples": ["walk forward", "come here", "go back", "step to the left",
                     "keep walking forward"],
    },
    "turn": {
        "desc": "Turn/rotate in place to the left or right.",
        "params": {
            "direction": {"values": ["left", "right"], "default": "left"},
            "speed": {"values": ["slow", "normal", "fast"], "default": DEFAULT_SPEED},
            "duration_s": {"type": "number|null",
                           "desc": "seconds to turn; null = a short turn", "default": None},
        },
        "examples": ["turn right", "spin left", "rotate to the right"],
    },
    "stop": {
        "desc": "Stop all motion immediately (zero velocity). Safety command.",
        "params": {},
        "examples": ["stop", "halt", "stay", "don't move"],
    },
    # --- Posture / FSM (LocoClient FSM ids) -------------------------------- #
    "stand_up": {
        "desc": "Stand up to the normal standing posture.",
        "params": {},
        "examples": ["stand up", "get up", "stand"],
    },
    "balance_stand": {
        "desc": "Enter balanced standing mode (ready to walk, actively balancing).",
        "params": {},
        "examples": ["balance", "ready to walk", "balance stand"],
    },
    "sit": {
        "desc": "Sit down.",
        "params": {},
        "examples": ["sit", "sit down"],
    },
    "squat": {
        "desc": "Squat / crouch down.",
        "params": {},
        "examples": ["squat", "crouch", "get low"],
    },
    "high_stand": {
        "desc": "Stand at maximum height (legs extended).",
        "params": {},
        "examples": ["stand tall", "stand high", "raise up"],
    },
    "low_stand": {
        "desc": "Stand at minimum height (legs bent low).",
        "params": {},
        "examples": ["stand low", "lower yourself"],
    },
    "damp": {
        "desc": "Damping mode: go limp/compliant. Soft, safe rest of the actuators.",
        "params": {},
        "examples": ["relax", "go limp", "damp", "loosen up"],
    },
    "zero_torque": {
        "desc": "Zero-torque mode: motors produce no torque. Use only when secured.",
        "params": {},
        "examples": ["zero torque", "release the motors", "power down the joints"],
    },
    "start": {
        "desc": "Enter the main operational control state (ready state) after boot.",
        "params": {},
        "examples": ["start", "get ready", "wake up"],
    },
    # --- Gestures (LocoClient.WaveHand / ShakeHand) ------------------------ #
    "wave_hand": {
        "desc": "Wave a hand as a greeting.",
        "params": {
            "turn": {"type": "bool",
                     "desc": "true = wave while turning toward the person", "default": False},
        },
        "examples": ["wave", "say hi", "wave hello", "wave and turn to me"],
    },
    "shake_hand": {
        "desc": "Offer/perform a handshake.",
        "params": {},
        "examples": ["shake hands", "give me your hand", "let's shake"],
    },
    # --- Arm preset actions (G1ArmActionClient.ExecuteAction) -------------- #
    "arm_action": {
        "desc": "Perform a preset upper-body arm gesture chosen by name.",
        "params": {
            "action": {"values": list(ARM_ACTION_IDS.keys()), "default": "release_arm"},
        },
        "examples": ["put your hands up", "clap", "give me a high five", "give me a hug",
                     "make a heart", "blow a kiss", "cross your arms to say no",
                     "put your arms down"],
        "notes": "Map 'put/lower your arms down', 'rest your arms' or 'let go' to "
                 "action=release_arm (the arms-at-rest pose).",
    },
    # --- Fallback ---------------------------------------------------------- #
    "unknown": {
        "desc": "Use ONLY when the command matches no skill above or is not a robot "
                "command. Do not force an unrelated command into another skill.",
        "params": {},
        "examples": ["what's the weather", "tell me a joke", "(unintelligible)"],
    },
}


# --------------------------------------------------------------------------- #
# Prompt construction (built from the catalog so it never drifts from validation)
# --------------------------------------------------------------------------- #
def _params_line(spec):
    """Render a skill's params spec as a compact one-line hint for the prompt."""
    if not spec:
        return "no params"
    parts = []
    for name, p in spec.items():
        if "values" in p:
            opts = "|".join(str(v) for v in p["values"])
            parts.append(f"{name} ({opts})")
        else:
            parts.append(f"{name} ({p.get('type', 'value')})")
    return ", ".join(parts)


def build_system_prompt():
    """Build the interpreter system prompt from the SKILLS catalog."""
    lines = [
        "You control a Unitree G1 humanoid robot. Convert the user's spoken command "
        "into ONE skill call. The command is usually in Spanish (Rioplatense "
        "dialect) but may be in English — understand either.",
        "",
        "Respond with ONLY a valid JSON object, no markdown, no text before or after:",
        '{"skill": <one skill name>, "params": {<params for that skill>}, '
        '"say": <short spoken confirmation IN THE SAME LANGUAGE as the command>}',
        "",
        "Rules:",
        "- Pick exactly ONE skill from the list. If nothing fits, use \"unknown\".",
        "- Include only the params that skill defines; omit a param to use its default.",
        "- \"say\" is a brief, natural confirmation to speak back, written in the "
        "SAME language the command used (do not translate to English). For "
        "\"unknown\", say you did not understand.",
        "- Safety: any command to stop/freeze/hold still maps to \"stop\".",
        "- For walk/turn, set continuous=true when the command implies going until "
        "told to stop (e.g. \"keep walking\", \"seguí caminando\").",
        "",
        "Skills:",
    ]
    for name, s in SKILLS.items():
        lines.append(f"- {name}: {s['desc']} params: {_params_line(s['params'])}")
        if s.get("notes"):
            lines.append(f"    note: {s['notes']}")
        if s.get("examples"):
            lines.append(f"    e.g. {'; '.join(s['examples'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Validation / normalization of the model's output against the catalog
# --------------------------------------------------------------------------- #
def _coerce_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "si", "sí")
    return default


def _coerce_number(v, default):
    if isinstance(v, bool):  # bool is a subclass of int — reject it explicitly
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return default
    return default


def normalize_intent(parsed):
    """Validate/normalize a parsed model object into a safe intent dict.

    Guarantees the returned dict has a known `skill` and only the params that skill
    declares, each coerced to its type with the declared default on anything
    missing or invalid. Unknown skills collapse to "unknown". This is what keeps a
    hallucinated field or type from reaching the executor.
    """
    if not isinstance(parsed, dict):
        return {"skill": "unknown", "params": {}, "say": ""}

    skill = parsed.get("skill")
    if not isinstance(skill, str) or skill not in SKILLS:
        skill = "unknown"

    raw_params = parsed.get("params")
    if not isinstance(raw_params, dict):
        raw_params = {}

    params = {}
    for name, p in SKILLS[skill]["params"].items():
        default = p.get("default")
        if name not in raw_params:
            params[name] = default
            continue
        val = raw_params[name]
        if "values" in p:
            params[name] = val if val in p["values"] else default
        elif p.get("type") == "bool":
            params[name] = _coerce_bool(val, default)
        elif p.get("type", "").startswith("number"):
            params[name] = None if val is None else _coerce_number(val, default)
        else:
            params[name] = val

    say = parsed.get("say")
    return {"skill": skill, "params": params, "say": say if isinstance(say, str) else ""}


# --------------------------------------------------------------------------- #
# The interpreter
# --------------------------------------------------------------------------- #
def interpret(text, model, image_b64=None, url=OLLAMA_HOST, timeout=120,
              num_ctx=8192, max_tokens=1024):
    """Interpret a spoken/typed command into a validated skill intent.

    text       : the transcribed command (what /transcribe returned).
    model      : Ollama model tag (e.g. "qwen3-vl:4b"); an instruct model gives the
                 fastest reply since command parsing needs no reasoning.
    image_b64  : optional current camera frame — unused by the SDK-action skills but
                 accepted so future vision skills can share this entry point.

    Returns a dict:
      { ok, skill, params, say, understood, content, elapsed_ms,
        in_tokens, out_tokens }
    `ok` is False only if the model produced no parseable JSON (the intent then
    safely falls back to skill "unknown"). Raises requests.RequestException on a
    network/server failure (same contract as query_vlm).
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "skill": "unknown", "params": {},
                "say": "", "understood": "", "content": "", "elapsed_ms": 0.0,
                "in_tokens": None, "out_tokens": None}

    user_msg = {"role": "user", "content": f"Command: {text}"}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            user_msg,
        ],
        "stream": True,
        "format": "json",  # force valid JSON in content
        "options": {
            "temperature": 0.0,   # deterministic parsing
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    # qwen3-vl "thinking" checkpoints ignore think=false, but we ask for it anyway:
    # command parsing wants a direct JSON answer, not a reasoning block.
    if model_supports_thinking(model, url):
        payload["think"] = False

    t0 = time.perf_counter()
    content, reasoning, done_reason, in_tok, out_tok = stream_chat(
        payload, url=url, timeout=timeout)
    elapsed = time.perf_counter() - t0

    parsed, ok = extract_json(content)
    intent = normalize_intent(parsed if ok else None)

    return {
        "ok": ok,
        "skill": intent["skill"],
        "params": intent["params"],
        "say": intent["say"],
        "understood": text,
        "content": content,
        "elapsed_ms": round(elapsed * 1000, 1),
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }
