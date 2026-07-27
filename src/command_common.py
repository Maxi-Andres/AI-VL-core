#!/usr/bin/env python3
"""
command_common.py — voice/text command interpreter for Unitree robots.

This is the "command interpreter" from ROBOT_CONTROL.md (Phase 1): it turns a
spoken command (already transcribed to text by /transcribe) into a structured
**skill JSON** that a downstream executor maps to Unitree SDK calls. It reuses the
Ollama client in vlm_common (same /api/chat streaming path) — it does NOT talk to
the robot and does NOT move anything; it only decides WHAT should happen.

    speech --/transcribe--> text --interpret()--> { "skill": ..., "params": ... }
                                                        |
                                        (Phase 2) skill executor --> unitree_sdk2

Multi-robot: the same interpreter serves BOTH Unitree platforms on this machine —
the **G1** humanoid (`LocoClient` + `G1ArmActionClient`) and the **Go2** quadruped
"dog" (`SportClient`). Each robot has its OWN skill catalog because their actions
differ (the G1 has arm poses; the Go2 has dog tricks like flips / walk-upright).
The `robot` argument selects which catalog to use.

Scope: every action the respective SDK client already ships (locomotion, posture,
gestures/tricks). Vision-guided skills (grab/place) are NOT here yet — they need
perception-3D (Phase 3+) and are left out so the interpreter never claims a
capability the stack cannot execute.

Each robot's SKILLS catalog is the SINGLE SOURCE OF TRUTH: the model prompt is
built from it and the model's output is validated against it. Add a skill in one
place and both the prompt and the validation pick it up. The executor-facing
numbers (speed presets, arm action IDs) live here too so Phase 2 has one place to
read.
"""
import json
import time

import vlm_common
from vlm_common import extract_json, model_supports_thinking, OLLAMA_HOST, stream_chat


# --------------------------------------------------------------------------- #
# Executor-facing constants (read by the Phase 2 skill executor, not by the LLM)
# --------------------------------------------------------------------------- #
# Categorical speeds -> concrete Move(vx, vy, vyaw) velocities. The interpreter
# only emits the category ("slow|normal|fast"); the executor turns it into
# (vx, vyaw). Kept conservative on purpose — start gentle. Per robot because the
# dog (Go2) safely moves faster than the humanoid (G1, which can fall).
#   vx   = forward/back linear speed  [m/s]  (also used for strafing vy)
#   vyaw = turn rate                  [rad/s]
G1_SPEED_PRESETS = {
    "slow":   {"vx": 0.2, "vyaw": 0.3},
    "normal": {"vx": 0.4, "vyaw": 0.6},
    "fast":   {"vx": 0.7, "vyaw": 1.0},
}
GO2_SPEED_PRESETS = {
    "slow":   {"vx": 0.3, "vyaw": 0.5},
    "normal": {"vx": 0.6, "vyaw": 1.0},
    "fast":   {"vx": 1.2, "vyaw": 2.0},
}
DEFAULT_SPEED = "slow"

# Default bounded-step duration (seconds) when the command does not say how long
# and is not "continuous". The executor issues Move() for this long, then stops.
DEFAULT_STEP_S = 2.0

# G1 arm preset actions -> SDK action IDs (G1ArmActionClient.ExecuteAction(id), see
# unitree_sdk2 g1_arm_action_client.hpp `action_map`). The interpreter emits the
# NAME; the executor resolves the ID here. (Go2 has no arms.)
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

# Shared param specs (identical across robots that walk on a velocity command).
_WALK_PARAMS = {
    "direction": {"values": ["forward", "backward", "left", "right"],
                  "default": "forward"},
    "speed": {"values": ["slow", "normal", "fast"], "default": DEFAULT_SPEED},
    "duration_s": {"type": "number|null",
                   "desc": "seconds to move; null = one short step", "default": None},
    "continuous": {"type": "bool",
                   "desc": "true = keep going until 'stop'", "default": False},
}
_TURN_PARAMS = {
    "direction": {"values": ["left", "right"], "default": "left"},
    "speed": {"values": ["slow", "normal", "fast"], "default": DEFAULT_SPEED},
    "duration_s": {"type": "number|null",
                   "desc": "seconds to turn; null = a short turn", "default": None},
}
_UNKNOWN_SKILL = {
    "desc": "Use ONLY when the command matches no skill above or is not a robot "
            "command. Do not force an unrelated command into another skill.",
    "params": {},
    "examples": ["what's the weather", "tell me a joke", "(unintelligible)"],
}


# --------------------------------------------------------------------------- #
# G1 (humanoid) skill catalog — maps to LocoClient + G1ArmActionClient
# --------------------------------------------------------------------------- #
# Each skill: a one-line description (goes into the prompt) and a `params` spec
# mapping param name -> {"values"/"type", "default"}. `params` empty = no params.
# `examples` are English canonical utterances shown to the model (the code stays
# English-only per repo convention); the model is told commands usually arrive in
# Spanish (Rioplatense) and must handle either language.
G1_SKILLS = {
    # --- Locomotion (LocoClient.Move / StopMove) --------------------------- #
    "walk": {
        "desc": "Walk / move the body in a straight direction.",
        "params": _WALK_PARAMS,
        "examples": ["walk forward", "come here", "go back", "step to the left",
                     "keep walking forward"],
    },
    "turn": {
        "desc": "Turn/rotate in place to the left or right.",
        "params": _TURN_PARAMS,
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
    "unknown": _UNKNOWN_SKILL,
}


# --------------------------------------------------------------------------- #
# Go2 (quadruped "dog") skill catalog — maps to go2 SportClient
# --------------------------------------------------------------------------- #
# See unitree_sdk2 include/unitree/robot/go2/sport/sport_client.hpp. The Go2 has
# NO arms; instead it has dog postures and acrobatic tricks. Some tricks (flips)
# are risky — the executor must gate them (clear space, secured), but the
# interpreter still recognizes them.
GO2_SKILLS = {
    # --- Locomotion (SportClient.Move / StopMove) -------------------------- #
    "walk": {
        "desc": "Walk / move in a straight direction.",
        "params": _WALK_PARAMS,
        "examples": ["walk forward", "come here", "go back", "step to the left",
                     "keep walking forward"],
    },
    "turn": {
        "desc": "Turn/rotate in place to the left or right.",
        "params": _TURN_PARAMS,
        "examples": ["turn right", "spin left", "rotate to the right"],
    },
    "stop": {
        "desc": "Stop all motion immediately (zero velocity). Safety command.",
        "params": {},
        "examples": ["stop", "halt", "stay", "don't move"],
    },
    # --- Posture (SportClient) --------------------------------------------- #
    "stand_up": {
        "desc": "Stand up with locked/stiff legs (firm stand).",
        "params": {},
        "examples": ["stand up", "get up", "stand firm"],
    },
    "balance_stand": {
        "desc": "Normal standing mode, actively balancing and ready to walk.",
        "params": {},
        "examples": ["balance", "ready", "normal stand"],
    },
    "stand_down": {
        "desc": "Lie down / lower the body to the ground (prone).",
        "params": {},
        "examples": ["lie down", "get down", "down"],
    },
    "sit": {
        "desc": "Sit down (dog sitting posture).",
        "params": {},
        "examples": ["sit", "sit down"],
    },
    "rise_sit": {
        "desc": "Get up from the sitting posture.",
        "params": {},
        "examples": ["get up from sitting", "rise", "stop sitting"],
    },
    "recovery_stand": {
        "desc": "Recover to standing after a fall or from lying down.",
        "params": {},
        "examples": ["recover", "get back up", "stand up after falling"],
    },
    "damp": {
        "desc": "Damping mode: go limp/compliant (soft, safe rest).",
        "params": {},
        "examples": ["relax", "go limp", "damp"],
    },
    # --- Gestures / tricks (SportClient) ----------------------------------- #
    "hello": {
        "desc": "Greet: raise a front paw and wave hello.",
        "params": {},
        "examples": ["say hi", "wave", "greet", "give me your paw"],
    },
    "stretch": {
        "desc": "Do a stretch.",
        "params": {},
        "examples": ["stretch", "stretch out"],
    },
    "scrape": {
        "desc": "Scrape / bow gesture (front down, rear up).",
        "params": {},
        "examples": ["bow", "take a bow", "scrape"],
    },
    "heart": {
        "desc": "Make a heart gesture.",
        "params": {},
        "examples": ["make a heart", "do the heart"],
    },
    "dance1": {
        "desc": "Perform dance routine 1.",
        "params": {},
        "examples": ["dance", "dance one", "do a dance"],
    },
    "dance2": {
        "desc": "Perform dance routine 2.",
        "params": {},
        "examples": ["dance two", "the other dance"],
    },
    "front_jump": {
        "desc": "Jump forward.",
        "params": {},
        "examples": ["jump", "jump forward", "hop"],
    },
    "front_pounce": {
        "desc": "Pounce forward.",
        "params": {},
        "examples": ["pounce", "lunge forward"],
    },
    "front_flip": {
        "desc": "Front flip (acrobatic — needs clear space; risky).",
        "params": {},
        "examples": ["front flip", "do a flip"],
    },
    "back_flip": {
        "desc": "Back flip (acrobatic — needs clear space; risky).",
        "params": {},
        "examples": ["backflip", "flip backwards"],
    },
    "left_flip": {
        "desc": "Side flip to the left (acrobatic — risky).",
        "params": {},
        "examples": ["side flip", "flip to the left"],
    },
    "handstand": {
        "desc": "Handstand: front paws on the ground, rear legs up.",
        "params": {"on": {"type": "bool",
                          "desc": "true = enter, false = exit", "default": True}},
        "examples": ["handstand", "do a handstand", "stop the handstand"],
        "notes": "Spanish 'hacé el pino' / 'el pino' means do a handstand.",
    },
    "walk_upright": {
        "desc": "Stand and walk on the hind legs (upright).",
        "params": {"on": {"type": "bool",
                          "desc": "true = enter, false = exit", "default": True}},
        "examples": ["stand on two legs", "walk upright", "get down from upright"],
    },
    "pose": {
        "desc": "Posing mode: hold a body attitude / pose.",
        "params": {"on": {"type": "bool",
                          "desc": "true = enter, false = exit", "default": True}},
        "examples": ["strike a pose", "pose", "stop posing"],
    },
    "set_gait": {
        "desc": "Switch the walking gait / locomotion style.",
        "params": {
            "gait": {"values": ["classic", "free_walk", "trot_run", "static_walk",
                                "economic", "cross_step"], "default": "classic"},
        },
        "examples": ["switch to trot", "walk normally", "use classic gait",
                     "do the cross step"],
    },
    "unknown": _UNKNOWN_SKILL,
}


# --------------------------------------------------------------------------- #
# Robot registry — selects the catalog + executor constants per robot
# --------------------------------------------------------------------------- #
ROBOTS = {
    "g1": {
        "label": "Unitree G1 (humanoid)",
        "intro": "You control a Unitree G1 humanoid robot.",
        "skills": G1_SKILLS,
        "speed_presets": G1_SPEED_PRESETS,
        "arm_action_ids": ARM_ACTION_IDS,
    },
    "go2": {
        "label": "Unitree Go2 (quadruped robot dog)",
        "intro": "You control a Unitree Go2 quadruped robot dog.",
        "skills": GO2_SKILLS,
        "speed_presets": GO2_SPEED_PRESETS,
        "arm_action_ids": {},
    },
}
DEFAULT_ROBOT = "g1"


def _resolve(robot):
    """Return a valid robot id, falling back to DEFAULT_ROBOT on anything unknown."""
    return robot if robot in ROBOTS else DEFAULT_ROBOT


def list_robots():
    """[{id, label}] for every robot — lets the UI build a selector."""
    return [{"id": rid, "label": r["label"]} for rid, r in ROBOTS.items()]


def catalog(robot):
    """The skill catalog + executor constants for one robot (for GET /skills)."""
    rid = _resolve(robot)
    r = ROBOTS[rid]
    return {
        "robot": rid,
        "skills": {name: {"desc": s["desc"], "params": s["params"]}
                   for name, s in r["skills"].items()},
        "speed_presets": r["speed_presets"],
        "arm_actions": r["arm_action_ids"],
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


def build_system_prompt(robot=DEFAULT_ROBOT):
    """Build the interpreter system prompt from a robot's SKILLS catalog."""
    r = ROBOTS[_resolve(robot)]
    lines = [
        f"{r['intro']} Convert the user's spoken command into ONE skill call. The "
        "command is usually in Spanish (Rioplatense dialect) but may be in "
        "English — understand either.",
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
    for name, s in r["skills"].items():
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


def normalize_intent(parsed, robot=DEFAULT_ROBOT):
    """Validate/normalize a parsed model object into a safe intent dict.

    Guarantees the returned dict has a known `skill` (for THIS robot) and only the
    params that skill declares, each coerced to its type with the declared default
    on anything missing or invalid. Unknown skills collapse to "unknown". This is
    what keeps a hallucinated field or type from reaching the executor.
    """
    skills = ROBOTS[_resolve(robot)]["skills"]
    if not isinstance(parsed, dict):
        return {"skill": "unknown", "params": {}, "say": ""}

    skill = parsed.get("skill")
    if not isinstance(skill, str) or skill not in skills:
        skill = "unknown"

    raw_params = parsed.get("params")
    if not isinstance(raw_params, dict):
        raw_params = {}

    params = {}
    for name, p in skills[skill]["params"].items():
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
def interpret(text, model, robot=DEFAULT_ROBOT, image_b64=None, url=OLLAMA_HOST,
              timeout=120, num_ctx=8192, max_tokens=1024):
    """Interpret a spoken/typed command into a validated skill intent.

    text       : the transcribed command (what /transcribe returned).
    model      : Ollama model tag (e.g. "qwen3-vl:4b"); an instruct model gives the
                 fastest reply since command parsing needs no reasoning.
    robot      : which robot's catalog to use ("g1" | "go2"); unknown -> default.
    image_b64  : optional current camera frame — unused by the SDK-action skills but
                 accepted so future vision skills can share this entry point.

    Returns a dict:
      { ok, robot, skill, params, say, understood, content, elapsed_ms,
        in_tokens, out_tokens }
    `ok` is False only if the model produced no parseable JSON (the intent then
    safely falls back to skill "unknown"). Raises requests.RequestException on a
    network/server failure (same contract as query_vlm).
    """
    rid = _resolve(robot)
    text = (text or "").strip()
    if not text:
        return {"ok": False, "robot": rid, "skill": "unknown", "params": {},
                "say": "", "understood": "", "content": "", "elapsed_ms": 0.0,
                "in_tokens": None, "out_tokens": None}

    user_msg = {"role": "user", "content": f"Command: {text}"}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt(rid)},
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
    intent = normalize_intent(parsed if ok else None, rid)

    return {
        "ok": ok,
        "robot": rid,
        "skill": intent["skill"],
        "params": intent["params"],
        "say": intent["say"],
        "understood": text,
        "content": content,
        "elapsed_ms": round(elapsed * 1000, 1),
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }
