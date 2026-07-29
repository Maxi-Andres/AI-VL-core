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
# --------------------------------------------------------------------------- #
# WHERE EVERY ID COMES FROM. Tag convention used throughout this file:
#
#   [robot] The ROBOT published it — read from its own answer (arm GetActionList api
#           7107, or loco GetFsmId 7001) with g1_fsm_watch.py. Authoritative: this is
#           the firmware talking.
#   [sdk]   Hardcoded in the unitree_sdk2 / unitree_ros2 headers on this machine.
#   [web]   Documented outside the SDK (CMU Robotics Knowledgebase, QUADRUPED G1
#           docs). NOT confirmed against the robot yet.
#
# Note WHAT the robot publishes and what it does not: GetActionList returns the ARM
# subsystem completely (every preset action + every named routine), so the block below
# is finished — there is nothing left to discover there. It says NOTHING about the
# locomotion FSM (Run / Walk / Climb / Lie up), which is a different subsystem with no
# "list" api at all; those ids can only be read one at a time with GetFsmId while the
# robot is put into each mode. See FSM_IDS in the executor's g1_commands.py.
# --------------------------------------------------------------------------- #
# ALL [robot] — the full arm list as this G1 published it: 23 actions, not the 16 the
# SDK header hardcodes. The robot's own name for each is in the comment. Two things the
# SDK table got wrong: it repeats id 12 for both left and right kiss (the robot
# separates them, right = 13), and it omits 8 actions entirely.
ARM_ACTION_IDS = {
    "release_arm": 99,    # release_arm — return arms to rest / release a held pose
    "turn_back_wave": 1,  # turn_back_wave (robot restricts it to fsm 500/501)
    "two_hand_kiss": 11,  # blow_kiss_with_both_hands
    "left_kiss": 12,      # blow_kiss_with_left_hand
    "right_kiss": 13,     # blow_kiss_with_right_hand  (SDK header says 12 — wrong)
    "hands_up": 15,       # both_hands_up
    "clap": 17,           # clamp
    "high_five": 18,      # high_five
    "hug": 19,            # hug
    "heart": 20,          # make_heart_with_both_hands   (mode_machine 5/6 only)
    "right_heart": 21,    # make_heart_with_right_hand   (mode_machine 5/6 only)
    "reject": 22,         # refuse
    "right_hand_up": 23,  # right_hand_up
    "x_ray": 24,          # ultraman_ray
    "face_wave": 25,      # wave_under_head
    "high_wave": 26,      # wave_above_head
    "shake_hand": 27,     # shake_hand
    "box_win_left": 28,   # box_left_hand_win            (mode_machine 5/6 only)
    "box_win_right": 29,  # box_right_hand_win           (mode_machine 5/6 only)
    "box_win_both": 30,   # box_both_hand_win            (mode_machine 5/6 only)
    "hand_on_heart": 33,  # right_hand_on_heart
    "hands_up_right": 34,  # both_hands_up_deviate_right
    "forward_push": 36,   # forward_push                 (mode_machine 5/6 only)
}

# ALL [robot] — named "teach" actions the robot stores separately from the id-indexed
# ones above. They run through a DIFFERENT api (ExecuteAction by NAME, arm api 7108) and
# each has a duration. Same GetActionList answer — this is the app's "dance".
CUSTOM_ACTIONS = {
    "Waist_Drum_Dance": 9.5,
    "Scratch_head": 8.1,
    "Spin_discs": 6.9,
    "Throw_money": 8.1,
}

# Display names for the arm actions, matching what the Unitree G1 phone app calls
# them, so an operator reads the SAME word in both places. Keys are the skill param
# values above; the UI falls back to a prettified key for anything missing.
ARM_ACTION_LABELS = {
    "release_arm": "Release arm",
    "two_hand_kiss": "Two-hand kiss",
    "left_kiss": "Left kiss",
    "right_kiss": "Right kiss",
    "hands_up": "Hands up",
    "clap": "Clap",
    "high_five": "High five",
    "hug": "Hug",
    "heart": "Arm heart",
    "right_heart": "Right heart",
    "reject": "Reject",
    "right_hand_up": "Right hand up",
    "x_ray": "X-ray",
    "face_wave": "Face wave",
    "high_wave": "High wave",
    "shake_hand": "Handshake",
    "turn_back_wave": "Turn back + wave",
    "box_win_left": "Box win (left)",
    "box_win_right": "Box win (right)",
    "box_win_both": "Box win (both)",
    "hand_on_heart": "Hand on heart",
    "hands_up_right": "Hands up (to the right)",
    "forward_push": "Forward push",
}

# The named dance/teach actions, labelled for the UI.
CUSTOM_ACTION_LABELS = {
    "Waist_Drum_Dance": "Waist drum dance",
    "Scratch_head": "Scratch head",
    "Spin_discs": "Spin discs",
    "Throw_money": "Throw money",
}

# --------------------------------------------------------------------------- #
# SAFE_MODE gating — skills the executor refuses while safe mode is ON.
# --------------------------------------------------------------------------- #
# Criterion (apply it when adding a skill): block anything that can make the robot
# LOSE ITS SUPPORT (fall or go limp) or change its control mode unpredictably.
# Controlled posture changes that keep the robot balanced the whole way down
# (sit, squat, stand heights) are NOT blocked — they are the normal way to park it.
# Mirrored by the executor's per-robot DANGEROUS_SKILLS; the UI reads this list from
# GET /skills so all three tiers agree instead of each keeping its own copy.
G1_DANGEROUS = {
    "zero_torque",    # motors produce no torque -> a standing humanoid collapses
    "damp",           # limp/compliant -> collapses from any standing posture
    "dance",          # multi-second whole-body routine: needs clear space around it
    "squat_sdk",      # the SDK's squat (fsm 2): observed half-falling on this robot
    "set_fsm_id",     # raw state jump: can land the robot in a mode that drops it
    "set_speed_mode",  # raw speed mode: undocumented values, can mean "run"
    "switch_mode",    # swaps the whole motion controller out from under the robot
}
GO2_DANGEROUS = {
    "front_flip", "back_flip", "left_flip",  # acrobatics: airborne, needs clear space
    "front_jump", "front_pounce",
    "handstand", "walk_upright",             # balances on two legs -> tips over easily
    "damp",                                  # limp -> drops to the floor
    "dance1", "dance2",                      # whole-body routines: leaves its footprint
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
        "label": "Walk",
        "desc": "Walk / move the body in a straight direction.",
        "params": _WALK_PARAMS,
        "examples": ["walk forward", "come here", "go back", "step to the left",
                     "keep walking forward"],
    },
    "turn": {
        "label": "Turn",
        "desc": "Turn/rotate in place to the left or right.",
        "params": _TURN_PARAMS,
        "examples": ["turn right", "spin left", "rotate to the right"],
    },
    "stop": {
        "label": "Stop",
        "desc": "Stop all motion immediately (zero velocity). Safety command.",
        "params": {},
        "examples": ["stop", "halt", "stay", "don't move"],
    },
    # --- Posture / FSM (LocoClient FSM ids) -------------------------------- #
    "stand_up": {
        "label": "Preparation (stand up)",
        "desc": "FSM 4 — the preparatory standing posture the phone app and the "
                "handheld controller call Ready/Preparation (L1+UP). This is the step "
                "between Damping and a locomotion mode, and also how it rises from a "
                "squat ('squat up').",
        "params": {},
        "examples": ["stand up", "get up", "stand", "squat up", "get ready",
                     "preparation"],
    },
    "balance_stand": {
        "label": "Balance stand",
        "desc": "Enter balanced standing mode (ready to walk, actively balancing).",
        "params": {},
        "examples": ["balance", "ready to walk", "balance stand"],
    },
    "sit": {
        "label": "Seating",
        "desc": "Sit down.",
        "params": {},
        "examples": ["sit", "sit down", "seating"],
    },
    "squat": {
        "label": "Squat / up (toggle)",
        "desc": "FSM 706 [robot: confirmed off the wire] — exactly what the phone app's "
                "Squat and Squat up both send. ONE toggle: from standing it squats down "
                "and parks damped; from down it stands back up and restores the "
                "locomotion mode. This is the squat that works — the SDK's FSM 2 "
                "half-falls on this robot.",
        "params": {},
        "examples": ["squat", "crouch", "get low", "squat up", "get up from the squat"],
        "notes": "Same skill both ways: send it again to reverse the last direction.",
    },
    "squat_sdk": {
        "label": "Squat (SDK, unstable)",
        "hidden": True,
        "desc": "FSM 2 [sdk] — the SDK's documented squat. Kept only for reference: on "
                "this robot it HALF-FALLS. Use `squat` (706) instead.",
        "params": {},
    },
    "high_stand": {
        "label": "High stand",
        "desc": "Stand at maximum height (legs extended).",
        "params": {},
        "examples": ["stand tall", "stand high", "raise up"],
    },
    "low_stand": {
        "label": "Low stand",
        "desc": "Stand at minimum height (legs bent low).",
        "params": {},
        "examples": ["stand low", "lower yourself"],
    },
    "damp": {
        "label": "Damping",
        "desc": "Damping mode: go limp/compliant. A STANDING robot collapses — hold "
                "it or have it already low before using this.",
        "params": {},
        "examples": ["relax", "go limp", "damp", "damping", "loosen up"],
    },
    "zero_torque": {
        "label": "Zero torque",
        "desc": "Zero-torque mode: motors produce no torque. Use only when secured.",
        "params": {},
        "examples": ["zero torque", "release the motors", "power down the joints"],
    },
    "start": {
        "label": "Main operation (1-DoF waist)",
        "desc": "FSM 500 [sdk] — the app's Main Operation Control (controller R1+X): "
                "the normal walking controller for the 1-DoF-waist G1. On a "
                "3-DoF-waist robot use 'Walk mode (3-DoF waist)' (501) instead — 500 "
                "is reported to jitter there.",
        "params": {},
        "examples": ["start", "main operation", "wake up"],
    },
    # The locomotion modes the phone app offers after Preparation. FSM ids are NOT in
    # the SDK (it wraps only 0-4 and 500); these come from the CMU Robotics
    # Knowledgebase G1 page + the handheld-controller map in the QUADRUPED G1 docs,
    # and both 501 and 801 are independently proven to exist by the SDK's own arm
    # error header ("actions are only supported in fsm id {500, 501, 801}").
    # The robot now ANSWERS a bad id (7302), so a wrong guess shows up as an error in
    # the UI instead of silently doing nothing.
    # The walk/run modes come in a pair, one per waist variant. Same controller, one id
    # for the 1-DoF-waist robot and one for the 3-DoF-waist robot — a robot rejects the
    # id that is not for its own variant with error 7302. THIS robot answers 501/802,
    # so it is the 3-DoF-waist G1 and the *_waist ones are the ones that work on it.
    "run": {
        "label": "Run mode (1-DoF waist)",
        "desc": "FSM 801 [web] — run controller (the handheld's R2+X) for the "
                "1-DoF-waist G1. Only switches the controller; use walk/the joysticks "
                "to actually move.",
        "params": {},
        "examples": ["run mode", "switch to running", "enable run"],
        "notes": "'run forward' / 'corré para adelante' is walk with speed=fast — NOT "
                 "this skill, which only changes the locomotion controller.",
    },
    "run_waist": {
        "label": "Run mode (3-DoF waist)",
        "desc": "FSM 802 [robot: confirmed] — the app's Run on the 3-DoF-waist G1, the "
                "counterpart of 801. Only switches the controller.",
        "params": {},
        "examples": ["run mode with waist", "run"],
    },
    "lie_up": {
        "label": "Lie up (get up)",
        "desc": "FSM 702 [robot: confirmed] — the app's Lie up: stand up from lying on "
                "the floor. Run it from Damping; the robot drives its own stages and "
                "ends in a locomotion mode.",
        "params": {},
        "examples": ["get up", "lie up", "stand up from the floor", "levantate del piso"],
    },
    "climb": {
        "label": "Climb (stairs)",
        "desc": "FSM 812 [robot: confirmed] — the app's Climb, the stair-climbing "
                "controller, on the 3-DoF-waist G1. Only switches the controller; use "
                "walk/the joysticks to actually move.",
        "params": {},
        "examples": ["climb", "stair mode", "climb the stairs"],
    },
    "walk_waist": {
        "label": "Walk mode (3-DoF waist)",
        "desc": "FSM 501 [robot: observed live] — the regular walking controller for "
                "the 3-DoF-waist G1: the SAME thing 'Main operation' (500) is for the "
                "1-DoF-waist robot, not an extra feature on top of it.",
        "params": {},
        "examples": ["waist control mode", "walk with waist control", "walk mode"],
    },
    # --- Gestures (LocoClient.WaveHand / ShakeHand) ------------------------ #
    "wave_hand": {
        "label": "Wave",
        "desc": "Wave a hand as a greeting. Needs the Preparation state (FSM 500).",
        "params": {
            "turn": {"type": "bool",
                     "desc": "true = wave while turning toward the person", "default": False},
        },
        "examples": ["wave", "say hi", "wave hello", "wave and turn to me"],
    },
    "shake_hand": {
        "label": "Handshake",
        "desc": "Handshake. TWO STAGES: on=true offers the hand, on=false ends it and "
                "brings the arm back. Needs the Preparation state (FSM 500).",
        "params": {
            "on": {"type": "bool",
                   "desc": "true = offer the hand (start), false = end the handshake",
                   "default": True},
        },
        "examples": ["shake hands", "give me your hand", "let's shake",
                     "ok, let go of my hand"],
        "notes": "'let go', 'that's enough', 'suéltame' after a handshake mean "
                 "shake_hand with on=false (the release stage).",
    },
    # --- Arm preset actions (G1ArmActionClient.ExecuteAction) -------------- #
    "arm_action": {
        "label": "Arm action",
        "desc": "Perform a preset upper-body arm gesture chosen by name. Needs the "
                "Preparation state (FSM 500).",
        "params": {
            "action": {"values": list(ARM_ACTION_IDS.keys()),
                       "labels": ARM_ACTION_LABELS, "default": "release_arm"},
        },
        "examples": ["put your hands up", "clap", "give me a high five", "give me a hug",
                     "make a heart", "blow a kiss", "cross your arms to say no",
                     "put your arms down"],
        "notes": "Map 'put/lower your arms down', 'rest your arms' or 'let go' to "
                 "action=release_arm (the arms-at-rest pose).",
    },
    "dance": {
        "label": "Dance",
        "desc": "Run one of the robot's stored named routines (the app's dances). "
                "Whole-body and several seconds long — needs clear space.",
        "params": {
            "name": {"values": list(CUSTOM_ACTIONS.keys()),
                     "labels": CUSTOM_ACTION_LABELS, "default": "Waist_Drum_Dance"},
        },
        "examples": ["dance", "do the drum dance", "spin the discs", "throw money"],
        "notes": "These are the named 'teach' actions the robot reports separately from "
                 "the numbered arm actions; 'stop_dance' cuts one short.",
    },
    "stop_dance": {
        "label": "Stop dance",
        "desc": "Cut a running named routine short.",
        "params": {},
        "examples": ["stop dancing", "that's enough dancing"],
    },
    # --- Raw mode control (hidden from the interpreter, UI-only) ----------- #
    # `hidden` skills are NOT offered to the LLM and are rejected by
    # normalize_intent — a hallucinated state jump on a humanoid is unacceptable.
    # They exist so an operator can reach (and DISCOVER) the modes Unitree does not
    # document: the phone app has Run, Walk (waist control), Climb and Lie up, whose
    # FSM ids are not published in either the SDK or unitree_ros2. What IS proven:
    # ids 501 and 801 exist beyond the documented {0,1,2,3,4,500} (the arm-action
    # error says actions only work in fsm id {500, 501, 801}). Recipe: put the robot
    # in the mode from the app, read GetFsmId / rt/sportmodestate, then add it above
    # as a normal named skill.
    "set_fsm_id": {
        "label": "Set FSM id",
        "hidden": True,
        "desc": "Raw LocoClient SetFsmId — jump straight to a state machine id.",
        "params": {"fsm_id": {"type": "number", "desc": "state id", "default": 500}},
    },
    "set_speed_mode": {
        "label": "Set speed mode",
        "hidden": True,
        "desc": "Raw LocoClient SetSpeedMode (api 7107). Values are undocumented — "
                "the app's Run mode is the likely consumer.",
        "params": {"mode": {"type": "number", "desc": "speed mode", "default": 0}},
    },
    "switch_mode": {
        "label": "Switch motion mode",
        "hidden": True,
        "desc": "Motion switcher SelectMode by name/alias — how the app swaps whole "
                "controllers (e.g. the walk/run/climb modes).",
        "params": {"name": {"type": "string", "desc": "mode name or alias",
                            "default": ""}},
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
        "label": "Damping",
        "desc": "Damping mode: go limp/compliant — the dog drops to the floor.",
        "params": {},
        "examples": ["relax", "go limp", "damp", "damping"],
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
        "short": "G1",
        "intro": "You control a Unitree G1 humanoid robot.",
        "skills": G1_SKILLS,
        "speed_presets": G1_SPEED_PRESETS,
        "arm_action_ids": ARM_ACTION_IDS,
        "dangerous": G1_DANGEROUS,
    },
    "go2": {
        "label": "Unitree Go2 (quadruped robot dog)",
        "short": "Go2",
        "intro": "You control a Unitree Go2 quadruped robot dog.",
        "skills": GO2_SKILLS,
        "speed_presets": GO2_SPEED_PRESETS,
        "arm_action_ids": {},
        "dangerous": GO2_DANGEROUS,
    },
}
DEFAULT_ROBOT = "g1"


def _resolve(robot):
    """Return a valid robot id, falling back to DEFAULT_ROBOT on anything unknown."""
    return robot if robot in ROBOTS else DEFAULT_ROBOT


def list_robots():
    """[{id, label, short}] for every robot — lets the UI build a selector (`label`)
    and compact status pills (`short`)."""
    return [{"id": rid, "label": r["label"], "short": r["short"]}
            for rid, r in ROBOTS.items()]


def _default_label(name):
    return name.replace("_", " ").capitalize()


def catalog(robot):
    """The skill catalog + executor constants for one robot (for GET /skills).

    Carries everything a UI needs so it never keeps its own copy: the display
    `label` (matched to the Unitree phone app's wording), `hidden` for raw controls
    that must not reach the interpreter, and `dangerous` — the skills the executor
    refuses while safe mode is on."""
    rid = _resolve(robot)
    r = ROBOTS[rid]
    return {
        "robot": rid,
        "skills": {name: {"desc": s["desc"],
                          "label": s.get("label") or _default_label(name),
                          "hidden": bool(s.get("hidden")),
                          "params": s["params"]}
                   for name, s in r["skills"].items()},
        "speed_presets": r["speed_presets"],
        "arm_actions": r["arm_action_ids"],
        "dangerous": sorted(r["dangerous"]),
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
        if s.get("hidden"):
            continue  # raw mode controls: operator-only, never offered to the model
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
    # A hidden skill is never in the prompt, so the model can only produce one by
    # hallucinating — and a raw FSM/mode jump on a humanoid is not something a
    # misheard sentence gets to do. Collapse it like any unknown name.
    if (not isinstance(skill, str) or skill not in skills
            or skills[skill].get("hidden")):
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
