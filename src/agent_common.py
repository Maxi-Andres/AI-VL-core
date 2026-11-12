#!/usr/bin/env python3
"""
agent_common.py — the /agent tab's brain: an instruction in, the robot acting, a trace out.

The /command interpreter maps one sentence to ONE fixed skill. This is the step after it
(G1-GROOT.md paso 6, the presentation's "una pestaña nueva: video y chat"): a language model
that UNDERSTANDS the instruction, PLANS it, and carries it out with TOOLS, each tool's own steps
and result shown as they happen. GR00T joins later as one more tool ("manipulate"); the loop
does not change.

    heard → understood + plan → [thought → tool → its events → result]* → reply

Every event goes to `emit` as a dict the moment it happens (the service streams them as SSE).
The model is the app's VLM (Qwen3-VL-4B on llama-server), asked for compact JSON; nothing here
knows a word list: what to do is the model's call, what each tool does is the tool's.

Tools, as the model sees them, are TOOLS below. A tool's result goes back to the model, so the
next step depends on whether the last one worked — a failed pick is reported, not hidden.

The conversation so far rides on every instruction (2026-11-12, as the vision chat does since
2026-10-28): each earlier turn is what was asked, what the robot did with its tools and what it
answered, the newest that fit HISTORY_CHARS. So "ahora dejala en la mesa" knows which bottle
it is holding. It is context only: the model is told the instruction to carry out is the last
one, never to do an earlier one again.

Everything the vision chat can do, the agent can too (2026-11-12, "el agent tiene que tener
todas las herramientas del vlm"): `search` the internet (search_common, the same SearXNG),
`remember` a note or a picture of what it sees (the gateway keeps it in the robot's Memory, the
same one Live's "acordate…" fills: iacore only says so, in a "keep" event), and it is told what
the chat is told — the time, the robot's notes and its state (battery, link).
"""
from __future__ import annotations

import json
import math
import threading
import time

from vlm_common import LLAMACPP_HOST, extract_json, stream_chat

MAX_STEPS = 6
# Characters of earlier turns sent with an instruction. The model's window is 4096 tokens
# (~14 000 characters at 3.5 a token, vlm_common.CHARS_PER_TOKEN) for the system message and its
# tool list (~3 000), this run's own steps (up to ~2 500) and the answer (300 tokens): about
# 4 000 are left for the conversation.
HISTORY_CHARS = 4000
HISTORY_DONE_MAX = 8      # tool uses kept per earlier turn
TOOLS = {
    "look": {
        "args": {"question": "what to ask about what the robot's camera sees now"},
        "does": "Looks through the robot's head camera and answers a question about it. Use it "
                "to check where things are or whether an action worked.",
    },
    "pick": {
        "args": {"object": "the thing to pick up, in words, e.g. 'the water bottle'"},
        "does": "Picks the object up from the table in front of the robot with the right hand, "
                "lifts it, puts it back where it was and brings the arm home. MOVES THE ROBOT. "
                "Only objects standing on the table, about the size of a bottle or a cup.",
    },
    "navigate": {
        "args": {"target": "a thing it can see now, in words ('the chair', 'that person'): it "
                           "walks up to it and stops in front, facing it. Leave it out to go "
                           "by metres instead",
                 "forward_m": "metres forward (negative = back), at most 10",
                 "left_m": "metres to its left (negative = right), usually 0",
                 "turn_deg": "degrees to turn, at most 180: LEFT is positive, RIGHT is negative "
                             "('turn right' = -90, 'turn left' = 90, 'turn around' = 180)"},
        "does": "Walks the robot by itself, going round obstacles and people, up to 10 m. MOVES "
                "THE ROBOT. Give a target it can see, or forward_m/left_m (and turn_deg to face "
                "another way at the end); only turn_deg turns it in place. If the target is not "
                "in sight, turn and look first.",
    },
    "command": {
        "args": {"skill": "the name of one of the moves below",
                 "params": "that move's parameters, e.g. to clap: skill \"arm_action\" with "
                           "params {\"action\": \"clap\"}"},
        "does": "One of the robot's built-in moves. MOVES THE ROBOT. The moves: {skills}",
    },
    "search": {
        "args": {"query": "what to look up, in a few words, e.g. 'clima Buenos Aires hoy'"},
        "does": "Searches the internet and gives the first results with their sites and "
                "dates. Use it for anything outside the room and outside what you know: news, "
                "weather, prices, schedules, facts.",
    },
    "remember": {
        "args": {"note": "the fact to remember, in words (leave it out for only a picture)",
                 "picture": "true to also keep a picture of what the camera sees now"},
        "does": "Keeps a note, a picture, or both, in the robot's memory, where the team sees "
                "them; your notes come back to you with every instruction. Use it when the "
                "person asks you to remember something or to take a picture.",
    },
    "place": {
        "args": {"object": "the thing to move, in words, e.g. 'the water bottle'",
                 "into": "where to put it, in words, e.g. 'the cardboard box'"},
        "does": "Picks the object up from the table with the right hand, carries it over the "
                "other thing on the table, lets go of it there and brings the arm home. MOVES "
                "THE ROBOT. Use it to put something in or on something else; check with look.",
    },
}


def _tools_text():
    return "\n".join(f"- {name}({', '.join(f'{a}: {d}' for a, d in t['args'].items())}): "
                     f"{t['does']}".replace("{skills}", _skills_text())
                     for name, t in TOOLS.items())


# The command tool's moves: the command interpreter's G1 catalog (command_common, the single
# source of truth), minus what is not the agent's to do. Out: navigate's walking and turning;
# the skills safe mode holds back (they drop the robot or swap its controller) and the hidden
# raw ones; getting up off the floor and the stair controller, which only an operator who sees
# the robot should start; and the walk/run controllers: the mode is the operator's (Nav2 walks
# in Run at the Slow speed, 2026-11-11 — in Walk the G1 marks time at that speed).
COMMAND_LEFT_OUT = {"walk", "turn", "unknown", "lie_up", "climb",
                    "start", "run", "run_waist", "walk_waist"}


def command_skills():
    """{name: skill spec} the agent's command tool offers."""
    import command_common as cc
    cat = cc.catalog("g1")
    out = set(cat["dangerous"]) | COMMAND_LEFT_OUT
    return {n: s for n, s in cat["skills"].items() if n not in out and not s.get("hidden")}


def _skills_text():
    """Each move as the call that makes it: the skill's name, then its params. Written as
    "arm_action (Arm action; action: one of … clap …)" the 4B took "clap" for a move
    (2026-11-12)."""
    parts = []
    for name, s in command_skills().items():
        ps = []
        for pname, p in s["params"].items():
            if "values" in p:
                ps.append(f'"{pname}": one of {"|".join(p["values"])}')
            elif p.get("type") == "bool":
                ps.append(f'"{pname}": true or false')
        label = s.get("label") or name
        parts.append(f'skill "{name}" ({label})' + (f" with params {{{', '.join(ps)}}}" if ps else ""))
    return "; ".join(parts) + "."


PLAN_SYSTEM = (
    "You control a Unitree G1 humanoid robot. It walks around the room by itself, going round "
    "obstacles and people. You can use these tools:\n"
    "{tools}\n"
    "Read the instruction and answer ONLY JSON: {{\"understood\": what is being asked, in one "
    "short line, \"plan\": [the steps, each a few words], \"needs_robot\": true or false}}. "
    "If it is only a question or a chat, needs_robot is false and the plan may be empty. "
    "Write understood and the plan in the language of the instruction.")

STEP_SYSTEM = (
    "You control a Unitree G1 humanoid robot; it walks around the room by itself, going round "
    "obstacles and people. Tools:\n"
    "{tools}\n"
    "You are carrying out the instruction step by step. Given what has happened so far, answer "
    "ONLY JSON, one of:\n"
    "{{\"thought\": why, \"tool\": a tool name, \"args\": {{...}}}} to use a tool, or\n"
    "{{\"thought\": why, \"reply\": what to tell the person}} when it is done, impossible, or "
    "only needed words. Never claim something worked unless a tool result says so. Never do a "
    "step again that already worked: once the results show the instruction is done, reply. "
    "Reply in the "
    "language of the instruction, short and plain, as the robot, in your own words: never "
    "repeat or echo what the person said. The instruction comes from speech-to-text and may "
    "hold other people's talk around the robot: act only on what is said to you.")


def history_text(history, budget=HISTORY_CHARS):
    """[{"question", "answer", "done": [..]}] oldest first -> the newest turns that fit `budget`
    characters, as text for the prompt ("" with none). A turn with neither an answer nor
    anything done is skipped."""
    blocks = []
    used = 0
    for turn in reversed(history or []):
        turn = turn or {}
        q = str(turn.get("question") or "").strip()
        a = str(turn.get("answer") or "").strip()
        done = [str(d).strip() for d in (turn.get("done") or [])[:HISTORY_DONE_MAX] if str(d).strip()]
        if not q or not (a or done):
            continue
        lines = [f"Person: {q}"] + [f"  did: {d}" for d in done] + ([f"Robot: {a}"] if a else [])
        block = "\n".join(lines)
        if used + len(block) + 1 > budget:
            break
        used += len(block) + 1
        blocks.append(block)
    if not blocks:
        return ""
    return ("Earlier in this conversation (context only, oldest first; do NOT do any of it "
            "again):\n" + "\n".join(reversed(blocks)) + "\n\n")


def llm_json(system, user, url=LLAMACPP_HOST, max_tokens=300, image_b64=None):
    """One compact-JSON answer from the app's VLM, or {} if it gave none."""
    msg = {"role": "user", "content": user}
    if image_b64:
        msg["images"] = [image_b64]
    payload = {"model": "", "stream": True, "format": "json",
               "options": {"temperature": 0.1, "num_predict": max_tokens},
               "messages": [{"role": "system", "content": system}, msg]}
    content, *_ = stream_chat(payload, url=url, timeout=120)
    parsed, ok = extract_json(content)
    return parsed if ok and isinstance(parsed, dict) else {}


# --------------------------------------------------------------------------- #
# The tools
# --------------------------------------------------------------------------- #
def _frame_b64():
    """What the robot's head camera sees now, as a base64 JPEG ("" when it gave none)."""
    import base64

    import cv2

    import locate_object as lo
    ok, buf = cv2.imencode(".jpg", lo.grab_color(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def tool_look(args, emit, stop, url=LLAMACPP_HOST, frame=None):
    question = str(args.get("question") or "What do you see?")
    b64 = (frame or _frame_b64)()
    if not b64:
        return {"ok": False, "error": "no picture"}
    payload = {"model": "", "stream": True,
               "options": {"temperature": 0.1, "num_predict": 200},
               "messages": [{"role": "user", "content": question, "images": [b64]}]}
    content, *_ = stream_chat(payload, url=url, timeout=60)
    return {"ok": True, "answer": content.strip()}


SEARCH_RESULTS_CHARS = 1800     # of results told to the model: its window is shared


def tool_search(args, emit, stop, find=None):
    """The internet, as the vision chat searches it (search_common, the local SearXNG)."""
    import datetime
    from urllib.parse import urlsplit

    import search_common as sc
    query = " ".join(str(args.get("query") or "").split())[:200]
    if not query:
        return {"ok": False, "error": "no query: say what to look up"}
    emit({"type": "tool_event", "tool": "search", "step": "searching", "detail": query})
    try:
        if find is None:
            results = sc.search(query)
            if results:
                sc.read_pages(results, query)
        else:
            results = find(query)
    except Exception as exc:
        return {"ok": False, "error": f"the search failed ({type(exc).__name__}): do not guess"}
    if not results:
        return {"ok": True, "query": query, "results": "nothing found: say so, do not guess"}
    lines, used = [], 0
    for r in results:
        site = (urlsplit(r.get("url") or "").hostname or "").removeprefix("www.")
        when = f", {r['date']}" if r.get("date") else ""
        found = f" {r['found']}" if r.get("found") else ""
        line = f"{r.get('title') or ''} — {r.get('content') or ''}{found} ({site}{when})"
        if used + len(line) > SEARCH_RESULTS_CHARS:
            break
        lines.append(line)
        used += len(line)
    emit({"type": "tool_event", "tool": "search", "step": "found",
          "detail": f"{len(lines)} results"})
    return {"ok": True, "query": query, "today": datetime.date.today().isoformat(),
            "results": " | ".join(lines)}


def tool_remember(args, emit, stop, frame=None):
    """A note and/or a picture for the robot's memory. The gateway keeps it (it owns the
    Memory); this only hands it over, as a "keep" event the gateway takes off the stream."""
    note = " ".join(str(args.get("note") or "").split())[:500]
    picture = args.get("picture") in (True, "true", "True", 1)
    if not note and not picture:
        return {"ok": False, "error": "nothing to remember: give a note or picture=true"}
    image = (frame or _frame_b64)() if picture else ""
    if picture and not image:
        return {"ok": False, "error": "the camera gave no picture"}
    emit({"type": "keep", "note": note, **({"image": image} if image else {})})
    return {"ok": True, "kept": " and ".join(k for k, on in (("a note", note), ("a picture", image))
                                             if on)}


def tool_pick(args, emit, stop, start=None, poll_s=0.5):
    """Run a manipulation.PickAction to its end, every step of it emitted as a tool event.
    `stop` (the STOP button) cancels it; the arm then goes home before this returns."""
    return _run_action("pick", args, emit, stop, start, poll_s)


def tool_place(args, emit, stop, start=None, poll_s=0.5):
    """The same, carrying the object to args["into"] instead of putting it back."""
    into = str(args.get("into") or "").strip()
    if not into:
        return {"ok": False, "error": "no place to put it named"}
    return _run_action("place", args, emit, stop, start, poll_s, into=into)


def _run_action(tool, args, emit, stop, start, poll_s, **kw):
    import manipulation
    start = start or manipulation.start_pick
    obj = str(args.get("object") or "").strip()
    if not obj:
        return {"ok": False, "error": "no object named"}
    try:
        act = start(obj, **kw)
    except manipulation.ActionError as exc:
        return {"ok": False, "error": str(exc)}
    seen = 0
    while True:
        if stop.is_set():
            act.cancel()
        snap = act.snapshot(seen)
        for ev in snap["events"]:
            emit({"type": "tool_event", "tool": tool, "step": ev["step"],
                  "detail": ev["detail"]})
        seen += len(snap["events"])
        if snap["state"] != "running" and not act.thread.is_alive():
            result = snap.get("result") or {}
            out = {"ok": snap["state"] == "success", "state": snap["state"],
                   "held": bool(result.get("held")),
                   "last": snap["events"][-1]["detail"] if snap["events"] else ""}
            if "placed" in result:
                out["placed"] = bool(result["placed"])
            return out
        time.sleep(poll_s)


def _executor_walk(body, timeout=6):
    """POST the executor's /walk/control (the loop itself runs on the robot, in the relay)."""
    import json as _json
    import urllib.error
    import urllib.request

    import manipulation
    req = urllib.request.Request(f"{manipulation.EXECUTOR_URL}/walk/control?robot=g1",
                                 data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            out = _json.loads(exc.read() or b"{}")
        except ValueError:
            out = {}
        return {"ok": False, "error": out.get("error") or f"HTTP {exc.code}", **out}


WALK_POLL_S = 0.4
WALK_RUNNING = ("starting", "running")


# NAVIGATE (2026-11-12): one tool for walking, replacing walk and turn — the robot walks with
# Nav2 now (it sees and goes round things, up to the relay's 10 m), and two tools for one job
# were two chances for the small model to pick wrong. A target in words is found the way pick
# finds things (locate_object: YOLO finds, the VLM chooses, the depth camera measures) and the
# robot stops NAV_STANDOFF_M in front of it, facing it.
NAV_STANDOFF_M = 0.6
NAV_ARRIVED_M = 0.15          # closer than this to the stopping point: only turn to face it
NAV_FACING_DEG = 8.0          # ... and not even that when it already faces it
NAV_MAX_BOXES = 12            # YOLO's most confident, numbered for the VLM to choose from


def goal_toward(x, y, standoff=NAV_STANDOFF_M):
    """(forward_m, left_m, turn_deg) to stop `standoff` short of the point (x forward, y left
    of the robot, metres), facing it. Nothing to walk when it is closer than that already."""
    r = math.hypot(x, y)
    bearing = math.atan2(y, x)
    d = r - standoff
    if d < NAV_ARRIVED_M:
        return 0.0, 0.0, math.degrees(bearing)
    return d * math.cos(bearing), d * math.sin(bearing), math.degrees(bearing)


def locate_seen(phrase):
    """(x forward, y left) in metres of `phrase` in the robot's camera, and how it was found;
    (None, why) when it is not in sight or cannot be measured."""
    import locate_object as lo
    import perception3d as p3
    cal = p3.load_calibration(lo.CALIBRATION)
    z, info = lo.fetch_depth(frames=5)
    frame = lo.grab_color()
    boxes = sorted(lo.yolo_boxes(frame, None), key=lambda b: -b[2])[:NAV_MAX_BOXES]
    i = lo.vlm_choose(frame, boxes, phrase) if boxes else None
    if i is not None:
        box, how = boxes[i][0], f"yolo {boxes[i][1]}, chosen by the vlm"
    else:
        box, how = lo.vlm_box(frame, phrase), "the vlm's own box"
    if box is None:
        return None, f"I do not see {phrase}"
    h, w = frame.shape[:2]
    r = p3.locate(z, box, cal, color_size=(w, h), scale_m=info.get("scale_m", 0.001))
    if not r.get("ok"):
        return None, f"I see {phrase} but cannot tell how far it is ({r.get('error')})"
    return (float(r["torso"][0]), float(r["torso"][1])), how


def tool_navigate(args, emit, stop, call=None, poll_s=WALK_POLL_S, locate=None):
    """Walk to a target seen in words, or by metres and degrees, to the walk's end."""
    target = str(args.get("target") or "").strip()
    if target:
        locate = locate or locate_seen
        emit({"type": "tool_event", "tool": "navigate", "step": "looking",
              "detail": f"looking for {target}"})
        try:
            where, how = locate(target)
        except Exception as exc:                 # no depth snapshot, no camera
            return {"ok": False, "error": f"could not look: {exc}"}
        if where is None:
            return {"ok": False, "error": how}
        fwd, left, turn = goal_toward(*where)
        emit({"type": "tool_event", "tool": "navigate", "step": "seen",
              "detail": f"{target}: {math.hypot(*where):.1f} m away, "
                        f"{abs(turn):.0f} deg {'left' if turn >= 0 else 'right'} ({how})"})
        if fwd == left == 0.0 and abs(turn) < NAV_FACING_DEG:
            return {"ok": True, "state": "already there", "distance_m": round(math.hypot(*where), 2)}
    else:
        try:
            fwd, left, turn = (float(args.get(k) or 0)
                               for k in ("forward_m", "left_m", "turn_deg"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "forward_m, left_m and turn_deg must be numbers"}
        if not (fwd or left or turn):
            return {"ok": False, "error": "nowhere to go: give a target, or metres or degrees"}
    return _walk("navigate", fwd, left, turn, emit, stop, call, poll_s)


def _walk(tool, fwd, left, turn, emit, stop, call, poll_s):
    """One walk to a point (the relay's Walk, on the robot), followed to its end. `stop` (the
    STOP button) cancels it on the robot. The limits are the relay's: a refusal comes back as
    the tool's error, for the model to read."""
    call = call or _executor_walk
    out = call({"op": "goto", "dx": round(fwd, 3), "dy": round(left, 3),
                "dyaw": round(math.radians(turn), 4)})
    if not out.get("ok"):
        return {"ok": False, "error": out.get("error") or "the robot refused the walk"}
    phase, blocked = None, False
    while True:
        if stop.is_set():
            call({"op": "cancel"})
        w = (call({"op": "status"}) or {}).get("walk") or {}
        if w.get("phase") and w.get("phase") != phase and w.get("state") in WALK_RUNNING:
            phase = w["phase"]
            emit({"type": "tool_event", "tool": tool, "step": phase,
                  "detail": "on its way" if phase == "translate" else "turning"})
        if bool(w.get("blocked")) != blocked and w.get("state") in WALK_RUNNING:
            blocked = bool(w.get("blocked"))
            why = ((w.get("blocked") or {}).get("reason") or "something is in the way"
                   if blocked else "the way is clear again")
            emit({"type": "tool_event", "tool": tool, "step": "blocked" if blocked else "going on",
                  "detail": why})
        if w.get("state") not in WALK_RUNNING:
            err = w.get("error") or {}
            emit({"type": "tool_event", "tool": tool, "step": w.get("state") or "unknown",
                  "detail": w.get("reason") or (f"{err.get('pos_m', 0) * 100:.0f} cm and "
                                                f"{abs(err.get('yaw_deg', 0)):.0f} deg off")})
            return {"ok": w.get("state") == "success", "state": w.get("state"),
                    "off_cm": round(err.get("pos_m", 0) * 100) if err else None,
                    "off_deg": abs(err.get("yaw_deg", 0)) if err else None,
                    **({"reason": w["reason"]} if w.get("reason") else {})}
        time.sleep(poll_s)


def _executor_execute(body, timeout=20):
    """POST the executor's /execute: one skill, with safe mode ON whatever the page says."""
    import json as _json
    import urllib.error
    import urllib.request

    import manipulation
    req = urllib.request.Request(f"{manipulation.EXECUTOR_URL}/execute",
                                 data=_json.dumps({**body, "robot": "g1", "safe_mode": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            out = _json.loads(exc.read() or b"{}")
        except ValueError:
            out = {}
        return {"ok": False, "error": out.get("error") or f"HTTP {exc.code}", **out}


def _no_such_move(skill):
    """Why `skill` is not a move, in a way the model can act on: when it is a parameter's value
    of a move ("clap" is arm_action's action), it says which move and how to call it. Read off
    the catalog, nothing by hand: the model makes the call (2026-11-12: "Aplaudí" sent skill
    "clap", was told only the list of moves, and gave up)."""
    owners = [(name, pname) for name, spec in command_skills().items()
              for pname, p in spec["params"].items() if skill in (p.get("values") or [])]
    if owners:
        how = " or ".join(f'{{"skill": "{n}", "params": {{"{pn}": "{skill}"}}}}' for n, pn in owners)
        return f"'{skill}' is not a move but a value of a move's parameter: use {how}"
    return f"no move '{skill}': use one of {sorted(command_skills())}"


def tool_command(args, emit, stop, call=None):
    """One of the robot's built-in moves (command_skills), checked the way the command
    interpreter checks its own: unknown names and parameters never reach the robot."""
    import command_common as cc
    call = call or _executor_execute
    skill = str(args.get("skill") or "").strip()
    params = args.get("params") if isinstance(args.get("params"), dict) else {}
    if skill not in command_skills():
        return {"ok": False, "error": _no_such_move(skill)}
    intent = cc.normalize_intent({"skill": skill, "params": params}, robot="g1")
    emit({"type": "tool_event", "tool": "command", "step": skill,
          "detail": ", ".join(f"{k}={v}" for k, v in intent["params"].items()) or "no parameters"})
    out = call({"skill": intent["skill"], "params": intent["params"]})
    return {"ok": bool(out.get("ok")), "skill": skill,
            **({"error": out["error"]} if out.get("error") else {})}


TOOL_FNS = {"look": tool_look, "pick": tool_pick, "place": tool_place,
            "navigate": tool_navigate, "command": tool_command, "search": tool_search,
            "remember": tool_remember}


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class ModelDown(RuntimeError):
    pass


def about_now(notes=None, state=None, now=None):
    """What the vision chat is also told: the time, the robot's notes and its state."""
    import vlm_common as vc
    lines = [f"Now it is {vc._now_text(now, english=True)}."]
    if remembered := vc.remembered(notes):
        lines.append(remembered)
    # All of the state the chat would say for "how are you" (battery, link): an instruction
    # like "go charge if you are low" needs it without asking for it in those words.
    if facts := vc.state_facts(state, "how are you"):
        lines.append("About you: " + "; ".join(facts) + ".")
    return "\n".join(lines) + "\n\n"


def run(instruction, emit, stop=None, llm=llm_json, tools=None, max_steps=MAX_STEPS,
        history=None, notes=None, state=None):
    """See _run. A language model that stops answering ends the run with an error event and a
    plain reply, never an exception out of the stream."""
    def safe_llm(system, user):
        try:
            return llm(system, user)
        except Exception as exc:
            raise ModelDown(f"{type(exc).__name__}: {exc}") from None
    try:
        return _run(instruction, emit, stop, safe_llm, tools, max_steps, history, notes, state)
    except ModelDown as exc:
        emit({"type": "error", "text": f"the language model is not answering ({exc})"[:300]})
        reply = "El modelo de lenguaje no responde. / The language model is not answering."
        emit({"type": "reply", "text": reply})
        return reply


def _run(instruction, emit, stop=None, llm=llm_json, tools=None, max_steps=MAX_STEPS,
         history=None, notes=None, state=None):
    """Carry out `instruction`, emitting every step. Returns the final reply (also emitted).
    `history`: the conversation so far (history_text), sent with every call to the model.

    `llm(system, user)` -> dict and `tools` {name: fn(args, emit, stop)} are injectable, so the
    loop is tested without a model or a robot."""
    stop = stop or threading.Event()
    tools = tools or TOOL_FNS
    emit({"type": "heard", "text": instruction})
    earlier = about_now(notes, state) + history_text(history)
    # Said last, where the reply is written: "I can't clap right now" answered "Aplaudí"
    # (2026-11-12) with the prompt's "in the language of the instruction" far above it.
    import vlm_common as vc
    lang = f"\n(Any reply must be in {'English' if vc.asked_in_english(instruction) else 'Spanish'}.)"
    plan = llm(PLAN_SYSTEM.format(tools=_tools_text()),
               f"{earlier}Instruction now: {instruction}")
    understood = str(plan.get("understood") or instruction)
    steps = [str(s) for s in plan.get("plan") or [] if str(s).strip()]
    emit({"type": "understood", "text": understood})
    if steps:
        emit({"type": "plan", "steps": steps})
    steps_so_far = [f"Instruction: {instruction}", f"Understood: {understood}",
                    f"Plan: {'; '.join(steps) or '(none)'}"]
    for _ in range(max_steps):
        if stop.is_set():
            break
        step = llm(STEP_SYSTEM.format(tools=_tools_text()), earlier + "\n".join(steps_so_far) + lang)
        thought = str(step.get("thought") or "").strip()
        if thought:
            emit({"type": "thought", "text": thought})
        if step.get("reply"):
            reply = str(step["reply"]).strip()
            emit({"type": "reply", "text": reply})
            return reply
        name = step.get("tool")
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        if name not in tools:
            steps_so_far.append(f"Error: there is no tool '{name}'. Use one of {list(tools)} "
                           f"or reply.")
            emit({"type": "error", "text": f"no tool '{name}'"})
            continue
        emit({"type": "tool", "tool": name, "args": args})
        try:
            result = tools[name](args, emit, stop)
        except Exception as exc:                 # the trace says it, the model is told
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        emit({"type": "tool_result", "tool": name, "result": result})
        steps_so_far.append(f"Used {name}({json.dumps(args, ensure_ascii=False)}) -> "
                       f"{json.dumps(result, ensure_ascii=False)}")
    # The last words are the model's too, in the instruction's language; the fixed text is only
    # for a model that gave none, and is in both of the app's languages.
    why = ("the person pressed STOP" if stop.is_set() else
           f"you used all {max_steps} steps without finishing")
    end = llm(STEP_SYSTEM.format(tools=_tools_text()),
              earlier + "\n".join([*steps_so_far, f"You must stop now: {why}. Reply to the person."])
              + lang)
    reply = str(end.get("reply") or "").strip() or (
        "Me detuve. / Stopped." if stop.is_set() else
        "No pude terminarlo. / I could not finish it.")
    emit({"type": "reply", "text": reply})
    return reply
