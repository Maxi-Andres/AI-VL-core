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
"""
from __future__ import annotations

import json
import threading
import time

from vlm_common import LLAMACPP_HOST, extract_json, stream_chat

MAX_STEPS = 6
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
}


def _tools_text():
    return "\n".join(f"- {name}({', '.join(f'{a}: {d}' for a, d in t['args'].items())}): "
                     f"{t['does']}" for name, t in TOOLS.items())


PLAN_SYSTEM = (
    "You control a Unitree G1 humanoid robot standing at a table. You can use these tools:\n"
    "{tools}\n"
    "Read the instruction and answer ONLY JSON: {{\"understood\": what is being asked, in one "
    "short line, \"plan\": [the steps, each a few words], \"needs_robot\": true or false}}. "
    "If it is only a question or a chat, needs_robot is false and the plan may be empty. "
    "Write understood and the plan in the language of the instruction.")

STEP_SYSTEM = (
    "You control a Unitree G1 humanoid robot standing at a table. Tools:\n"
    "{tools}\n"
    "You are carrying out the instruction step by step. Given what has happened so far, answer "
    "ONLY JSON, one of:\n"
    "{{\"thought\": why, \"tool\": a tool name, \"args\": {{...}}}} to use a tool, or\n"
    "{{\"thought\": why, \"reply\": what to tell the person}} when it is done, impossible, or "
    "only needed words. Never claim something worked unless a tool result says so. Reply in the "
    "language of the instruction, short and plain.")


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
def tool_look(args, emit, stop, url=LLAMACPP_HOST):
    import base64

    import cv2

    import locate_object as lo
    question = str(args.get("question") or "What do you see?")
    frame = lo.grab_color()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return {"ok": False, "error": "no picture"}
    b64 = base64.b64encode(buf.tobytes()).decode()
    payload = {"model": "", "stream": True,
               "options": {"temperature": 0.1, "num_predict": 200},
               "messages": [{"role": "user", "content": question, "images": [b64]}]}
    content, *_ = stream_chat(payload, url=url, timeout=60)
    return {"ok": True, "answer": content.strip()}


def tool_pick(args, emit, stop, start=None, poll_s=0.5):
    """Run a manipulation.PickAction to its end, every step of it emitted as a tool event.
    `stop` (the STOP button) cancels it; the arm then goes home before this returns."""
    import manipulation
    start = start or manipulation.start_pick
    obj = str(args.get("object") or "").strip()
    if not obj:
        return {"ok": False, "error": "no object named"}
    try:
        act = start(obj)
    except manipulation.ActionError as exc:
        return {"ok": False, "error": str(exc)}
    seen = 0
    while True:
        if stop.is_set():
            act.cancel()
        snap = act.snapshot(seen)
        for ev in snap["events"]:
            emit({"type": "tool_event", "tool": "pick", "step": ev["step"],
                  "detail": ev["detail"]})
        seen += len(snap["events"])
        if snap["state"] != "running" and not act.thread.is_alive():
            return {"ok": snap["state"] == "success", "state": snap["state"],
                    "held": bool((snap.get("result") or {}).get("held")),
                    "last": snap["events"][-1]["detail"] if snap["events"] else ""}
        time.sleep(poll_s)


TOOL_FNS = {"look": tool_look, "pick": tool_pick}


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class ModelDown(RuntimeError):
    pass


def run(instruction, emit, stop=None, llm=llm_json, tools=None, max_steps=MAX_STEPS):
    """See _run. A language model that stops answering ends the run with an error event and a
    plain reply, never an exception out of the stream."""
    def safe_llm(system, user):
        try:
            return llm(system, user)
        except Exception as exc:
            raise ModelDown(f"{type(exc).__name__}: {exc}") from None
    try:
        return _run(instruction, emit, stop, safe_llm, tools, max_steps)
    except ModelDown as exc:
        emit({"type": "error", "text": f"the language model is not answering ({exc})"[:300]})
        reply = "El modelo de lenguaje no responde. / The language model is not answering."
        emit({"type": "reply", "text": reply})
        return reply


def _run(instruction, emit, stop=None, llm=llm_json, tools=None, max_steps=MAX_STEPS):
    """Carry out `instruction`, emitting every step. Returns the final reply (also emitted).

    `llm(system, user)` -> dict and `tools` {name: fn(args, emit, stop)} are injectable, so the
    loop is tested without a model or a robot."""
    stop = stop or threading.Event()
    tools = tools or TOOL_FNS
    emit({"type": "heard", "text": instruction})
    plan = llm(PLAN_SYSTEM.format(tools=_tools_text()), instruction)
    understood = str(plan.get("understood") or instruction)
    steps = [str(s) for s in plan.get("plan") or [] if str(s).strip()]
    emit({"type": "understood", "text": understood})
    if steps:
        emit({"type": "plan", "steps": steps})
    history = [f"Instruction: {instruction}", f"Understood: {understood}",
               f"Plan: {'; '.join(steps) or '(none)'}"]
    for _ in range(max_steps):
        if stop.is_set():
            break
        step = llm(STEP_SYSTEM.format(tools=_tools_text()), "\n".join(history))
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
            history.append(f"Error: there is no tool '{name}'. Use one of {list(tools)} "
                           f"or reply.")
            emit({"type": "error", "text": f"no tool '{name}'"})
            continue
        emit({"type": "tool", "tool": name, "args": args})
        try:
            result = tools[name](args, emit, stop)
        except Exception as exc:                 # the trace says it, the model is told
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        emit({"type": "tool_result", "tool": name, "result": result})
        history.append(f"Used {name}({json.dumps(args, ensure_ascii=False)}) -> "
                       f"{json.dumps(result, ensure_ascii=False)}")
    # The last words are the model's too, in the instruction's language; the fixed text is only
    # for a model that gave none, and is in both of the app's languages.
    why = ("the person pressed STOP" if stop.is_set() else
           f"you used all {max_steps} steps without finishing")
    end = llm(STEP_SYSTEM.format(tools=_tools_text()),
              "\n".join([*history, f"You must stop now: {why}. Reply to the person."]))
    reply = str(end.get("reply") or "").strip() or (
        "Me detuve. / Stopped." if stop.is_set() else
        "No pude terminarlo. / I could not finish it.")
    emit({"type": "reply", "text": reply})
    return reply
