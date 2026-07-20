#!/usr/bin/env python3
"""
service.py — iacore's HTTP inference service (one of three independent apps).

iacore is the inference CORE: it owns the YOLO detector and the Ollama VLM client
(src/yolo_common.py, src/vlm_common.py) and the heavy deps (ultralytics/torch).
The CLI (menu.py) still drives it locally; THIS file additionally exposes the same
detection over the network so the other two apps can reach it BY PORT — never by
file path. The three apps:

    frontend (browser UI)  ->  backend (gateway)  ->  iacore (this service)

Each can run on a different machine. The backend talks to this service over HTTP;
the browser never talks to iacore directly. Endpoints:

    GET  /health             liveness probe.
    GET  /options            models / scopes / variants / defaults (from config.json).
    GET  /classes?model=     class names a YOLO model can detect.
    POST /detect             raw image bytes in body + params -> {objects, ...}.
    POST /vlm                {image(base64), scope, variant, model} -> VLM JSON.
    POST /vlm/stream         {image(base64), prompt, model} -> plain text, STREAMED
                             (free-prompt answer, token by token, for spoken replies).
    POST /command            {text, image?, model} -> {skill, params, say, ...}
                             (Unitree G1 command interpreter: speech text -> skill JSON).
    GET  /skills             the G1 skill catalog the interpreter can emit.
    POST /transcribe         raw audio bytes in body -> {text, ...} (speech-to-text).
    GET  /tts/voices         installed Piper voices (for the UI voice picker).
    POST /speak              {text, voice} -> WAV audio (neural text-to-speech).

Run (from the iacore repo root, venv active):
    uvicorn service:app --host 0.0.0.0 --port 8001
Config via env: IACORE_PORT (informational), OLLAMA_URL (defaults to config.json).
"""
import base64
import io
import os
import sys

# The detection logic lives in src/ within THIS repo (a local file import, not a
# cross-app dependency). Put it on the path like menu.py does.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from PIL import Image

import yolo_common
import vlm_common
import command_common
import asr_common
import tts_common

app = FastAPI(title="iacore — inference service")

# The backend is the normal caller, but allow any origin: this service is meant to
# sit on an internal network behind the backend gateway anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG = yolo_common.load_config()
OLLAMA_URL = os.environ.get("OLLAMA_URL", CFG.get("url", vlm_common.OLLAMA_HOST))


# --------------------------------------------------------------------------- #
# Request models. Typed bodies validate at the edge (malformed input -> 422 with
# a clear message) instead of raising deep inside the handler. Defaults come from
# config.json (loaded once into CFG at startup), matching the CLI's behavior.
# --------------------------------------------------------------------------- #
class VlmRequest(BaseModel):
    image: str = ""
    model: str | None = None
    scope: str = CFG.get("scope", "industrial")
    variant: str | None = CFG.get("variant")
    max_tokens: int = CFG.get("max_tokens", 8192)
    num_ctx: int = CFG.get("num_ctx", 16384)
    think: bool = CFG.get("think", True)
    prompt: str | None = None


class VlmStreamRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: str | None = None
    max_tokens: int = CFG.get("max_tokens", 8192)
    num_ctx: int = CFG.get("num_ctx", 16384)


class SpeakRequest(BaseModel):
    text: str = ""
    voice: str | None = None


class CommandRequest(BaseModel):
    text: str = ""                       # transcribed spoken command
    image: str = ""                      # optional base64 frame (for future vision skills)
    model: str | None = None
    num_ctx: int = CFG.get("num_ctx", 16384)
    max_tokens: int = 1024               # a skill JSON is tiny; cap output low for speed


def _decode(data):
    """Decode JPEG/PNG bytes into an RGB PIL image (what ultralytics/VLM expect)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def _list_ollama_models():
    """Installed Ollama models via /api/tags; [config default] on failure."""
    import requests
    try:
        host = vlm_common.host_of(OLLAMA_URL)
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return [CFG.get("model", "qwen3-vl:4b")]


@app.get("/health")
def health():
    return {"status": "ok", "service": "iacore"}


@app.get("/options")
def options():
    """Everything the UI needs to populate its controls (proxied by the backend)."""
    scopes = {
        s: {
            "label": vlm_common.SCOPE_LABELS[s],
            "variants": list(vlm_common.PROMPT_VARIANTS[s]),
        }
        for s in vlm_common.PROMPT_VARIANTS
    }
    return {
        "yolo_models": yolo_common.list_models(),
        "vlm_models": _list_ollama_models(),
        "scopes": scopes,
        "defaults": {
            "yolo_model": CFG.get("yolo_model", "yolov8n.pt"),
            "conf": CFG.get("yolo_conf", 0.25),
            "imgsz": CFG.get("yolo_imgsz", 640),
            "classes": CFG.get("yolo_classes", []),
            "vlm_model": CFG.get("model", "qwen3-vl:4b"),
            "scope": CFG.get("scope", "industrial"),
            "variant": CFG.get("variant", "v1_original"),
        },
    }


@app.get("/classes")
def classes(model: str = ""):
    model = model or CFG.get("yolo_model", "yolov8n.pt")
    return {"model": model, "classes": yolo_common.class_names(model)}


@app.post("/detect")
async def detect(
    request: Request,
    model: str = Query(None),
    conf: float = Query(None),
    imgsz: int = Query(None),
    classes: str = Query(""),
):
    """Run YOLO on one frame. The image is the raw request body (JPEG/PNG bytes);
    detection params come as query string. Returns the {objects, ...} contract.
    """
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty body (expected image bytes)"}, status_code=400)
    try:
        img = _decode(data)
    except Exception as e:
        return JSONResponse({"error": f"bad image: {e}"}, status_code=400)
    cls = [c for c in classes.split(",") if c] or None
    # Offload the blocking inference to a thread so the event loop stays free.
    res = await run_in_threadpool(
        yolo_common.detect,
        model or CFG.get("yolo_model", "yolov8n.pt"),
        img,
        conf=conf if conf is not None else CFG.get("yolo_conf", 0.25),
        imgsz=imgsz if imgsz is not None else CFG.get("yolo_imgsz", 640),
        classes=cls,
    )
    return {
        "objects": res["objects"],
        "n": res["n"],
        "elapsed_ms": round(res["elapsed"] * 1000, 1),
        "speed": res["speed"],
    }


@app.post("/vlm")
async def vlm(req: VlmRequest):
    """Run the Ollama VLM on one frame. Slow (seconds) -> plain request/response."""
    b64 = req.image
    if "," in b64:                       # tolerate a data-URI prefix
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
        size = _decode(raw).size         # (width, height) for bbox normalization
    except Exception as e:
        return JSONResponse({"error": f"bad image: {e}"}, status_code=400)

    model = req.model or CFG.get("model", "qwen3-vl:4b")

    def _run():
        return vlm_common.query_vlm(
            b64, model,
            scope=req.scope,
            variant=req.variant,
            max_tokens=req.max_tokens,
            num_ctx=req.num_ctx,
            think=req.think,
            prompt=req.prompt,
            url=OLLAMA_URL,
            size=size,
        )

    try:
        res = await run_in_threadpool(_run)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

    return {
        "model": model,
        "ok": res["ok"],
        "parsed": res["parsed"],
        "content": res["content"],
        "reasoning": res["reasoning"],
        "elapsed_ms": round(res["elapsed"] * 1000, 1),
        "finish_reason": res["finish_reason"],
        "did_think": res["did_think"],
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    language: str = Query(None),
    translate: bool = Query(False),
):
    """Transcribe one dictated audio clip. The audio is the raw request body
    (webm/opus or mp4/aac bytes as the browser recorded it); optional params come
    as query string. Runs Whisper locally via faster-whisper. Slow enough (model +
    decode) to offload to a thread so the event loop stays free."""
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty body (expected audio bytes)"}, status_code=400)
    try:
        res = await run_in_threadpool(
            asr_common.transcribe, data, language=language or None, translate=translate
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return {
        "text": res["text"],
        "language": res["language"],
        "elapsed_ms": round(res["elapsed"] * 1000, 1),
    }


@app.post("/vlm/stream")
async def vlm_stream(req: VlmStreamRequest):
    """Stream a free-prompt answer as plain UTF-8 text, chunk by chunk, so the
    browser can display it live and speak it sentence by sentence as it arrives.
    Only the free-prompt (plain-text) case is streamed; detection stays on /vlm."""
    b64 = req.image
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    prompt = req.prompt.strip()
    if not prompt:
        return JSONResponse({"error": "missing 'prompt'"}, status_code=400)
    try:
        base64.b64decode(b64)
    except Exception as e:
        return JSONResponse({"error": f"bad image: {e}"}, status_code=400)

    model = req.model or CFG.get("model", "qwen3-vl:4b")

    def gen():
        try:
            for piece in vlm_common.query_vlm_stream(
                b64, model, prompt,
                max_tokens=req.max_tokens,
                num_ctx=req.num_ctx,
                url=OLLAMA_URL,
            ):
                yield piece
        except Exception as e:
            # Surface the error inline so the client sees why the stream stopped.
            yield f"\n[error] {e}"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/command")
async def command(req: CommandRequest):
    """Interpret a spoken/typed command into a Unitree G1 skill JSON.

    This is the language brain (ROBOT_CONTROL.md Phase 1): it maps the transcribed
    text to ONE skill + params over the fixed catalog in command_common.SKILLS
    (locomotion, posture, gestures, arm presets — every action the SDK ships). It
    does NOT move the robot; a downstream executor turns the skill into SDK calls.
    Reuses the Ollama client, so it's slow-ish (seconds) -> offload to a thread."""
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    b64 = req.image
    if "," in b64:                       # tolerate a data-URI prefix
        b64 = b64.split(",", 1)[1]

    model = req.model or CFG.get("model", "qwen3-vl:4b")

    def _run():
        return command_common.interpret(
            text, model,
            image_b64=b64 or None,
            url=OLLAMA_URL,
            num_ctx=req.num_ctx,
            max_tokens=req.max_tokens,
        )

    try:
        res = await run_in_threadpool(_run)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

    return {
        "model": model,
        "ok": res["ok"],
        "skill": res["skill"],
        "params": res["params"],
        "say": res["say"],
        "understood": res["understood"],
        "content": res["content"],
        "elapsed_ms": res["elapsed_ms"],
    }


@app.get("/skills")
def skills():
    """The G1 skill catalog the /command interpreter can emit (single source of
    truth in command_common.SKILLS). Lets the UI/executor discover skills + params
    without hard-coding them."""
    return {
        "skills": {
            name: {"desc": s["desc"], "params": s["params"]}
            for name, s in command_common.SKILLS.items()
        },
        "speed_presets": command_common.SPEED_PRESETS,
        "arm_actions": command_common.ARM_ACTION_IDS,
    }


@app.get("/tts/voices")
def tts_voices():
    """List installed Piper voices so the UI can offer a voice picker."""
    return {"voices": tts_common.list_voices(), "default": tts_common.TTS_VOICE}


@app.post("/speak")
async def speak(req: SpeakRequest):
    """Neural text-to-speech via Piper. Returns a WAV the browser plays. Slow-ish
    (synthesis) -> offload to a thread so the event loop stays free."""
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    try:
        res = await run_in_threadpool(tts_common.synthesize, text, req.voice)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return Response(content=res["wav"], media_type="audio/wav")
