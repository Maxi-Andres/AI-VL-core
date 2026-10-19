#!/usr/bin/env python3
"""
asr_common.py — Core for the speech-to-text (ASR) path.

iacore already owns the two detection paths (vlm_common.py, yolo_common.py); this
module adds a THIRD, unrelated capability used by the browser UI: transcribing a
short dictated audio clip into text so the user can speak the VLM "free prompt"
instead of typing it. The flow mirrors the rest of iacore — the backend gateway
relays the audio bytes here over HTTP, and `service.py` exposes a `/transcribe`
endpoint that calls `transcribe()` below.

It runs OpenAI Whisper locally via **faster-whisper** (CTranslate2), so nothing
leaves the machine (consistent with the "everything runs local at runtime"
property of the ecosystem). Defaults are tuned to NOT fight the VLM for the 8 GB
GPU: a small `base` model on the CPU with `int8` weights transcribes a few seconds
of dictation in well under a second. Everything is env-configurable so the model
can be moved to the GPU.

On the GPU, CTranslate2 4.x needs CUDA 12's cuBLAS (`libcublas.so.12`), while this
venv's torch brings CUDA 13: without help it fails with "Library libcublas.so.12 is
not found". `_preload_cuda12()` loads it first from the nvidia-cublas-cu12 wheel if
installed, else from the copy Ollama ships (/usr/local/lib/ollama/cuda_v12) — already
on every machine install.sh set up, and what linux/vlm-server.sh runs on.

`faster_whisper` is imported LAZILY (inside the function that needs it) so the rest
of iacore keeps working even when faster-whisper is not installed — exactly like
`yolo_common` does with `ultralytics`.

Config via env:
    ASR_MODEL         Whisper model size/name (default "base"; run.sh uses "small").
    ASR_DEVICE        "cpu" (default) or "cuda" (run.sh's; falls back to the CPU).
    ASR_COMPUTE_TYPE  CTranslate2 compute type (default "int8"; run.sh: "int8_float16").
    ASR_CUDA12_DIR    a folder with CUDA 12's libcublas(.Lt).so.12, tried first.
    ASR_HOTWORDS      names to spell right, comma-separated (default "Bender", the G1).
    ASR_VAD           "0" turns off skipping the stretches without voice (default on).
    ASR_LANGUAGE      force a language (e.g. "es"/"en"); unset => auto-detect.
"""
import io
import logging
import os
import time

# Model config from the environment. Read once at import; the process is restarted to
# change them (like the rest of iacore). linux/run.sh exports small / cuda /
# int8_float16, the winner of a 2026-10-16 benchmark on 30 Argentine speakers
# (OpenSLR 61): 2.6 % WER in 0.16 s with 482 MB of VRAM, where these CPU defaults give
# 8.0 % in 0.47 s and large-v3-turbo 2.2-2.6 % in 0.25 s with 1.3-2.2 GB.
ASR_MODEL = os.environ.get("ASR_MODEL", "base")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE") or None
# Names Whisper must spell right: the G1 is "Bender", which it heard as "vende", "venga",
# "Déjdenme" and "Bengal" through the robot's own mic (2026-10-19). Comma-separated; "" for none.
ASR_HOTWORDS = os.environ.get("ASR_HOTWORDS", "Bender").replace(",", " ").strip() or None
# Skip the stretches with no voice (faster-whisper's Silero VAD). Without it, silence and fan
# noise came back as "¡Suscríbete!", and with the hotword alone as a phantom "Bender" — which
# would wake the robot with nobody talking. Cost on 30 Argentine speakers: 1.1 -> 1.5 % WER
# (one word), same speed. "0" turns it off.
ASR_VAD = os.environ.get("ASR_VAD", "1") != "0"

# Where a CUDA 12 cuBLAS may live, in order (see the module docstring).
CUDA12_DIRS = [os.environ["ASR_CUDA12_DIR"]] if os.environ.get("ASR_CUDA12_DIR") else []
CUDA12_DIRS += ["/usr/local/lib/ollama/cuda_v12"]


def _preload_cuda12():
    """dlopen CUDA 12's cuBLAS globally, so CTranslate2 finds it by soname.

    Returns the folder it came from, or None when there is none (CTranslate2 then
    raises its own error, which says what is missing).
    """
    import ctypes
    dirs = list(CUDA12_DIRS)
    try:
        import nvidia.cublas.lib as wheel  # nvidia-cublas-cu12, if someone installed it
        dirs.insert(0, os.path.dirname(wheel.__file__))
    except ImportError:
        pass
    for d in dirs:
        lt, blas = (os.path.join(d, n) for n in ("libcublasLt.so.12", "libcublas.so.12"))
        if os.path.exists(lt) and os.path.exists(blas):
            ctypes.CDLL(lt, mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL(blas, mode=ctypes.RTLD_GLOBAL)
            return d
    return None


# --------------------------------------------------------------------------- #
# Lazy faster-whisper import + cached model
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}


def _load_whisper_class():
    """Import and return the faster-whisper `WhisperModel` class.

    Raises a clear, actionable error if the package is missing — it is the only
    extra dependency the ASR path needs (the VLM/YOLO paths do NOT need it).
    """
    try:
        from faster_whisper import WhisperModel  # heavy import; only when used
    except ImportError as e:
        raise ImportError(
            "The speech-to-text path needs the 'faster-whisper' package, which is "
            "not installed. Install it with:  pip install faster-whisper\n"
            "(the VLM/YOLO paths do NOT need it)."
        ) from e
    return WhisperModel


def load_model(model=None, device=None, compute_type=None):
    """Load (and cache) a WhisperModel keyed by (model, device, compute_type).

    The first load of a model name downloads its weights from HuggingFace into the
    faster-whisper cache (~150 MB for `base`); afterwards it is cached in-process so
    repeated transcriptions never reload it.
    """
    model = model or ASR_MODEL
    device = device or ASR_DEVICE
    compute_type = compute_type or ASR_COMPUTE_TYPE
    key = (model, device, compute_type)
    if key not in _MODEL_CACHE:
        WhisperModel = _load_whisper_class()
        if device == "cuda":
            _preload_cuda12()
            try:
                m = WhisperModel(model, device="cuda", compute_type=compute_type)
                # A missing cuBLAS only shows on the first matrix product, not at load:
                # run the encoder once on half a second of silence to find out now.
                import numpy as np
                list(m.transcribe(np.zeros(8000, np.float32), language="en")[0])
                _MODEL_CACHE[key] = m
            except (RuntimeError, ValueError) as e:
                # No GPU, no CUDA 12 cuBLAS, no room: dictation still works, ~7x slower
                # (small: 0.16 s on the GPU, 1.1 s on the CPU, 2026-10-16).
                logging.getLogger("uvicorn.error").warning(
                    "Whisper on the GPU failed (%s); using the CPU instead.", e)
                _MODEL_CACHE[key] = WhisperModel(model, device="cpu", compute_type="int8")
        else:
            _MODEL_CACHE[key] = WhisperModel(model, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def whisper_available():
    """True if 'faster-whisper' can be imported (so callers can hint at the install)."""
    try:
        _load_whisper_class()
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def transcribe(data, language=None, translate=False):
    """Transcribe raw audio BYTES into text and return a small result dict.

    `data`      : the audio file bytes exactly as the browser recorded them
                  (webm/opus on Chrome/Android, mp4/aac on iOS Safari). faster-whisper
                  decodes either via its bundled PyAV/ffmpeg — no separate ffmpeg
                  binary is required.
    `language`  : force a language code (e.g. "es"/"en"); None => auto-detect
                  (falls back to ASR_LANGUAGE from the env when set).
    `translate` : when True, use Whisper's translate task (any language -> English).

    Returns: {text, language, elapsed} (elapsed in wall seconds). Raises on a genuine
    decode/inference error (the caller turns it into an HTTP error).
    """
    model = load_model()
    lang = language or ASR_LANGUAGE
    task = "translate" if translate else "transcribe"
    t0 = time.perf_counter()
    # transcribe() accepts a binary file-like object; wrap the bytes in BytesIO so
    # nothing touches disk. `segments` is a generator that runs the model lazily as
    # it is consumed, so joining it is what actually does the work.
    segments, info = model.transcribe(io.BytesIO(data), language=lang, task=task,
                                      hotwords=ASR_HOTWORDS, vad_filter=ASR_VAD)
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.perf_counter() - t0
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "elapsed": elapsed,
    }
