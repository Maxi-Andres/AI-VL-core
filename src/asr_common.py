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
can be moved to the GPU later.

`faster_whisper` is imported LAZILY (inside the function that needs it) so the rest
of iacore keeps working even when faster-whisper is not installed — exactly like
`yolo_common` does with `ultralytics`.

Config via env:
    ASR_MODEL         Whisper model size/name (default "base"). e.g. tiny/base/small.
    ASR_DEVICE        "cpu" (default) or "cuda".
    ASR_COMPUTE_TYPE  CTranslate2 compute type (default "int8"; e.g. "float16" on GPU).
    ASR_LANGUAGE      force a language (e.g. "es"/"en"); unset => auto-detect.
"""
import io
import os
import time

# Model config from the environment (with sensible, GPU-friendly defaults). Read
# once at import; the process is restarted to change them (like the rest of iacore).
ASR_MODEL = os.environ.get("ASR_MODEL", "base")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE") or None


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
    segments, info = model.transcribe(io.BytesIO(data), language=lang, task=task)
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.perf_counter() - t0
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "elapsed": elapsed,
    }
