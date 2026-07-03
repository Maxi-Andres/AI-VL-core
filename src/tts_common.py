#!/usr/bin/env python3
"""
tts_common.py — Core for the text-to-speech (TTS) path.

The browser can already read answers aloud with its own OS voices, but those vary
per device and often sound robotic. This module adds a local **neural** TTS via
**Piper** (CTranslate/ONNX voices), so the assistant speaks with a nicer, uniform
voice on every device and stays fully offline — the same local-first choice we
made for Whisper (STT). The backend gateway relays the request here and
`service.py` exposes `/speak` (text -> WAV) and `/tts/voices` (list installed
voices).

Voice models are plain `<name>.onnx` (+ `<name>.onnx.json`) files living in a
voices directory; the install script downloads a couple of Spanish voices. Drop
more `.onnx` files there to add voices (browse rhasspy/piper-voices on
HuggingFace). `piper` is imported LAZILY so the rest of iacore keeps working when
piper-tts is not installed — exactly like the YOLO/ASR paths.

Config via env:
    TTS_VOICES_DIR   folder holding the `<name>.onnx` voices (default: piper_voices/
                     at the repo root).
    TTS_VOICE        default voice name when the caller doesn't pass one
                     (default "es_AR-daniela-high").
"""
import io
import os
import time
import wave

# Repo root = parent of this src/ folder (paths resolve relative to the repo, not src/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TTS_VOICES_DIR = os.environ.get("TTS_VOICES_DIR") or os.path.join(
    _REPO_ROOT, "piper_voices"
)
TTS_VOICE = os.environ.get("TTS_VOICE", "es_AR-daniela-high")


# --------------------------------------------------------------------------- #
# Lazy piper import + cached voices
# --------------------------------------------------------------------------- #
_VOICE_CACHE = {}


def _load_piper_class():
    """Import and return piper's `PiperVoice` class (tolerates its module moves)."""
    try:
        from piper import PiperVoice
    except ImportError:
        try:
            from piper.voice import PiperVoice
        except ImportError as e:
            raise ImportError(
                "The text-to-speech path needs the 'piper-tts' package, which is "
                "not installed. Install it with:  pip install piper-tts\n"
                "(the browser's own voices work without it)."
            ) from e
    return PiperVoice


def list_voices():
    """Names of installed Piper voices (the `<name>.onnx` files in the voices dir)."""
    if not os.path.isdir(TTS_VOICES_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(TTS_VOICES_DIR)
        if f.lower().endswith(".onnx")
    )


def piper_available():
    """True if piper is importable AND at least one voice model is present."""
    try:
        _load_piper_class()
    except ImportError:
        return False
    return bool(list_voices())


def _model_path(name):
    return os.path.join(TTS_VOICES_DIR, f"{name}.onnx")


def load_voice(name=None):
    """Load (and cache) a Piper voice by name, falling back to the first installed
    one when the requested/default name is missing."""
    name = name or TTS_VOICE
    if not os.path.exists(_model_path(name)):
        avail = list_voices()
        if not avail:
            raise FileNotFoundError(
                f"No Piper voice models found in {TTS_VOICES_DIR}. Add a "
                f"<name>.onnx (+ .onnx.json) or run the installer."
            )
        name = avail[0]
    if name not in _VOICE_CACHE:
        PiperVoice = _load_piper_class()
        _VOICE_CACHE[name] = PiperVoice.load(_model_path(name))
    return _VOICE_CACHE[name]


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def synthesize(text, voice=None):
    """Synthesize `text` into a WAV (16-bit mono) and return {wav, elapsed, ...}.

    Handles both the classic piper API (`synthesize_stream_raw` yielding raw PCM)
    and the newer chunk API (`synthesize` yielding AudioChunk objects).
    """
    v = load_voice(voice)
    t0 = time.perf_counter()
    pcm = bytearray()
    sample_rate = getattr(getattr(v, "config", None), "sample_rate", None)

    if hasattr(v, "synthesize_stream_raw"):
        for chunk in v.synthesize_stream_raw(text):
            pcm += chunk
    else:
        # Newer piper: synthesize() yields AudioChunk with int16 PCM bytes.
        for chunk in v.synthesize(text):
            data = getattr(chunk, "audio_int16_bytes", None)
            if data is None:
                data = bytes(getattr(chunk, "audio", b""))
            pcm += data
            sample_rate = getattr(chunk, "sample_rate", sample_rate)

    sample_rate = sample_rate or 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    return {
        "wav": buf.getvalue(),
        "elapsed": time.perf_counter() - t0,
        "sample_rate": sample_rate,
    }
