#!/usr/bin/env python3
"""
tts_common.py — Core for the text-to-speech (TTS) path.

The browser can already read answers aloud with its own OS voices, but those vary
per device and often sound robotic. This module adds LOCAL **neural** TTS so the
assistant speaks with a nicer, uniform voice on every device and stays fully
offline — the same local-first choice we made for Whisper (STT). Two engines sit
behind one contract (three, counting Supertonic); `service.py` exposes `/speak` (text -> WAV) and `/tts/voices`
(list installed voices), and the engine is chosen per request from the voice name:

  - **Piper** (`es_AR-daniela-high`, …): `<name>.onnx` (+ `.onnx.json`) files in a
    voices directory; lightweight, CPU, ONNX. The installer downloads a few Spanish
    voices; drop more `.onnx` files there to add voices (rhasspy/piper-voices).
  - **Kokoro** (`ef_dora`, `em_alex`, `em_santa`): a single 82M-param ONNX model
    (`kokoro-v1.0.onnx` + `voices-v1.0.bin`) that runs on CPU and sounds a notch
    better than Piper. Phonemization ships in-process via `espeakng-loader` (no
    system espeak-ng needed). The installer downloads the model files.
  - **Supertonic 3** (`supertonic-M1`, …): Supertone's 66M-param ONNX model, CPU,
    ~0.5 s per sentence. Its 10 voices all came back word for word through Whisper
    (2026-10-19, results/tts_samples/new_2026-10-19/); the app offers the five that were
    picked by ear: M1, M2, M3, F2, F4. Model files in supertonic_models/ (the installer
    downloads them; ~380 MB). License: BigScience Open RAIL-M (use-based restrictions,
    commercial use allowed).

Both engines are imported LAZILY so the rest of iacore keeps working when a TTS
package is not installed — exactly like the YOLO/ASR paths. When a voice can't be
served by its engine, synthesize() falls back to whatever engine IS available.

Config via env:
    TTS_VOICES_DIR   folder holding the Piper `<name>.onnx` voices (default:
                     piper_voices/ at the repo root).
    TTS_VOICE        default voice name when the caller doesn't pass one
                     (default "es_AR-daniela-high").
    KOKORO_MODELS_DIR folder holding the Kokoro model files (default:
                     kokoro_models/ at the repo root).
    KOKORO_MODEL / KOKORO_VOICES  explicit paths to the .onnx / .bin (override
                     KOKORO_MODELS_DIR).
    KOKORO_LANG      Kokoro phonemizer language (default "es").
    SUPERTONIC_MODELS_DIR  folder holding the Supertonic 3 files (default:
                     supertonic_models/ at the repo root).
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

KOKORO_MODELS_DIR = os.environ.get("KOKORO_MODELS_DIR") or os.path.join(
    _REPO_ROOT, "kokoro_models"
)
KOKORO_MODEL = os.environ.get("KOKORO_MODEL") or os.path.join(
    KOKORO_MODELS_DIR, "kokoro-v1.0.onnx"
)
KOKORO_VOICES = os.environ.get("KOKORO_VOICES") or os.path.join(
    KOKORO_MODELS_DIR, "voices-v1.0.bin"
)
KOKORO_LANG = os.environ.get("KOKORO_LANG", "es")
# Spanish voices shipped in Kokoro v1.0. Listed as a constant so /tts/voices does
# not have to load the ~310 MB model just to enumerate them.
KOKORO_VOICE_NAMES = ("ef_dora", "em_alex", "em_santa")

SUPERTONIC_MODELS_DIR = os.environ.get("SUPERTONIC_MODELS_DIR") or os.path.join(
    _REPO_ROOT, "supertonic_models"
)
SUPERTONIC_PREFIX = "supertonic-"
# The five of its ten voices that were chosen by ear (2026-10-19).
SUPERTONIC_VOICE_NAMES = tuple(SUPERTONIC_PREFIX + v for v in ("M1", "M2", "M3", "F2", "F4"))


# --------------------------------------------------------------------------- #
# Piper engine (lazy import + cached voices)
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
                "The Piper text-to-speech path needs the 'piper-tts' package, which "
                "is not installed. Install it with:  pip install piper-tts\n"
                "(the browser's own voices work without it)."
            ) from e
    return PiperVoice


def piper_voices():
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
    return bool(piper_voices())


def _model_path(name):
    return os.path.join(TTS_VOICES_DIR, f"{name}.onnx")


def load_voice(name=None):
    """Load (and cache) a Piper voice by name, falling back to the first installed
    one when the requested/default name is missing."""
    name = name or TTS_VOICE
    if not os.path.exists(_model_path(name)):
        avail = piper_voices()
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


def _synthesize_piper(text, voice):
    """Synthesize with Piper. Returns (pcm_bytes, sample_rate).

    Handles both the classic API (`synthesize_stream_raw` yielding raw PCM) and the
    newer chunk API (`synthesize` yielding AudioChunk objects).
    """
    v = load_voice(voice)
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
    return bytes(pcm), (sample_rate or 22050)


# --------------------------------------------------------------------------- #
# Kokoro engine (lazy import + cached model)
# --------------------------------------------------------------------------- #
_KOKORO_CACHE = {}


def _load_kokoro():
    """Load (and cache) the single Kokoro ONNX model. Raises a clear error if the
    package or the model files are missing."""
    try:
        from kokoro_onnx import Kokoro
    except ImportError as e:
        raise ImportError(
            "The Kokoro text-to-speech path needs the 'kokoro-onnx' package, which "
            "is not installed. Install it with:  pip install kokoro-onnx\n"
            "(Piper and the browser's own voices work without it)."
        ) from e
    if not (os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES)):
        raise FileNotFoundError(
            f"Kokoro model files not found in {KOKORO_MODELS_DIR}. Expected "
            f"kokoro-v1.0.onnx + voices-v1.0.bin — run the installer to download them."
        )
    if "instance" not in _KOKORO_CACHE:
        _KOKORO_CACHE["instance"] = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    return _KOKORO_CACHE["instance"]


def kokoro_available():
    """True if kokoro-onnx is importable AND its model files are present."""
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES)


def kokoro_voices():
    """The Spanish Kokoro voices, if the engine is usable (else empty)."""
    return list(KOKORO_VOICE_NAMES) if kokoro_available() else []


# Kokoro speaks ~7.5 dB quieter than Piper (RMS -22 dBFS against -15, peaks -3 to -6 dBFS
# against 0; measured 2026-10-19): on the robot at 80/100, "Santa" sounded clearly lower than
# Daniela at the same setting. Piper normalises its output; Kokoro does not. So Kokoro is
# brought to Piper's level, with a soft limiter above KNEE so the peaks the gain pushes past
# full scale are rounded off instead of clipped.
TARGET_RMS = 0.18
KNEE = 0.8


def _level_like_piper(samples):
    """Float samples in [-1, 1], gained to TARGET_RMS RMS and soft-limited under 1.0."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms < 1e-4:
        return x                                   # silence: nothing to level
    y = x * (TARGET_RMS / rms)
    over = np.abs(y) > KNEE
    y[over] = np.sign(y[over]) * (KNEE + (1 - KNEE) * np.tanh((np.abs(y[over]) - KNEE) / (1 - KNEE)))
    return y


def _synthesize_kokoro(text, voice):
    """Synthesize with Kokoro. Returns (pcm_bytes, sample_rate). Kokoro yields
    float32 samples in [-1, 1] at 24 kHz; convert to 16-bit little-endian PCM."""
    import numpy as np

    k = _load_kokoro()
    name = voice if voice in KOKORO_VOICE_NAMES else KOKORO_VOICE_NAMES[0]
    samples, sample_rate = k.create(text, voice=name, speed=1.0, lang=KOKORO_LANG)
    samples = _level_like_piper(samples)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, int(sample_rate)


# --------------------------------------------------------------------------- #
# Supertonic engine (lazy import + cached model)
# --------------------------------------------------------------------------- #
_SUPERTONIC_CACHE = {}


def _supertonic_files_present():
    return os.path.exists(os.path.join(SUPERTONIC_MODELS_DIR, "onnx", "vocoder.onnx"))


def _load_supertonic():
    """Load (and cache) the Supertonic 3 model. Never downloads: a missing model is an
    error naming the installer, like Kokoro's."""
    try:
        from supertonic import TTS
    except ImportError as e:
        raise ImportError(
            "The Supertonic text-to-speech path needs the 'supertonic' package, which is "
            "not installed. Install it with:  pip install supertonic\n"
            "(Piper, Kokoro and the browser's own voices work without it)."
        ) from e
    if not _supertonic_files_present():
        raise FileNotFoundError(
            f"Supertonic model files not found in {SUPERTONIC_MODELS_DIR} — run the "
            f"installer to download them."
        )
    if "instance" not in _SUPERTONIC_CACHE:
        _SUPERTONIC_CACHE["instance"] = TTS(model="supertonic-3",
                                            model_dir=SUPERTONIC_MODELS_DIR,
                                            auto_download=False)
    return _SUPERTONIC_CACHE["instance"]


def supertonic_available():
    """True if supertonic is importable AND its model files are present."""
    try:
        import supertonic  # noqa: F401
    except ImportError:
        return False
    return _supertonic_files_present()


def supertonic_voices():
    """The Supertonic voices the app offers, if the engine is usable (else empty)."""
    return list(SUPERTONIC_VOICE_NAMES) if supertonic_available() else []


def _halve_rate(samples):
    """Float samples at 2·r down to r: a windowed-sinc low-pass at 0.45·r, then every other
    sample. Supertonic speaks at 44.1 kHz, and the G1's sender resamples to 16 kHz linearly,
    with no filter — the top of a 44.1 kHz voice would fold back as hiss. At 22.05 kHz it
    arrives like a Piper voice, which the sender already handles, in half the bytes."""
    import numpy as np

    n = np.arange(-31, 32)
    taps = 0.45 * np.sinc(0.45 * n) * np.hamming(len(n))
    taps /= taps.sum()
    return np.convolve(samples, taps, mode="same")[::2].astype(np.float32)


def _synthesize_supertonic(text, voice):
    """Synthesize with Supertonic 3 in Spanish. Returns (pcm_bytes, sample_rate). It is
    levelled like Kokoro so a voice change never changes the loudness on the robot."""
    import numpy as np

    tts = _load_supertonic()
    name = voice if voice in SUPERTONIC_VOICE_NAMES else SUPERTONIC_VOICE_NAMES[0]
    style = tts.get_voice_style(name[len(SUPERTONIC_PREFIX):])
    samples, _duration = tts.synthesize(text, voice_style=style, lang="es")
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    rate = int(tts.sample_rate)
    if rate >= 44100:
        samples, rate = _halve_rate(samples), rate // 2
    samples = _level_like_piper(samples)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, rate


# --------------------------------------------------------------------------- #
# Public API (engine-agnostic)
# --------------------------------------------------------------------------- #
def list_voices():
    """All installed neural voices across engines (Piper, Kokoro, Supertonic). The
    names are engine-routable: synthesize() picks the engine from the name."""
    return piper_voices() + kokoro_voices() + supertonic_voices()


def _pcm_to_wav(pcm, sample_rate):
    """Wrap raw 16-bit mono PCM into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _pick_engine(name):
    """Decide which engine serves `name` (already stripped of any prefix), with a
    graceful fallback to whatever engine is actually available."""
    if name.startswith(SUPERTONIC_PREFIX):
        if supertonic_available() or not (piper_available() or kokoro_available()):
            return "supertonic"
        return "kokoro" if kokoro_available() else "piper"
    engine = "kokoro" if name in KOKORO_VOICE_NAMES else "piper"
    if engine == "piper" and not piper_available() and kokoro_available():
        return "kokoro"
    if engine == "kokoro" and not kokoro_available() and piper_available():
        return "piper"
    return engine


def synthesize(text, voice=None):
    """Synthesize `text` into a WAV (16-bit mono) and return {wav, elapsed, ...}.

    The engine is chosen from the voice name: a Kokoro voice (ef_*/em_*) uses
    Kokoro, a "supertonic-*" one Supertonic, anything else uses Piper. An explicit "kokoro:" / "piper:" prefix on the
    voice forces the engine.
    """
    name = voice or TTS_VOICE
    engine = None
    if name.startswith("kokoro:"):
        engine, name = "kokoro", name.split(":", 1)[1]
    elif name.startswith("piper:"):
        engine, name = "piper", name.split(":", 1)[1]
    if engine is None:
        engine = _pick_engine(name)

    t0 = time.perf_counter()
    if engine == "supertonic":
        pcm, sample_rate = _synthesize_supertonic(text, name)
    elif engine == "kokoro":
        pcm, sample_rate = _synthesize_kokoro(text, name)
    else:
        pcm, sample_rate = _synthesize_piper(text, name)
    return {
        "wav": _pcm_to_wav(pcm, sample_rate),
        "elapsed": time.perf_counter() - t0,
        "sample_rate": sample_rate,
    }
