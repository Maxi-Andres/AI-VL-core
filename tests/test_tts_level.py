"""Kokoro is brought to Piper's loudness. No model: the leveller is fed a made-up signal.

Every test here names the defect it catches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tts_common


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_a_quiet_voice_comes_out_as_loud_as_piper():
    """Catches: "Santa" ~7.5 dB under Daniela at the same robot volume (2026-10-19)."""
    t = np.arange(24000) / 24000
    quiet = 0.076 * np.sqrt(2) * np.sin(2 * np.pi * 180 * t)       # Kokoro's measured RMS
    out = tts_common._level_like_piper(quiet)
    assert abs(_rms(out) - tts_common.TARGET_RMS) < 0.02


def test_the_gain_never_clips():
    rng = np.random.default_rng(0)
    speechy = rng.laplace(0, 0.05, 48000).clip(-0.7, 0.7)          # rare high peaks, like speech
    out = tts_common._level_like_piper(speechy)
    assert np.max(np.abs(out)) <= 1.0


def test_silence_stays_silent():
    assert np.all(tts_common._level_like_piper(np.zeros(1000)) == 0)


# --------------------------------------------------------------------------- #
# Supertonic (2026-10-19): routed by name, offered only when installed, sent at half rate
# --------------------------------------------------------------------------- #
def test_a_supertonic_voice_goes_to_supertonic(monkeypatch):
    """Catches: "supertonic-M2" handed to Piper, which would fall back to its first voice."""
    monkeypatch.setattr(tts_common, "supertonic_available", lambda: True)
    assert tts_common._pick_engine("supertonic-M2") == "supertonic"
    assert tts_common._pick_engine("em_santa") == "kokoro"


def test_without_its_model_supertonic_is_not_offered_and_falls_back(monkeypatch):
    """Catches: the picker listing voices that answer with an error on a machine without
    the model files."""
    monkeypatch.setattr(tts_common, "supertonic_available", lambda: False)
    monkeypatch.setattr(tts_common, "kokoro_available", lambda: True)
    assert tts_common.supertonic_voices() == []
    assert tts_common._pick_engine("supertonic-M1") == "kokoro"


def test_halving_the_rate_keeps_speech_and_drops_what_would_alias():
    """Catches: a 44.1 kHz voice reaching the robot's linear 16 kHz resampler unfiltered —
    a 15 kHz tone would fold back to an audible 1 kHz."""
    t = np.arange(44100) / 44100
    voice = np.sin(2 * np.pi * 300 * t)
    hiss = np.sin(2 * np.pi * 15000 * t)
    assert _rms(tts_common._halve_rate(voice)[100:-100]) > 0.65    # ~0.707 kept
    assert _rms(tts_common._halve_rate(hiss)[100:-100]) < 0.02
