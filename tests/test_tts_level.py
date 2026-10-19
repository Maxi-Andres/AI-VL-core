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
