"""What /transcribe asks Whisper for. A fake model records the call: no weights, no GPU.

Every test here names the defect it catches.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asr_common


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, _audio, **kwargs):
        self.kwargs = kwargs
        return iter(()), type("Info", (), {"language": "es"})()


def _call(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(asr_common, "load_model", lambda *a, **k: fake)
    asr_common.transcribe(b"RIFF....", language="es")
    return fake.kwargs


def test_the_robots_name_is_a_hotword(monkeypatch):
    """Catches: "Bender" heard as "vende", "venga", "Déjdenme" through the G1's mic
    (2026-10-19) — the name the voice loop listens for."""
    assert "Bender" in _call(monkeypatch)["hotwords"]


def test_stretches_without_voice_are_skipped(monkeypatch):
    """Catches: the VAD turned off — silence then came back as "¡Suscríbete!", and with the
    hotword as a phantom "Bender" that would wake the robot with nobody talking."""
    assert _call(monkeypatch)["vad_filter"] is True
