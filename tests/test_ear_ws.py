"""WS /ws/ear: the robot's mic stream in, "Bender" events out, mute honoured. Speech-to-text is
faked (no model); the audio is made up at the G1's measured levels (see tests/test_ear.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
from fastapi.testclient import TestClient  # noqa: E402
from test_ear import noise, tone  # noqa: E402

import service  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    said = iter(["Bender, ¿qué ves?", "otra cosa"])
    monkeypatch.setattr(service.asr_common, "transcribe",
                        lambda wav, language=None: {"text": next(said, "")})
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    with TestClient(service.app) as c:
        yield c


def _send(ws, pcm):
    for i in range(0, len(pcm), 5120):
        ws.send_bytes(pcm[i:i + 5120])


def test_a_spoken_command_comes_back_as_events(client):
    with client.websocket_connect("/ws/ear") as ws:
        _send(ws, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
        events = [ws.receive_json() for _ in range(3)]
    assert [e["type"] for e in events] == ["speech", "heard", "command"]
    assert events[-1]["text"] == "qué ves"


def test_muted_audio_is_never_transcribed(client, monkeypatch):
    """Catches: the robot answering its own voice — the browser mutes the ear while it speaks."""
    calls = []
    monkeypatch.setattr(service.asr_common, "transcribe",
                        lambda wav, language=None: calls.append(1) or {"text": "Bender, eco"})
    with client.websocket_connect("/ws/ear") as ws:
        ws.send_text(json.dumps({"type": "mute", "on": True}))
        _send(ws, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
        ws.send_text("not json")                    # ignored, not fatal
    assert calls == []
