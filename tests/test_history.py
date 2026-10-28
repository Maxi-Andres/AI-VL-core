"""The assistant follows the conversation: earlier questions and answers ride on each new one,
as text, within the server's window.

Every test names the defect it catches. No model: the payload sent upstream is captured.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
from fastapi.testclient import TestClient  # noqa: E402

import vlm_common as vc  # noqa: E402


def test_the_turns_go_in_order_before_the_question():
    """Catches: "¿y la de la izquierda?" answered with no idea what it follows."""
    msgs = vc.history_messages([{"question": "¿Qué ves?", "answer": "Una mesa y dos sillas."},
                                {"question": "¿De qué color es la mesa?", "answer": "Marrón."}])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "¿Qué ves?" and msgs[3]["content"] == "Marrón."
    assert all("images" not in m for m in msgs), "old frames would fill the 4096 window"


def test_only_the_newest_turns_that_fit_go():
    """Catches: a long conversation overflowing the window and the frame being cut instead."""
    old = [{"question": f"pregunta {i}", "answer": "x" * 1000} for i in range(10)]
    msgs = vc.history_messages(old, budget=2500)
    questions = [m["content"] for m in msgs if m["role"] == "user"]
    assert questions == ["pregunta 8", "pregunta 9"]


def test_unanswered_turns_are_skipped():
    msgs = vc.history_messages([{"question": "¿Qué ves?", "answer": ""}, {"question": "", "answer": "x"}])
    assert msgs == []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    import service
    sent = []
    monkeypatch.setattr(vc, "_openai_chunks", lambda payload, url, timeout: sent.append(payload) or iter(()))
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(vc, "model_supports_thinking", lambda *a, **k: False)
    import search_common
    monkeypatch.setattr(search_common, "model_says_web", lambda q, url: False)   # never the live server
    with TestClient(service.app) as c:
        c.sent = sent
        yield c


def test_the_service_passes_the_conversation_to_the_model(client):
    r = client.post("/vlm/stream", json={
        "image": "aGVsbG8=", "prompt": "¿Y la silla?",
        "history": [{"question": "¿Qué ves?", "answer": "Una mesa y una silla."}]})
    assert r.status_code == 200
    roles = [m["role"] for m in client.sent[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.sent[0]["messages"][-1]["content"] == "¿Y la silla?"


def test_the_stream_ends_saying_how_full_the_window_is(client, monkeypatch):
    """Catches: no way to tell when the conversation stops fitting and older turns drop."""
    chunks = [{"choices": [{"delta": {"content": "Marrón."}}]},
              {"choices": [], "usage": {"prompt_tokens": 1500, "completion_tokens": 20}}]
    monkeypatch.setattr(vc, "_openai_chunks", lambda payload, url, timeout: iter(chunks))
    monkeypatch.setattr(vc, "llamacpp_window", lambda url=None: 4096)
    turns = [{"question": f"q{i}", "answer": "x" * 2000} for i in range(4)]
    text = client.post("/vlm/stream", json={"image": "aGk=", "prompt": "¿Y la silla?", "history": turns}).text
    answer, mark = text.split(vc.USAGE_MARK)
    assert answer == "Marrón."
    import json
    usage = json.loads(mark.rstrip("\x1e"))
    assert usage == {"turns_sent": 2, "turns_total": 4, "prompt_tokens": 1500,
                     "completion_tokens": 20, "window": 4096}
