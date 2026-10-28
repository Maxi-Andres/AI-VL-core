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
    turns = [{"question": f"q{i}", "answer": "x" * 3000} for i in range(4)]   # 2 fit in 4096
    text = client.post("/vlm/stream", json={"image": "aGk=", "prompt": "¿Y la silla?", "history": turns}).text
    answer, mark = text.split(vc.USAGE_MARK)
    assert answer == "Marrón."
    import json
    usage = json.loads(mark.rstrip("\x1e"))
    assert usage == {"turns_sent": 2, "turns_total": 4, "prompt_tokens": 1500,
                     "completion_tokens": 20, "window": 4096, "answer_room": vc.ANSWER_TOKENS}


def test_the_turns_fill_the_window_not_a_fixed_cap():
    """Catches: the first questions forgotten with 1.3k of 4.1k tokens still free (a fixed
    5000 characters, 2026-10-28)."""
    system = vc.FREE_PROMPT_SYSTEM
    budget = vc.history_budget(4096, system, "¿y el de la izquierda?", frame=1100)
    assert budget > 7000
    used = 1100 + (len(system) + budget) / vc.CHARS_PER_TOKEN + vc.ANSWER_TOKENS
    assert used <= 4096, "the frame, the turns and the answer must still fit"


def test_web_results_and_no_frame_move_the_room():
    results = "x" * 3500
    with_results = vc.history_budget(4096, results, "q", frame=0)
    plain = vc.history_budget(4096, "s", "q", frame=1100)
    assert with_results < plain + 1100 * vc.CHARS_PER_TOKEN
    assert vc.history_budget(4096, "x" * 20000, "q", frame=1100) == 0
    assert vc.history_budget(None, "s", "q", frame=1100) == vc.HISTORY_CHARS, "no window: the old cap"


def test_a_big_frame_leaves_less_room():
    """Catches: a 1600x1200 photo costing 1911 tokens, not the 1100 assumed, and the answer left
    with ~330 tokens (2026-10-28)."""
    import base64
    import io

    from PIL import Image

    def jpeg(w, h):
        buf = io.BytesIO()
        Image.new("RGB", (w, h)).save(buf, "JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    assert vc.frame_tokens(jpeg(1600, 1200)) >= 1900
    assert 1024 <= vc.frame_tokens(jpeg(615, 410)) < 1100, "the server's floor"
    assert vc.frame_tokens("bm90IGFuIGltYWdl") == vc.FRAME_TOKENS
