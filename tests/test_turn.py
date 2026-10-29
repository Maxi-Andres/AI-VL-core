"""Did the person finish, or stop mid-sentence? The VLM server decides (turn_common); here its
answers are faked: the parsing, the closing marks and a server that cannot answer are tested."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402 - after the sys.path line, like the other tests
import requests  # noqa: E402

import turn_common as tc  # noqa: E402


class _Said:
    def __init__(self, text):
        self.text, self.sent = text, []

    def __call__(self, url, timeout=None, json=None):
        self.sent.append(json)
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


@pytest.mark.parametrize("answer, cut", [("CUT", True), ("DONE", False), ("cut.", True), ("", False)])
def test_the_model_s_answer_is_read(monkeypatch, answer, cut):
    monkeypatch.setattr(tc.requests, "post", _Said(answer))
    assert tc.looks_cut("Acordate de que yo llevo", "http://vlm") is cut


def test_a_closed_question_is_finished_without_asking(monkeypatch):
    # Whisper closes "?" and "!" where the intonation closed.
    said = _Said("CUT")
    monkeypatch.setattr(tc.requests, "post", said)
    assert tc.looks_cut("¿Qué pone el celular?", "http://vlm") is False
    assert said.sent == []


def test_a_server_that_cannot_answer_never_holds(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(tc.requests, "post", down)
    assert tc.looks_cut("Acordate de que", "http://vlm") is False
