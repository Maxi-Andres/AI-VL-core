"""Does a message ask the robot to keep a picture or a note? The VLM server decides
(keep_common); here its answers are faked, the parsing and the route are what is tested."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import keep_common as kc  # noqa: E402
import vlm_common as vc  # noqa: E402


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


@pytest.mark.parametrize("answer, kept", [
    ("PICTURE", ("picture", "")),
    ("NOTE: hoy es el cumpleaños de Mateo", ("note", "hoy es el cumpleaños de Mateo")),
    ('NOTE: "the code is 1234"', ("note", "the code is 1234")),
    ("NO", (None, "")),
    ("NOTE:", (None, "")),
])
def test_the_model_s_answer_is_read(monkeypatch, answer, kept):
    said = _Said(answer)
    monkeypatch.setattr(kc.requests, "post", said)
    assert kc.wants_kept("A notar que hoy es el bompeaño de Mateo", "http://vlm") == kept
    assert said.sent[0]["messages"][-1]["content"] == "A notar que hoy es el bompeaño de Mateo"


def test_a_server_that_cannot_answer_is_said_as_such(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(kc.requests, "post", down)
    assert kc.wants_kept("Guardá esto", "http://vlm") is None


def test_the_route(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    import service
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(kc, "wants_kept", lambda text, url: ("note", "el wifi es silk2026"))
    with TestClient(service.app) as c:
        r = c.post("/keep", json={"text": "guardá en tu memoria que el wifi es silk2026"}).json()
        assert r == {"decided": True, "keep": "note", "note": "el wifi es silk2026"}
        monkeypatch.setattr(kc, "wants_kept", lambda text, url: None)
        assert c.post("/keep", json={"text": "x"}).json()["decided"] is False
