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
    ("SEEN: la medida del tablero", ("seen", "la medida del tablero")),
    ("ASK", ("ask", "")),
    # One word is not a fact, it points at one: the picture has it ("acordate de esto").
    ("NOTE: esto", ("seen", "esto")),
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


def test_asking_back_never_answers_a_question(monkeypatch):
    # "What did I ask you to remember?" came back ASK: the robot would have answered a question
    # with "what should I remember?".
    monkeypatch.setattr(kc.requests, "post", _Said("ASK"))
    assert kc.wants_kept("What did I ask you to remember?", "http://vlm") == (None, "")
    assert kc.wants_kept("Acordate de que", "http://vlm") == ("ask", "")


def test_the_seen_fact_is_read_off_the_frame_in_the_language_asked(monkeypatch):
    said = _Said("La medida es 14.3 V.")
    monkeypatch.setattr(kc.requests, "post", said)
    assert kc.seen_fact("Acordate de la medida", "la medida", "AAAA", "http://vlm") == "La medida es 14.3 V."
    content = said.sent[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
    assert "Spanish" in content[1]["text"] and "la medida" in content[1]["text"]
    kc.seen_fact("Remember the reading", "the reading", "AAAA", "http://vlm")
    assert "English" in said.sent[1]["messages"][0]["content"][1]["text"]


def test_the_route_reads_the_frame_only_for_seen(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    import service
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(kc, "wants_kept", lambda text, url: ("seen", "la medida"))
    read = []
    monkeypatch.setattr(kc, "seen_fact", lambda said, what, image, url: read.append(image) or "La medida es 3 bar.")
    with TestClient(service.app) as c:
        r = c.post("/keep", json={"text": "acordate de la medida", "image": "AAAA"}).json()
        assert r == {"decided": True, "keep": "seen", "note": "La medida es 3 bar."}
        # No frame: what to remember stays as said, the gateway keeps it as a note.
        assert c.post("/keep", json={"text": "acordate de la medida"}).json()["note"] == "la medida"
        monkeypatch.setattr(kc, "wants_kept", lambda text, url: ("picture", ""))
        c.post("/keep", json={"text": "sacá una foto", "image": "AAAA"})
    assert read == ["AAAA"]
