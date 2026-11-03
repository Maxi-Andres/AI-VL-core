"""/actions/*: the pick as an HTTP action — started, followed, cancelled, one at a time.

No robot: manipulation.start_pick is replaced by a fake action with the same snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
from fastapi.testclient import TestClient  # noqa: E402

import manipulation  # noqa: E402
import service  # noqa: E402


class FakeAction:
    def __init__(self, phrase):
        self.id, self.phrase, self.state, self.cancelled = "a1", phrase, "running", False
        self.events = [{"t": 1.0, "step": "looking", "detail": f"finding '{phrase}'"},
                       {"t": 2.0, "step": "found", "detail": "yolo bottle #1"}]

    def snapshot(self, since=0):
        return {"id": self.id, "verb": "pick", "object": self.phrase, "state": self.state,
                "step": self.events[-1]["step"], "result": None, "events": self.events[since:]}

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    started = {}

    def start(phrase, **kw):
        if started:
            raise manipulation.ActionError("another action is running: pick 'the bottle'")
        started["a"] = FakeAction(phrase)
        return started["a"]
    monkeypatch.setattr(manipulation, "start_pick", start)
    monkeypatch.setattr(manipulation, "get", lambda i: started.get("a") if i == "a1" else None)
    with TestClient(service.app) as c:
        yield c, started


def test_a_pick_starts_and_is_followed_by_its_events(client):
    c, _ = client
    r = c.post("/actions/pick", json={"object": "the bottle"})
    assert r.status_code == 200 and r.json()["id"] == "a1" and r.json()["state"] == "running"
    later = c.get("/actions/a1", params={"since": 1}).json()
    assert [e["step"] for e in later["events"]] == ["found"], "only what was not seen yet"


def test_a_second_pick_while_one_runs_is_refused(client):
    """The defect: two actions steering one arm."""
    c, _ = client
    c.post("/actions/pick", json={"object": "the bottle"})
    r = c.post("/actions/pick", json={"object": "the cup"})
    assert r.status_code == 409 and "another action" in r.json()["error"]


def test_cancel_reaches_the_action(client):
    c, started = client
    c.post("/actions/pick", json={"object": "the bottle"})
    assert c.post("/actions/a1/cancel").status_code == 200 and started["a"].cancelled
    assert c.get("/actions/nope").status_code == 404


def test_an_empty_object_is_refused(client):
    c, _ = client
    assert c.post("/actions/pick", json={"object": ""}).status_code == 422


def test_the_agent_streams_its_trace_as_server_sent_events(client, monkeypatch):
    """The defect: the trace arriving all at once at the end, or not as one event per line."""
    import json

    import agent_common
    c, _ = client

    def fake_run(text, emit, stop=None):
        emit({"type": "heard", "text": text})
        emit({"type": "reply", "text": "Listo."})
        return "Listo."
    monkeypatch.setattr(agent_common, "run", fake_run)
    with c.stream("POST", "/agent", json={"text": "agarrá la botella"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    data = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: {\"")]
    assert [d["type"] for d in data] == ["heard", "reply"]
    assert "event: end" in body


def test_stop_cancels_every_action(client, monkeypatch):
    c, _ = client
    called = []
    monkeypatch.setattr(manipulation, "cancel_all", lambda: called.append(True))
    assert c.post("/agent/stop").json() == {"ok": True} and called
