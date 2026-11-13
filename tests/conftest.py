"""Shared test setup: no test reaches the live VLM server by accident.

query_vlm_stream asks llama-server whether a message is talk or about the frame
(vlm_common.talks_about_the_frame). A test that fakes the llamacpp backend would otherwise send
that question to whatever answers on :8081 — on the dev machine, the real server. Here it is
always "about the frame" (the reading prompt); the tests of the classifier itself are marked
`real_looks` and patch the network themselves.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vlm_common


def pytest_configure(config):
    config.addinivalue_line("markers", "real_looks: run vlm_common.talks_about_the_frame itself")
    config.addinivalue_line("markers", "real_interpreter: run agent_common._interpret itself "
                                       "(the test fakes the network under it)")


@pytest.fixture(autouse=True)
def _no_live_classifier(request, monkeypatch):
    if not request.node.get_closest_marker("real_looks"):
        monkeypatch.setattr(vlm_common, "talks_about_the_frame", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _no_live_transport(monkeypatch):
    """The agent's moves are narrowed to what the executor says the G1's link delivers
    (agent_common.sendable). No test asks the executor: everything gets through (as on DDS)
    unless the test says otherwise."""
    import agent_common
    monkeypatch.setitem(agent_common._SENDABLE, "at", float("inf"))
    monkeypatch.setitem(agent_common._SENDABLE, "value", None)


@pytest.fixture(autouse=True)
def _no_live_interpreter(request, monkeypatch):
    """The agent's command tool asks the command interpreter (the live model) what a move in
    words is. A test that needs it passes its own `interpret`."""
    if request.node.get_closest_marker("real_interpreter"):
        return
    import agent_common

    def no_model(words, moves):
        raise AssertionError(f"a test asked the live interpreter about {words!r}")
    monkeypatch.setattr(agent_common, "_interpret", no_model)
