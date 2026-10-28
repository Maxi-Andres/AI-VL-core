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


@pytest.fixture(autouse=True)
def _no_live_classifier(request, monkeypatch):
    if not request.node.get_closest_marker("real_looks"):
        monkeypatch.setattr(vlm_common, "talks_about_the_frame", lambda *a, **k: True)
