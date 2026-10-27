"""GET /status: whether the VLM server answers and how full the GPU is, for the header.

No server and no GPU: requests.get and subprocess.run are replaced by fakes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import service  # noqa: E402


class _Ok:
    ok = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    with TestClient(service.app) as c:
        yield c


def test_status_reports_vlm_and_the_whole_cards_memory(client, monkeypatch):
    """Catches: the GPU figure parsed wrong (name with a comma, units) or the VLM probe going to
    the wrong place."""
    seen = []
    monkeypatch.setattr(requests, "get", lambda url, timeout: seen.append(url) or _Ok())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="7509, 8151, NVIDIA GeForce RTX 5060, Laptop\n"))
    body = client.get("/status").json()
    assert body["vlm"]["online"] is True
    assert seen and seen[0].endswith(("/health", "/api/tags"))
    assert body["gpu"] == {"used_mb": 7509, "total_mb": 8151, "name": "NVIDIA GeForce RTX 5060, Laptop"}


def test_status_survives_no_vlm_and_no_gpu(client, monkeypatch):
    """Catches: a down VLM server or a machine without nvidia-smi turning /status into a 500
    (the gateway would then call the whole AI engine down)."""
    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(subprocess, "run", no_smi)
    body = client.get("/status").json()
    assert body["vlm"]["online"] is False
    assert body["gpu"] is None
