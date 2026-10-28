"""/classes and /detect take only the YOLO weights /options lists.

ultralytics downloads a URL and unpickles any .pt it is given, so a free-form model name was
code execution on this machine. The model loader is replaced by a fake: if a refused name ever
reached it, the test sees it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402 - after the sys.path lines, like the other tests
from fastapi.testclient import TestClient  # noqa: E402

import service  # noqa: E402
import yolo_common  # noqa: E402

REFUSED = [
    "https://evil.example/x.pt",
    "gs://bucket/x.pt",
    "ul://someone/x",
    "/tmp/x.pt",
    "../../x.pt",
    "subdir/yolo11n.pt",
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    loaded = []
    monkeypatch.setattr(yolo_common, "class_names", lambda m: loaded.append(m) or ["person"])
    monkeypatch.setattr(yolo_common, "detect", lambda m, img, **kw: loaded.append(m) or {"objects": [], "n": 0})
    with TestClient(service.app) as c:
        c.loaded = loaded
        yield c


@pytest.mark.parametrize("name", REFUSED)
def test_classes_refuses_names_outside_the_list(client, name):
    r = client.get("/classes", params={"model": name})
    assert r.status_code == 400
    assert client.loaded == []


@pytest.mark.parametrize("name", REFUSED)
def test_detect_refuses_names_outside_the_list(client, name):
    r = client.post("/detect", params={"model": name}, content=b"not read")
    assert r.status_code == 400
    assert client.loaded == []


def test_listed_and_default_models_pass(client):
    assert client.get("/classes", params={"model": "yolo11n.pt"}).json()["classes"] == ["person"]
    assert client.get("/classes").status_code == 200
    assert client.loaded == ["yolo11n.pt", service.CFG.get("yolo_model", "yolov8n.pt")]
