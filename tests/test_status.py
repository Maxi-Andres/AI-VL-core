"""GET /status: whether the VLM server answers and which graphics cards there are, for the header.

No server and no GPU: requests.get and subprocess.run are replaced by fakes, and the kernel's
card list (/sys/class/drm) by a tree in tmp_path.
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


def _card(root: Path, n: int, vendor: str, slot: str, vram=None):
    """A /sys/class/drm/cardN whose device is a PCI slot, like the kernel's."""
    dev = root / "devices" / slot
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    if vram:
        (dev / "mem_info_vram_total").write_text(str(vram[1] * 2**20))
        (dev / "mem_info_vram_used").write_text(str(vram[0] * 2**20))
    (root / f"card{n}").mkdir()
    (root / f"card{n}" / "device").symlink_to(dev)
    (root / f"card{n}-DP-1").mkdir()         # an output, not a card: must be skipped


def _smi(stdout=None, lspci=None):
    """subprocess.run stand-in: nvidia-smi answers `stdout` (or is missing), lspci `lspci`."""
    def run(cmd, **_):
        if cmd[0] == "nvidia-smi" and stdout is not None:
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout)
        if cmd[0] == "lspci" and lspci and cmd[-1] in lspci:
            return subprocess.CompletedProcess(cmd, 0, stdout=lspci[cmd[-1]])
        raise FileNotFoundError(cmd[0])
    return run


def test_status_reports_vlm_and_every_nvidia_card(client, monkeypatch, tmp_path):
    """Catches: the figures parsed wrong (a name with a comma), a second card dropped, or the
    processor's own graphics listed next to the NVIDIA card (its small memory reads as full)."""
    seen = []
    monkeypatch.setattr(requests, "get", lambda url, timeout: seen.append(url) or _Ok())
    monkeypatch.setattr(subprocess, "run", _smi(
        "7509, 8151, NVIDIA GeForce RTX 5060, Laptop\n1200, 49140, NVIDIA L40S\n"))
    _card(tmp_path, 1, "0x1002", "0000:05:00.0", vram=(1990, 2048))
    monkeypatch.setattr(service, "DRM_ROOT", str(tmp_path))
    body = client.get("/status").json()
    assert body["vlm"]["online"] is True
    assert seen and seen[0].endswith(("/health", "/api/tags"))
    assert body["gpus"] == [
        {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 5060, Laptop", "used_mb": 7509, "total_mb": 8151},
        {"vendor": "nvidia", "name": "NVIDIA L40S", "used_mb": 1200, "total_mb": 49140},
    ]


def test_amd_reports_memory_from_sysfs_and_intel_is_named_without_it(client, monkeypatch, tmp_path):
    """Catches: a machine without NVIDIA showing no card at all, AMD's byte counts not turned
    into MB, or Intel (whose drivers give no figure) made up as zero."""
    monkeypatch.setattr(requests, "get", lambda url, timeout: _Ok())
    monkeypatch.setattr(subprocess, "run", _smi(lspci={
        "0000:03:00.0": '03:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" '
                        '"Navi 31 [Radeon RX 7900 XTX]" -rc8 "Sapphire" "Device 471e"\n',
        "0000:00:02.0": '00:02.0 "VGA compatible controller" "Intel Corporation" '
                        '"Raptor Lake-S GT1 [UHD Graphics 770]" -r04 "Dell" "Device 0c1a"\n'}))
    _card(tmp_path, 0, "0x1002", "0000:03:00.0", vram=(3072, 24560))
    _card(tmp_path, 1, "0x8086", "0000:00:02.0")
    monkeypatch.setattr(service, "DRM_ROOT", str(tmp_path))
    assert client.get("/status").json()["gpus"] == [
        {"vendor": "amd", "name": "AMD Radeon RX 7900 XTX", "used_mb": 3072, "total_mb": 24560},
        {"vendor": "intel", "name": "Intel UHD Graphics 770", "used_mb": None, "total_mb": None},
    ]


def test_status_survives_no_vlm_and_no_gpu(client, monkeypatch, tmp_path):
    """Catches: a down VLM server or a machine without any card turning /status into a 500 (the
    gateway would then call the whole AI engine down). No card is an empty list, not None."""
    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(subprocess, "run", _smi())
    monkeypatch.setattr(service, "DRM_ROOT", str(tmp_path))
    body = client.get("/status").json()
    assert body["vlm"]["online"] is False
    assert body["gpus"] == []
