"""tools/wakeword/review.py: labels are saved where bench.py reads them, and nothing else is."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "wakeword"))

import review


def session(tmp_path):
    s = tmp_path / "g1-2026-11-17"
    s.mkdir()
    rows = [{"wav": "a.wav", "label": "unlabeled", "whisper_hotword": "Bender"},
            {"wav": "b.wav", "label": "not", "whisper_hotword": "hola"}]
    (s / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return s


def test_a_label_set_by_listening_is_saved_and_shown(tmp_path):
    s = session(tmp_path)
    assert review.set_label(tmp_path, s.name, "a.wav", "bender")
    saved = [json.loads(x) for x in (s / "labels.jsonl").read_text().splitlines()]
    assert saved[0]["label"] == "bender" and saved[1]["label"] == "not"
    clips = review.load(tmp_path)["clips"]
    assert [c["label"] for c in clips] == ["bender", "not"]
    assert clips[0]["heard"]["hotword"] == "Bender"        # before bench.py has run


def test_nothing_outside_the_recordings_is_written(tmp_path):
    s = session(tmp_path)
    assert not review.set_label(tmp_path, "../etc", "a.wav", "bender")
    assert not review.set_label(tmp_path, s.name, "../labels.jsonl", "bender")
    assert not review.set_label(tmp_path, s.name, "a.wav", "maybe")
    assert not review.set_label(tmp_path, s.name, "missing.wav", "bender")
