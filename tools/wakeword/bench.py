"""Score ways of hearing "Bender" on the G1's own recordings: names caught, names made up.

Each recording in ~/.pilot/wakeword/*/labels.jsonl ("bender", "not", or "unlabeled") is heard by
every method, and the result goes to ~/.pilot/wakeword/bench.json, which review.py shows next to
the audio. The two numbers that decide (2026-11-17, how Alexa-class wake words are judged):

  recall        of the clips where someone said the name, how many woke it (goal: >= 95 %)
  false wakes   of the clips where nobody did, how many woke it — and, over the unbroken
                recordings (stream/*.wav), false wakes PER HOUR (goal: < 0.5)

Methods:
  whisper_hotword   the ear until 2026-11-17: Whisper steered to "Bender" finds the name
  whisper_confirm   ...and the same clip heard WITHOUT the hotword sounds like it (iacore 0c3d441)
  oww:<model>       an openWakeWord model (train.sh), its best score in the clip over a threshold

Runs in iacore's venv: `.venv/bin/python tools/wakeword/bench.py [--oww ~/wakeword/models/bender.onnx]`.
Whisper runs on the CPU here (the app holds the GPU), and what it heard is cached per clip.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import ear_common  # noqa: E402

RECORDED = Path.home() / ".pilot" / "wakeword"
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def clips(recorded: Path):
    """(session, row) for every labelled recording."""
    for labels in sorted(recorded.glob("*/labels.jsonl")):
        for line in labels.read_text(encoding="utf-8").splitlines():
            yield labels.parent, json.loads(line)


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


class Whisper:
    """Both hearings of a clip, cached: they take seconds each on the CPU."""

    def __init__(self, cache: Path, model: str):
        self.cache_path = cache
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}
        self.model_name = model
        self.model = None

    def heard(self, path: Path) -> dict:
        key = path.name
        if key not in self.cache:
            if self.model is None:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            out = {}
            for name, hot in (("hotword", "Bender"), ("plain", None)):
                segs, _ = self.model.transcribe(str(path), language="es", hotwords=hot,
                                                vad_filter=True)
                out[name] = "".join(s.text for s in segs).strip()
            self.cache[key] = out
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0))
        return self.cache[key]


class OpenWakeWord:
    def __init__(self, model_path: Path):
        sys.path.insert(0, str(Path.home() / "wakeword" / "src" / "openWakeWord"))
        from openwakeword.model import Model
        self.name = f"oww:{model_path.stem}"
        self.model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def scores(self, pcm: np.ndarray, pad_s: float = 0.0) -> list[float]:
        """The model's score for every 80 ms frame of `pcm`. `pad_s` of silence before and after
        a short clip: the model answers only once its ~1 s of context is full, and a 1.3 s clip
        ended before it heard the word in it (every score ~0.001, 2026-11-17)."""
        self.model.reset()
        if pad_s:
            pad = np.zeros(int(pad_s * 16000), dtype=np.int16)
            pcm = np.concatenate([pad, pcm, pad])
        out = []
        for i in range(0, len(pcm) - 1279, 1280):
            out.append(float(next(iter(self.model.predict(pcm[i:i + 1280]).values()))))
        return out


def stream_false_wakes(det: OpenWakeWord, recorded: Path, threshold: float) -> dict:
    """False wakes per hour over the unbroken recordings, where nobody is known to have said the
    name (sessions whose labels have no "bender" in that stretch are taken as all negative)."""
    seconds, wakes = 0.0, 0
    for path in sorted(recorded.glob("*/stream/*.wav")):
        pcm = read_pcm(path)
        seconds += len(pcm) / 16000
        above, last = False, -1e9
        for i, s in enumerate(det.scores(pcm)):
            t = i * 0.08
            if s >= threshold and not above and t - last > 2.0:     # one wake per 2 s
                wakes, last = wakes + 1, t
            above = s >= threshold
    hours = seconds / 3600
    return {"hours": round(hours, 3), "false_wakes": wakes,
            "per_hour": round(wakes / hours, 2) if hours else None}


def summary(rows: list[dict], method: str) -> dict:
    said = [r for r in rows if r["label"] == "bender"]
    not_said = [r for r in rows if r["label"] == "not"]
    caught = sum(r["woke"][method] for r in said)
    false = sum(r["woke"][method] for r in not_said)
    return {"recall": f"{caught}/{len(said)}", "false_wakes": f"{false}/{len(not_said)}"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recorded", type=Path, default=RECORDED)
    p.add_argument("--oww", type=Path, nargs="*", default=[], help="openWakeWord .onnx models")
    p.add_argument("--whisper", default="large-v3-turbo", help="the app's model (linux/run.sh)")
    a = p.parse_args()

    whisper = Whisper(a.recorded / "whisper_cache.json", a.whisper)
    detectors = [OpenWakeWord(m) for m in a.oww]
    rows = []
    for session, row in clips(a.recorded):
        path = session / row["wav"]
        if not path.exists():
            continue
        heard = whisper.heard(path)
        woke_hot, _ = ear_common.wake_command(heard["hotword"])
        woke = {"whisper_hotword": woke_hot,
                "whisper_confirm": woke_hot and ear_common.sounds_like_name(heard["plain"])}
        scores = {}
        if detectors:
            pcm = read_pcm(path)
            for det in detectors:
                best = max(det.scores(pcm, pad_s=1.5) or [0.0])
                scores[det.name] = round(best, 4)
                for th in THRESHOLDS:
                    woke[f"{det.name}@{th}"] = best >= th
        rows.append({"session": session.name, "wav": row["wav"], "label": row["label"],
                     "seconds": row.get("seconds"), "heard": heard, "scores": scores,
                     "woke": woke})
        print(f"{row['wav']} {row['label']:9} {heard['hotword'][:30]!r:34} "
              f"{heard['plain'][:30]!r:34} {scores}", flush=True)

    methods = list(rows[0]["woke"]) if rows else []
    result = {"methods": {m: summary(rows, m) for m in methods}, "clips": rows}
    for det in detectors:
        for th in THRESHOLDS:
            result["methods"][f"{det.name}@{th}"]["stream"] = stream_false_wakes(det, a.recorded, th)
    (a.recorded / "bench.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for m, s in result["methods"].items():
        print(f"{m:28} {s}")


if __name__ == "__main__":
    main()
