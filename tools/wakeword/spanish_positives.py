"""Synthetic "Bender"s said the Spanish way, from the app's own Spanish voices (iacore /speak).

openWakeWord's generator speaks English only (Piper's LibriTTS voices). The first model learned that
"Bender" and missed every one said to the G1 (2026-11-17: all 14 real ones scored ~0.001). These
add the Argentine, Spanish and Mexican voices the app has, each name said several ways and at
several speeds, as more positives for training (train.sh merges them before `augment`).

    .venv/bin/python tools/wakeword/spanish_positives.py      # with the app (iacore) running
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import wave
from pathlib import Path

import numpy as np

SAYINGS = ["Bender", "Bender.", "¡Bender!", "¿Bender?", "Bénder", "Bender...", "¡Bénder!",
           "Bender,"]
SPEEDS = [0.85, 0.92, 1.0, 1.08, 1.15]
RATE = 16000


def speak(url: str, text: str, voice: str) -> tuple[np.ndarray, int]:
    req = urllib.request.Request(f"{url}/speak", data=json.dumps({"text": text, "voice": voice}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r, wave.open(io.BytesIO(r.read())) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768, w.getframerate()


def resample(x: np.ndarray, src: float, dst: int) -> np.ndarray:
    """Linear resampling: from `src` to `dst` Hz. A speed change is a resample read at 16 kHz."""
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim(x: np.ndarray, floor: float = 0.01, keep_s: float = 0.1) -> np.ndarray:
    """Cut the silence around the word, keeping `keep_s` of it each side."""
    loud = np.flatnonzero(np.abs(x) > floor)
    if not len(loud):
        return x
    k = int(keep_s * RATE)
    return x[max(0, loud[0] - k): loud[-1] + k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iacore", default="http://127.0.0.1:8001")
    p.add_argument("--out", type=Path, default=Path.home() / "wakeword" / "data" / "spanish_positive")
    p.add_argument("--test-voice", default="es_MX-claude-high",
                   help="this voice's clips go to test/, never to train/")
    a = p.parse_args()
    with urllib.request.urlopen(f"{a.iacore}/tts/voices", timeout=10) as r:
        voices = json.load(r)["voices"]
    random.seed(17)
    n = 0
    for voice in voices:
        split = "test" if voice == a.test_voice else "train"
        (a.out / split).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(SAYINGS):
            try:
                x, rate = speak(a.iacore, text, voice)
            except OSError as e:
                print(f"{voice} {text!r}: {e}")
                continue
            for speed in SPEEDS:
                y = trim(resample(x, rate * speed, RATE))
                pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
                path = a.out / split / f"{voice}-{i}-{speed}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(pcm.tobytes())
                n += 1
        print(f"{voice}: done ({split})", flush=True)
    print(f"{n} clips in {a.out}")


if __name__ == "__main__":
    main()
