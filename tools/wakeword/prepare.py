"""Background audio for training "Bender": room echoes, everyday noise, and the G1's own mic.

Runs inside the training container (train.sh), writing 16 kHz mono 16-bit WAVs under /work/data:

  mit_rirs/        MIT's environmental impulse responses (how rooms echo), from Hugging Face
  audioset_16k/    a few AudioSet shards (speech, music, machines, everything), resampled
  g1_background/   the G1's own mic: what it hears with nobody saying the name

The G1 part is the one that matters most: "Bender" has to be found inside the robot's fan noise
and the room's talk, and openWakeWord's stock data has neither. It is taken only from recordings
LABELLED "not" in /recorded/*/labels.jsonl, and only from those `--g1-train-before` a time, so the
clips the benchmark (bench.py) scores the model on are never trained on.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
import torchaudio

RATE = 16000
AUDIOSET = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{}"


def save(path: Path, wave: torch.Tensor, rate: int) -> None:
    """One clip as 16 kHz mono 16-bit PCM."""
    if wave.dim() == 2:
        wave = wave.mean(0)
    if rate != RATE:
        wave = torchaudio.functional.resample(wave, rate, RATE)
    pcm = (wave.clamp(-1, 1).numpy() * 32767).astype(np.int16)
    scipy.io.wavfile.write(path, RATE, pcm)


def mit_rirs(out: Path) -> None:
    import datasets
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        print(f"{out}: already there")
        return
    rows = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses",
                                 split="train", streaming=True)
    n = 0
    for row in rows:
        audio = row["audio"]
        save(out / Path(audio["path"]).name, torch.tensor(audio["array"], dtype=torch.float32),
             audio["sampling_rate"])
        n += 1
    print(f"{out}: {n} impulse responses")


def audioset(out: Path, shards: list[str], tmp: Path) -> None:
    """AudioSet's balanced-train parquet shards (~690 MB, ~600 ten-second clips each)."""
    import io

    import pyarrow.parquet as pq
    import soundfile
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        local = tmp / shard.replace("/", "_")
        if not local.exists():
            print(f"downloading {shard}", flush=True)
            urllib.request.urlretrieve(AUDIOSET.format(shard), local)
        n = 0
        table = pq.ParquetFile(local)
        for batch in table.iter_batches(batch_size=32):
            for i, row in enumerate(batch.to_pylist()):
                audio = row.get("audio") or {}
                data = audio.get("bytes") if isinstance(audio, dict) else None
                if not data:
                    continue
                name = Path(audio.get("path") or f"{local.stem}-{n}-{i}").stem
                dest = out / f"{name}.wav"
                if dest.exists():
                    continue
                try:
                    wave, rate = soundfile.read(io.BytesIO(data), dtype="float32")
                except RuntimeError:
                    continue
                wave = torch.from_numpy(wave)
                save(dest, wave.T if wave.dim() == 2 else wave, rate)
                n += 1
        print(f"{shard}: {n} clips", flush=True)


def g1_background(out: Path, recorded: Path, before: str) -> None:
    """The G1 mic's clips labelled "not", recorded before `before` (YYYYmmdd-HHMMSS)."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for labels in sorted(recorded.glob("*/labels.jsonl")):
        for line in labels.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["label"] != "not" or row["wav"] >= before:
                continue
            src = labels.parent / row["wav"]
            if src.exists():
                wave, rate = torchaudio.load(str(src))
                save(out / row["wav"], wave, rate)
                n += 1
    print(f"{out}: {n} G1 clips (labelled 'not', before {before})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="/work/data", type=Path)
    p.add_argument("--recorded", default="/recorded", type=Path)
    p.add_argument("--audioset-shards", nargs="*",
                   default=["bal_train/00.parquet", "bal_train/01.parquet"])
    p.add_argument("--g1-train-before", default="20261117-115500",
                   help="G1 clips recorded before this go to training; the rest are the test")
    a = p.parse_args()
    mit_rirs(a.data / "mit_rirs")
    audioset(a.data / "audioset_16k", a.audioset_shards, a.data / "tmp")
    g1_background(a.data / "g1_background", a.recorded, a.g1_train_before)


if __name__ == "__main__":
    main()
