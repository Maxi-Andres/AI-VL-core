#!/usr/bin/env python3
"""
tts_benchmark.py — Which voice? Seconds per sentence, VRAM, and how well it is understood.

Every voice says the same sentences (the kind of thing the assistant answers: a scene, a
movement confirmation, a gauge reading). For each voice it reports:
  - seconds per sentence, warm: what the browser waits before it can play a sentence,
    since the app speaks one whole sentence at a time;
  - intelligibility: Whisper (small, on the GPU when it can) transcribes the sample back
    and the word error rate against the text is reported. A clear voice lands at ~6 %:
    the floor, because Whisper writes "320" where the sentence says "trescientos veinte".
    Measured 2026-10-16: the app's own voice, Piper es_AR-daniela-high, came out at 34 %
    ("tesonas", "detenda", once "420") while the other eleven stayed at 6-9 %.
Quality itself is for the ear: the samples are written to results/tts_samples/.

Engines: Piper and Kokoro through the app's tts_common (always), plus two optional ones
that are not dependencies of the app and are tested only when installed:
  - Kyutai Pocket TTS   (pip install pocket-tts)   Spanish, CPU or GPU
  - Qwen3-TTS           (pip install qwen-tts)     GPU; 0.6B needs ~2.3 GB

Usage:
    python3 src/tts_benchmark.py                          # Piper + Kokoro voices installed
    python3 src/tts_benchmark.py --voices es_MX-claude-high ef_dora
    python3 src/tts_benchmark.py --pocket cuda --qwen 0.6B
"""
import argparse
import io
import json
import os
import statistics
import time
import wave
from datetime import datetime

import asr_common
import tts_common
from asr_benchmark import edit_distance, words
from vlm_common import results_path

SENTENCES = [
    "Veo una oficina con dos personas trabajando frente a sus computadoras.",
    "Listo, camino un metro para adelante. Avisame si querés que me detenga.",
    "El manómetro marca trescientos veinte kilos por centímetro cuadrado.",
]


def _wav(pcm_int16, sample_rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_int16)
    return buf.getvalue()


def _float_to_wav(samples, sample_rate):
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1, 1) * 32767)
    return _wav(pcm.astype(np.int16).tobytes(), sample_rate)


# --------------------------------------------------------------------------- #
# Engines: each returns {name: say(text) -> wav bytes}
# --------------------------------------------------------------------------- #
def app_voices(names):
    return {v: (lambda text, v=v: tts_common.synthesize(text, v)["wav"]) for v in names}


def pocket_voices(device):
    import torch
    from pocket_tts import TTSModel
    from pocket_tts.default_parameters import get_default_voice_for_language
    m = TTSModel.load_model(language="spanish")
    m.to(device)
    state = m.get_state_for_audio_prompt(get_default_voice_for_language("spanish"))

    def say(text):
        chunks = [c.detach().float().cpu() for c in
                  m.generate_audio_stream(model_state=state, text_to_generate=text)]
        return _float_to_wav(torch.cat([c.reshape(-1) for c in chunks]).numpy(), m.sample_rate)
    return {f"pocket-spanish-{device}": say}


def qwen_voices(size, speakers):
    import torch
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice",
                                      device_map="cuda:0", dtype=torch.bfloat16)

    def make(spk):
        def say(text):
            wavs, sr = m.generate_custom_voice(text=text, speaker=spk, language="Spanish")
            return _float_to_wav(wavs[0], sr)
        return say
    return {f"qwen3tts-{size}-{s}": make(s) for s in speakers}


# --------------------------------------------------------------------------- #
def measure(name, say, out_dir, whisper):
    say("Hola.")                                        # warm-up, not counted
    times, audio_s, err, n, heard = [], 0.0, 0, 0, []
    for i, text in enumerate(SENTENCES):
        t0 = time.perf_counter()
        wav = say(text)
        times.append(time.perf_counter() - t0)
        with wave.open(io.BytesIO(wav)) as w:
            audio_s += w.getnframes() / w.getframerate()
        with open(os.path.join(out_dir, f"{name}_{i}.wav"), "wb") as f:
            f.write(wav)
        segments, _ = whisper.transcribe(io.BytesIO(wav), language="es")
        hyp = "".join(s.text for s in segments).strip()
        err += edit_distance(words(text, False), words(hyp, False))
        n += len(words(text, False))
        heard.append(hyp)
    return {"voice": name, "per_sentence_s": round(statistics.mean(times), 3),
            "real_time_factor": round(sum(times) / audio_s, 3),
            "roundtrip_wer": round(100 * err / n, 1), "heard": heard}


def main():
    ap = argparse.ArgumentParser(description="TTS speed and intelligibility, voice by voice.")
    ap.add_argument("--voices", nargs="*", default=None,
                    help="Piper/Kokoro voices (default: every installed one; none: skip them).")
    ap.add_argument("--pocket", choices=["cpu", "cuda"], help="Also test Pocket TTS there.")
    ap.add_argument("--qwen", help="Also test Qwen3-TTS of this size (0.6B or 1.7B).")
    ap.add_argument("--qwen-speakers", nargs="+", default=["aiden", "dylan"])
    args = ap.parse_args()

    out_dir = results_path("tts_samples")
    os.makedirs(out_dir, exist_ok=True)
    whisper = asr_common.load_model("small", "cuda", "int8_float16")   # CPU fallback inside

    engines = app_voices(tts_common.list_voices() if args.voices is None else args.voices)
    for flag, loader in ((args.pocket, lambda: pocket_voices(args.pocket)),
                         (args.qwen, lambda: qwen_voices(args.qwen, args.qwen_speakers))):
        if flag:
            try:
                engines.update(loader())
            except ImportError as e:
                print(f"[skip] {e} (optional engine, not installed)")

    rows = []
    for name, say in engines.items():
        row = measure(name, say, out_dir, whisper)
        rows.append(row)
        print(f"{name:28s} {row['per_sentence_s']:6.3f} s/sentence  "
              f"understood: {row['roundtrip_wer']:5.1f}% WER  | {row['heard'][2][:60]}",
              flush=True)
    out = results_path(f"tts_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sentences": SENTENCES, "rows": rows}, f, ensure_ascii=False, indent=1)
        f.write("\n")  # pre-commit's end-of-file-fixer
    print(f"\n[OK] Saved to {out}; samples in {out_dir}")


if __name__ == "__main__":
    main()
