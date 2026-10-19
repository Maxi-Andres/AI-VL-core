# Model benchmarks — VLM, speech-to-text, text-to-speech

Every model choice iacore makes for the app was measured on this machine (GeForce RTX 5060
8 GB, driver 595.91.07 / CUDA 13.2; AMD Ryzen 9 8945HX, 16 cores), not taken from a
leaderboard. This file keeps the numbers, how
they were taken, and how to take them again. The raw results of the 2026-10-16 runs are in
[`benchmarks/2026-10-16/`](benchmarks/2026-10-16/).

The budget every choice has to fit: the desktop holds ~1.2-1.4 GB of the 8 GB, the VLM
~4.5 GB, Whisper ~0.5 GB, YOLO a few hundred MB. What is left is the margin.

| Job | Chosen | Script |
|---|---|---|
| VLM (look at a frame, answer, detect, interpret commands) | Qwen3-VL 4B Instruct Q4_K_M under llama-server, `--image-min-tokens 1024` | `src/vlm_server_benchmark.py` |
| Speech-to-text | faster-whisper `small`, GPU, `int8_float16` | `src/asr_benchmark.py` |
| Text-to-speech | Piper `es_AR-daniela-high` for now; **under review** (see TTS below) | `src/tts_benchmark.py` |

---

## 1. VLM — 2026-10-16

### Method (`src/vlm_server_benchmark.py`)

Each candidate is served the way the app serves it: `../linux/vlm-server.sh` starts a
llama-server with that GGUF and its vision projector on its own port (8082, so a running
app is not touched), and the test goes through the same functions the service calls.

- **Ask**: the voice path (`query_vlm_stream`), three Spanish questions on photos scaled
  to 768 px, JPEG 85, the way the browser sends them; two rounds. Time to the first piece
  of text and to the full answer.
- **Facts**: eight questions with one right answer (a count, a brand, a display reading,
  a colour), checked with regexes.
- **Commands**: eight spoken G1 commands through `command_common.interpret`, checked
  against the expected skill.
- **Detect**: `query_vlm` scope `all` on the office photo: valid JSON, objects, time.
- **VRAM**: nvidia-smi peak during the run minus what was used before the server started.

**Every request is a fresh frame.** llama-server and Ollama both cache image embeddings:
the first version of the script reused the bytes and measured the second round of each
photo at 0.3 s with no vision work at all. The script now changes one pixel per request,
as the app never sends the same frame twice. `vlm_server_benchmark_20261016_130056.json`
(the cached run) is deliberately **not** kept.

### Why not Ollama

Ollama 0.30.6 starts its own llama-server with `--no-mmproj-offload` (its log says "CLIP
using CPU backend"): the vision encoder runs on the CPU whatever the model or the free
VRAM. No Ollama option changes it; `qwen3-vl:2b` takes the same 6.4 s, and so does the
official GGUF imported with `ollama create`.

| served by | model | first word | answer | facts | commands | VRAM |
|---|---|---:|---:|---:|---:|---:|
| Ollama | `qwen3-vl:4b-instruct` | 6.81 s | 7.34 s | 8/8 | 8/8 | 3.2 GB |
| llama-server | Qwen3-VL 4B Q4_K_M, model's own token floor | 0.18 s | 0.79 s | 7/8 | 8/8 | 4.4 GB |
| llama-server | same, `--image-min-tokens 1024` | **0.60 s** | **1.22 s** | **8/8** | **8/8** | 4.5 GB |

The token floor: Ollama forces 1024 tokens per image; the model's own floor turns a 768 px
frame into ~430 tokens, and the brand on an instrument's nameplate (Chauvin Arnoux) is then
misread. Keeping 1024 costs 0.4 s and buys that back.

Detection (24 objects on the office photo) takes 9.5-12.6 s on llama-server against 18.6-20
s on Ollama: it is ~1300 tokens of JSON, so it is bound by generation speed, not vision.

### Every candidate that could run on 8 GB

| model | first word | answer | gen tok/s | facts | commands | VRAM | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **Qwen3-VL 4B Q4, 1024 tokens** | 0.60 s | 1.22 s | 112 | 8 | 8 | 4.5 GB | **chosen** |
| Qwen3-VL 2B Q8 | 0.13 s | 0.49 s | 166 | 6 | 7 | 3.6 GB | "levantate" → `lie_up`; detection JSON loops to the token limit |
| Qwen3-VL 2B Q8, 1024 tokens | 0.44 s | 0.83 s | 161 | 7 | 7 | 3.6 GB | same command error |
| Qwen3-VL 4B Q8, 1024 tokens | 0.57 s | 1.61 s | 80 | 8 | 8 | 6.2 GB | as accurate, no room left for YOLO |
| Qwen3-VL 8B Q4 | — | — | — | — | — | OOM | out of memory on the first image |
| Qwen3-VL 8B Q4, 26/36 layers on GPU | 1.69 s | 6.81 s | 16 | 8 | 8 | 5.1 GB | accurate but 5-11 s per answer, 90 s per detection |
| Gemma 3 4B Q4 | 0.69 s | 1.06 s | 109 | 7 | 7 | 3.9 GB | **"pará" → `walk`**; answers Spanish questions in English; empty detection |
| InternVL3 8B Q4 | 0.52 s | 1.24 s | 79 | 6 | 8 | 5.1 GB | fits, misreads, detects 10 of ~24 objects |
| MiniCPM-V 4.5 Q4 | — | — | — | 5 | 0 | 5.1 GB | OOM on the GPU; with 24 layers there, every answer comes back empty (it spends it reasoning) |
| SmolVLM2 2.2B Q8 | 0.56 s | 1.12 s | 189 | 3 | 2 | 3.4 GB | too weak |

Raw files: `vlm_*.json`. GGUFs came from the publishers' own Hugging Face repos
(`Qwen/Qwen3-VL-*-Instruct-GGUF`, `ggml-org/*-GGUF`, `openbmb/MiniCPM-V-4_5-gguf`).

---

## 2. Speech-to-text — 2026-10-16

### Method (`src/asr_benchmark.py`)

30 sentences, one from each of 30 different Argentine speakers (15 women, 15 men), from
OpenSLR 61, "Crowdsourced high-quality Argentinian Spanish speech data set" (Google,
CC BY-SA 4.0), through its Hugging Face mirror `ylacombe/google-argentinian-spanish`.
`--fetch` downloads them into `asr_data/` (git-ignored). Each configuration runs in its own
process; WER is word edit distance over the reference word count after lower-casing and
dropping punctuation, and "no accents" also drops the tildes. Beam size 5, language forced
to Spanish.

| engine | device | WER | no accents | per clip | VRAM |
|---|---|---:|---:|---:|---:|
| Whisper base int8 (old default) | CPU | 8.0 % | 6.6 % | 0.47 s | 0 |
| Whisper small int8 | CPU | 2.9 % | 1.5 % | 1.14 s | 0 |
| Parakeet TDT 0.6B v3 int8 (onnx-asr) | CPU | 4.4 % | 2.6 % | 0.21 s | 0 |
| Whisper base fp16 | GPU | 6.9 % | 5.8 % | 0.13 s | 386 MB |
| Whisper small fp16 | GPU | 2.6 % | 1.1 % | 0.16 s | 834 MB |
| **Whisper small int8_float16** | GPU | **2.6 %** | **1.1 %** | **0.16 s** | **482 MB** |
| Whisper medium fp16 | GPU | 3.3 % | 1.8 % | 0.25 s | 2.1 GB |
| Whisper large-v3-turbo int8_float16 | GPU | 2.6 % | 1.1 % | 0.25 s | 1.3 GB |
| Whisper large-v3-turbo fp16 | GPU | 2.2 % | 1.1 % | 0.27 s | 2.2 GB |

`small` on the GPU matches turbo, in two thirds of the time and a third of the VRAM. What
it still gets wrong is almost all spelling: "champú" → "shampoo", "Helena" → "Elena",
"recomendas" without its tilde; the one real miss was "ví" → "vio". Parakeet is the plan B
if VRAM runs out: no GPU, 0.21 s, but twice the errors.

On the GPU, CTranslate2 4.8 needs CUDA 12's cuBLAS while torch brings CUDA 13:
`asr_common` loads Ollama's `cuda_v12/libcublas.so.12` first, and falls back to the CPU
when the GPU fails (it checks on half a second of silence, because a missing cuBLAS only
shows at the first transcription).

Raw file: `asr_all_configs.jsonl`, one line per configuration with every clip's "said"
and "heard" (written by the prototype of `asr_benchmark.py`, so its keys are `outs` and
`wer_noaccents`).

---

## 3. Text-to-speech — 2026-10-16

### Method (`src/tts_benchmark.py`)

Three sentences of the kind the assistant says (a scene, a movement confirmation, a
gauge reading). **Seconds per sentence** is what the browser waits before it can play a
sentence, since the app speaks one whole sentence at a time. **Intelligibility**: Whisper
`small` transcribes each sample back; a clear voice lands at ~6 %, the floor, because
Whisper writes "320" where the sentence says "trescientos veinte". Three sentences are 32
words, so one word is 3 %: treat a difference under ~6 points as noise. Quality is for the
ear: the samples go to `results/tts_samples/`.

| voice | engine | accent | where | per sentence | understood (WER) |
|---|---|---|---|---:|---:|
| **es_AR-daniela-high (current)** | Piper | Argentina | CPU | 0.40 s | **34.4 %** |
| es_MX-claude-high | Piper | Mexico | CPU | 0.15 s | 9.4 % |
| es_MX-ald-medium | Piper | Mexico | CPU | 0.12 s | 9.4 % |
| es_ES-davefx-medium | Piper | Spain | CPU | 0.08 s | 9.4 % |
| es_ES-sharvard-medium | Piper | Spain | CPU | 0.11 s | 6.2 % |
| ef_dora | Kokoro | Spain | CPU | 0.65 s | 6.2 % |
| em_alex | Kokoro | Spain | CPU | 0.65 s | 6.2 % |
| em_santa | Kokoro | Spain | CPU | 0.55 s | 6.2 % |
| Pocket TTS `spanish` (lola) | Kyutai | Spain | GPU 573 MB | 0.32 s | 9.4 % |
| Pocket TTS `spanish` (lola) | Kyutai | Spain | CPU | 1.27 s (first audio 0.10 s streamed) | 9.4 % |
| Pocket TTS `spanish_24l` | Kyutai | Spain | GPU 1.5 GB | 0.72 s | 9.4 % |
| Qwen3-TTS 0.6B CustomVoice (aiden, dylan) | Alibaba | neutral | GPU 2.3 GB | 4.06-4.13 s | 9.4 % |

The app's own voice is the least intelligible: "dos tesonas" for "dos personas",
"detenda" for "detenga", and in one run **"420" for "trescientos veinte"**, which matters
for a robot that reads gauges. Slowing it down (Piper `length_scale` 1.15 and 1.3) does not
fix it; it is the voice model. It is also the only Argentine voice. Choosing a replacement
is open: listen to the samples first.

**Re-checked 2026-10-19** (same sentence, Whisper back): pocket-tts 3.3 streams on 4 CPU
threads with the first audio at **0.08 s** and 3.4× real time, and its catalog voices speak
the `spanish` model too (charles, marius, javert, giovanni understood word for word; rafael
stopped after the first sentence). Cloning a voice from a recording needs the gated weights
(accept the terms on Hugging Face). **Supertonic 3** (`supertonic` 1.3.1, 66M, ONNX, CPU,
OpenRAIL): 10 voices (M1-M5, F1-F5), all ten understood word for word, ~1.3 s for 8 s of audio
(no streaming: one sentence at a time, like Kokoro). Neither is wired into `tts_common` yet;
samples in `results/tts_samples/new_2026-10-19/`.

Not viable on this card next to the VLM: Qwen3-TTS (4 s per sentence; the `qwen-tts` 0.1.1
package does not stream) and Chatterbox Multilingual (3.5-4 GB per NVIDIA's guide; not
measured). Both are candidates for a bigger GPU.

Raw files: `tts_speed.jsonl`, `tts_intelligibility.jsonl` (what Whisper heard per sample).

---

## 4. Reading check — 2026-10-19

`src/vlm_reading_check.py`: 57 questions with a known answer over the 33 photos in `fotos/`
(brand, model, display reading, units, counts, what a thing is), asked through
`query_vlm_stream` as the voice assistant asks them — with the robot's identity on — 3 times
each with a fresh frame, at 1152 px. Expected answers read off the photos by eye.

| identity | right | always right | never |
|---|---:|---:|---:|
| Bender (G1) | 159/171 (93 %) | 53/57 | 4 |
| none | same 4 failures, same answers | | |

The identity costs nothing here. The four the model never gets, with or without it: the
gauge pointer of `clean/1` (says ~250 kg/cm²; it rests on 0), **21.65 kVA** (reads 2165 — the
decimal point of that display), the cable glands of `clean/25` (says 16; there are 4) and the
small "Spectra" badge of `clean/31` (reads "EPCIA").

## Re-running

```bash
cd AI-VL-core/src
../.venv/bin/python vlm_server_benchmark.py --list                       # which GGUFs are present
../.venv/bin/python vlm_server_benchmark.py --ollama qwen3-vl:4b-instruct --image-min-tokens 0 1024
VLM_NGL=26 ../.venv/bin/python vlm_server_benchmark.py --models qwen3vl-8b-q4   # a model that does not fit
../.venv/bin/python asr_benchmark.py --fetch && ../.venv/bin/python asr_benchmark.py
../.venv/bin/python vlm_reading_check.py                                  # needs ../linux/run.sh up (:8081)
../.venv/bin/python tts_benchmark.py                                      # + --pocket cuda --qwen 0.6B if installed
```

Close the app first (`run.sh` holds ~5 GB of the GPU): with it running, the same Whisper
configuration measured 0.27 s instead of 0.16 s. The optional engines are not app
dependencies: `pip install "onnx-asr[cpu,hub]"` (Parakeet), `pocket-tts`, `qwen-tts`.
