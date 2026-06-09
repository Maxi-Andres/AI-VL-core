# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PoC harness for evaluating vision-language models (VLMs) served by **Ollama** for industrial inspection (mining / oil & gas). The VLM is prompted in Spanish (Rioplatense) to detect instruments/objects and return structured JSON with normalized bounding boxes. The JSON contract is meant to feed a downstream VLA stage and is consumed in production by the "Silk AI Proxy Gateway / F1.9".

## Commands

```bash
pip install requests                 # only dependency

# Interactive menu (main entry point) — pick model/image/scope, persists to config.json
python3 menu.py

# Smoke test: one image, prints latency + parsed JSON
python3 03_smoke_test.py fotosClean/1.jpeg
python3 03_smoke_test.py fotosClean/1.jpeg --model qwen3-vl:4b --scope todo

# Benchmark: all images in a folder, N runs per image, across models
python3 04_benchmark.py ./fotosClean --runs 3
python3 04_benchmark.py ./fotosClean --models qwen3-vl:8b qwen3-vl:4b
```

See `README.md` for the full flag reference.

Prerequisite: an Ollama server reachable at `http://localhost:11434` with the target models pulled. Check with `curl http://localhost:11434/api/version`. Override the endpoint on either script with `--url`. Use `ollama ps` to see whether a model loaded fully on GPU or split CPU/GPU.

### Model / hardware constraint (important)

The smoke-test default is `qwen3-vl:4b` (not 8b) because of VRAM. On an 8 GB GPU (e.g. RTX 5060), `qwen3-vl:8b` does **not** fit and Ollama splits it ~53% CPU / 47% GPU, pushing latency to ~85–110 s/image. `qwen3-vl:4b` (~3.3 GB) loads 100% on GPU and runs at ~18–25 s/image. Verify the split with `ollama ps`.

`qwen3-vl` is a **reasoning ("thinking") model** and on Ollama 0.30.6 `"think": false` does **not** actually disable thinking — it only shortens it (verified empirically). So fighting the reasoning is futile; the code embraces it instead. The real failure mode is **token starvation**: if the reasoning consumes the whole output budget, `content` comes back **empty** with `finish_reason: "length"`. The fix is budget, not the flag — give it room with `--max-tokens` (output budget, `num_predict`, default 4096) **and** `--num-ctx` (default 8192).

Two knobs, and they interact:
- **`max_tokens` / `num_predict`** = ceiling on tokens the model **generates** (thinking + answer).
- **`num_ctx`** = the full context window (input + output), and the value `ollama ps` shows under *context*. The image+prompt input alone is ~1000–2600 tokens, so `num_ctx` must exceed input + desired output or the answer gets squeezed out. Bonus: a larger `num_ctx` also lets Ollama feed the image at **higher resolution** (more image tokens → more detail).

Reasoning is **streamed live**: the smoke test (and menu) print the model's thinking dimmed in real time (`verbose=True` in `query_vlm`). `--think`/`--no-think` toggle requesting reasoning; default is on so you can watch it.

## Architecture

All entry points share **`vlm_common.py`** — there is no longer any prompt/client duplication. Change prompts, the JSON schema, or the Ollama client in that one module and every entry point picks it up. Layout:

- `vlm_common.py` — the core. Holds `SCOPES` (the detection-mode prompts), `query_vlm()` (the streaming Ollama client), `extract_json()`/`render_result()`, `image_size()`/`normalize_bboxes()` (bbox post-processing), and the `config.json` load/save (`DEFAULT_CONFIG`, `load_config`, `save_config`).
- `menu.py` — interactive launcher (run with no args / IDE "Play"). Reads/writes `config.json`, lists Ollama models via `/api/tags`, lists images from the folder, and calls `run_smoke` / `run_benchmark` (imported from the numbered scripts via `importlib`, since `03_...` isn't a valid module identifier).
- `03_smoke_test.py` — single-image check. Exposes `run_smoke(...)`; `main()` just parses args (defaults pulled from `config.json`) and calls it.
- `04_benchmark.py` — latency/reliability benchmark. Exposes `run_benchmark(...)`; computes P50/P95/mean latency and JSON-valid rate per model, prints a table, writes `benchmark_resultados.json`.

Both scripts hit Ollama's **native** endpoint (`/api/chat`) with `stream: true`, **not** the OpenAI-compatible `/v1/chat/completions`. This was a deliberate switch: the OpenAI endpoint ignores `think` and mixes reasoning into `content` (so a thinking model can return empty `content`), whereas the native endpoint puts reasoning in a separate `thinking` field and the JSON answer cleanly in `content`. Streaming is what lets us print the reasoning live. Images go in the message's `images: [base64]` array (raw base64, no data-URI prefix). Requests use `format: "json"` and `options: {temperature: 0.1, num_predict: max_tokens, num_ctx}`. `host_of()` normalizes any `url` (including an old `…/v1/chat/completions` from a stale `config.json`) down to the base host before appending `/api/chat`.

**Bounding boxes:** qwen3-vl returns bbox in **absolute pixel** coordinates of the source image, not normalized. `query_vlm` normalizes them to 0–1 via `normalize_bboxes()`, using `image_size()` (a dependency-free JPEG/PNG header reader) — any bbox value > 1 is treated as pixels and divided by width/height. Callers must pass `size=image_size(path)`.

### Detection scopes

Two modes, both keyed in `SCOPES`:
- `industrial` — open detection of **any** industrial instrument. `tipo` is a coarse family (`presion|temperatura|caudal|nivel|electrica|analisis|control|vibracion|valvula|epp|otro`) and `descripcion` is free text for the specifics. The prompt lists per-family examples (manómetro, termopar, caudalímetro Coriolis, sensor radar, etc.) as *reference, not a closed list*, and tells the model not to over-deliberate on the category — this is what stops it from burning reasoning tokens deciding where a flow meter "belongs".
- `todo` — any object, free-form `tipo`. Looser; output varies more between runs.

An empty `{"objetos": []}` means the model genuinely saw nothing matching the scope — not a parsing bug. (Earlier `industrial` empties on thermometer/sensor images were the taxonomy being too narrow, since fixed.)

### config.json

Created on first run from `DEFAULT_CONFIG`. The CLI scripts read it for their defaults; any flag overrides for that run only (does not write back). The menu writes choices back. Keys: `model`, `image`, `folder`, `scope`, `max_tokens`, `num_ctx`, `think`, `url`, `benchmark_models`, `benchmark_runs`.

### Evaluation targets (from internal doc "F1.8")

The benchmark exists to pick the primary VLM for the PoC against three criteria: detection precision, **P95 latency < 1.5 s on-prem**, and a high valid-JSON rate for the VLM→VLA contract. Default model lineup compared: `qwen3-vl:8b`, `qwen3-vl:4b`, `qwen2.5vl:7b`.

## Data

`fotosClean/` holds the lab test images (`*.jpeg`). Supported extensions in the benchmark: jpg, jpeg, png, bmp, webp.
