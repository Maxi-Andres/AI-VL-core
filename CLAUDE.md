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

`qwen3-vl` is a **reasoning ("thinking") model**. Both scripts disable reasoning by sending `"think": false` in the payload — this is the real switch (the older `/no_think` prompt suffix is unreliable and was removed). Pass `--think` to re-enable it. With reasoning on, the model can spend the entire `max_tokens` budget thinking and return an **empty `content`** (`finish_reason: "length"`); that empties-out is the symptom, not malformed JSON. `--max-tokens` (default 4096) is the ceiling. The smoke test prints `finish_reason` and falls back to showing `reasoning_content` so an empty answer is diagnosable.

## Architecture

All entry points share **`vlm_common.py`** — there is no longer any prompt/client duplication. Change prompts, the JSON schema, or the Ollama client in that one module and every entry point picks it up. Layout:

- `vlm_common.py` — the core. Holds `SCOPES` (the detection-mode prompts), `query_vlm()` (the Ollama client), `extract_json()`/`render_result()`, and the `config.json` load/save (`DEFAULT_CONFIG`, `load_config`, `save_config`).
- `menu.py` — interactive launcher (run with no args / IDE "Play"). Reads/writes `config.json`, lists Ollama models via `/api/tags`, lists images from the folder, and calls `run_smoke` / `run_benchmark` (imported from the numbered scripts via `importlib`, since `03_...` isn't a valid module identifier).
- `03_smoke_test.py` — single-image check. Exposes `run_smoke(...)`; `main()` just parses args (defaults pulled from `config.json`) and calls it.
- `04_benchmark.py` — latency/reliability benchmark. Exposes `run_benchmark(...)`; computes P50/P95/mean latency and JSON-valid rate per model, prints a table, writes `benchmark_resultados.json`.

Both scripts hit Ollama's **OpenAI-compatible** endpoint (`/v1/chat/completions`), not the native API. Images are base64-encoded as a `data:image/jpeg;base64,...` `image_url` part. Requests use `temperature=0.1`, `response_format={"type":"json_object"}`, `"think": false`, and `max_tokens` from config.

### Detection scopes

Two modes, both keyed in `SCOPES`:
- `industrial` — closed taxonomy `manometro|termometro|valvula|sensor|epp|otro`, each with free-text `descripcion`. More reliable.
- `todo` — any object, free-form `tipo`. Looser; output varies more between runs.

An empty `{"objetos": []}` means the model genuinely saw nothing matching the scope — not a parsing bug. (Earlier `industrial` empties on thermometer/sensor images were the taxonomy being too narrow, since fixed.)

### config.json

Created on first run from `DEFAULT_CONFIG`. The CLI scripts read it for their defaults; any flag overrides for that run only (does not write back). The menu writes choices back. Keys: `model`, `image`, `folder`, `scope`, `max_tokens`, `think`, `url`, `benchmark_models`, `benchmark_runs`.

### Evaluation targets (from internal doc "F1.8")

The benchmark exists to pick the primary VLM for the PoC against three criteria: detection precision, **P95 latency < 1.5 s on-prem**, and a high valid-JSON rate for the VLM→VLA contract. Default model lineup compared: `qwen3-vl:8b`, `qwen3-vl:4b`, `qwen2.5vl:7b`.

## Data

`fotosClean/` holds the lab test images (`*.jpeg`). Supported extensions in the benchmark: jpg, jpeg, png, bmp, webp.
