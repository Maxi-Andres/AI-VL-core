# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A two-script PoC harness for evaluating vision-language models (VLMs) served by **Ollama** for industrial inspection (mining / oil & gas). The VLM is prompted in Spanish (Rioplatense) to detect industrial instruments — manómetros (pressure gauges), válvulas (valves), and EPP (personal protective equipment) — and return structured JSON with normalized bounding boxes. The JSON contract is meant to feed a downstream VLA stage and is consumed in production by the "Silk AI Proxy Gateway / F1.9".

## Commands

```bash
pip install requests                 # only dependency

# Smoke test: one image, prints latency + parsed JSON
python3 03_smoke_test.py fotosClean/1.jpeg
python3 03_smoke_test.py fotosClean/1.jpeg --model qwen3-vl:4b

# Benchmark: all images in a folder, N runs per image, across models
python3 04_benchmark.py ./fotosClean --runs 3
python3 04_benchmark.py ./fotosClean --models qwen3-vl:8b qwen3-vl:4b
```

Prerequisite: an Ollama server reachable at `http://localhost:11434` with the target models pulled. Check with `curl http://localhost:11434/api/version`. Override the endpoint on either script with `--url`.

## Architecture

Both scripts hit Ollama's **OpenAI-compatible** endpoint (`/v1/chat/completions`), not the native Ollama API. Images are base64-encoded and sent as a `data:image/jpeg;base64,...` `image_url` content part. Requests use `temperature=0.1` and `max_tokens=768`.

The two scripts deliberately duplicate the core pieces (system/user prompt, image encoding, JSON extraction) rather than sharing a module — keep that in mind when changing prompt wording or the JSON schema, as edits must be mirrored in **both** files to keep results comparable:

- `03_smoke_test.py` — single-image sanity check. Has the richer prompt (asks for `lectura`/reading and `confianza`/confidence per object) and a forgiving `extract_json` that strips markdown fences and falls back to a balanced-brace regex.
- `04_benchmark.py` — latency + reliability benchmark. Computes P50/P95/mean latency and JSON-valid rate per model, prints a comparison table, and writes `benchmark_resultados.json`. Uses a terser prompt and only checks JSON validity (`is_valid_json`) without keeping the parsed payload.

### Evaluation targets (from internal doc "F1.8")

The benchmark exists to pick the primary VLM for the PoC against three criteria: detection precision, **P95 latency < 1.5 s on-prem**, and a high valid-JSON rate for the VLM→VLA contract. Default model lineup compared: `qwen3-vl:8b`, `qwen3-vl:4b`, `qwen2.5vl:7b`.

## Data

`fotosClean/` holds the lab test images (`*.jpeg`). Supported extensions in the benchmark: jpg, jpeg, png, bmp, webp.
