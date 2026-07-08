# CLAUDE.md

Guidance for Claude Code when working in the **iacore** repo (`AI-VL-core`).

## What this is

The **inference core** of the live-video PoC: it evaluates vision-language models
(VLMs) served by **Ollama** and an in-process **Ultralytics YOLO** detector for
industrial inspection (mining / oil & gas). It returns structured JSON with
normalized bounding boxes that feeds a downstream VLA stage (consumed in production
by the "Silk AI Proxy Gateway / F1.9").

Two detection paths, built as mirror images of each other:
- **VLM path** (`vlm_*` files): the Ollama VLM. Open-vocabulary, reasons over a
  prompt, slower. Invoked when the user asks a question about a frame.
- **YOLO path** (`yolo_*` files): the in-process Ultralytics detector that the real
  deployment runs on the **live video stream**. Emits the **same JSON contract** as
  the VLM (`type`/`description` = class label, `reading` = null).

## Three-app boundary

iacore is one of THREE independent apps, each its own git repo, that communicate
**over the network by port, never by file path** (each may run on a different
machine):

```
frontend (browser UI)  ──HTTP/WS──▶  backend (gateway)  ──HTTP──▶  iacore (this repo)
```

`service.py` (FastAPI, `uvicorn service:app --port 8001`) is the **ONLY** networked
surface — it exposes detection over HTTP (`/detect`, `/vlm`, `/vlm/stream`) plus
speech (`/transcribe`, `/speak`, `/tts/voices`) and metadata (`/options`,
`/classes`, `/health`). The CLI (`menu.py`) runs the same code locally on single
images. `service.py`/`menu.py` import `src/` locally — fine, same repo. Do **not**
add a Python import or filesystem path to the other two apps; they only know each
other's URLs (env-configured). The CLI and `src/` stay deployment-agnostic.

## Code conventions

- **Everything in this repo is in English — absolutely everything**: comments,
  docstrings, identifiers/function names, all user-facing strings (menu text,
  prints, argparse help, table headers), the README, the VLM prompts, and the JSON
  output keys. **The only file that must NOT be translated/touched is `FIX.txt`.**
  The user converses in Spanish (Rioplatense) — that's fine for chat only; never
  reintroduce Spanish into the code or docs.
- **VLM→VLA JSON contract (external, do not break silently):** keys `objects`,
  `type`, `description`, `reading`, `confidence`, `bbox`; the `type` family enum is
  `pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other`;
  the second detection scope is `all` (was renamed from `todo`). Consumed downstream
  by "Silk AI Proxy Gateway / F1.9" — the consumer must match these English keys.
- **NEVER run `git commit` or `git push`.** The user reviews every change manually
  before committing. Make edits, verify, report — leave committing to the user.

## Layout

Code lives in `src/`; `menu.py` (the entry point) stays at the repo root and
bootstraps `src/` onto `sys.path`. The `src/` scripts import each other as siblings
(so they also run directly). VLM and YOLO files live side by side in `src/`
(prefixed `vlm_`/`yolo_`, no sub-folders) and share one `config.json` and one
`results/` folder. All entry points share **`src/vlm_common.py`** — change prompts,
the JSON schema, or the Ollama client there and every entry point picks it up.
Heavy optional deps (`ultralytics`, `faster_whisper`, `piper`) are **lazy-imported**
so each path works without the others' deps. Test images live under `fotos/`;
benchmark output goes to `results/` (git-ignored).

For the exact module/function map, endpoints, and call graph, query the
**codebase-memory** graph (`get_architecture`, `search_graph`, `trace_path`) rather
than a hand-maintained list here.

## Running

`./setup.sh` builds `.venv/` from `requirements.txt` (system Python is 3.14 /
PEP 668, so deps live in the venv). Then `python3 menu.py` (interactive) or run the
`src/*_scan.py` / `*_benchmark.py` scripts directly. Prerequisite: an Ollama server
at `http://localhost:11434` with the target models pulled
(`curl http://localhost:11434/api/version`; `ollama ps` shows GPU/CPU split). See
`README.md` for the full CLI/flag reference.

## Deep VLM/model tuning → the `ollama-vlm-tuning` skill

VRAM/GPU-split math, `qwen3-vl` Thinking-vs-Instruct checkpoints, the unreliable
`think` flag, `num_ctx`/`num_predict` interaction and token starvation, the native
`/api/chat` rationale, bbox normalization, detection scopes, the `config.json` key
catalog, and the F1.8 targets now live in the **`ollama-vlm-tuning`** skill (loads
on demand). Read it before choosing a model, editing benchmark sweeps, or debugging
empty/truncated VLM output. Long-form design/training docs are in
`docs/ARCHITECTURE.md` and `docs/YOLO.md`.
