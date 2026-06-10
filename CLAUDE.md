# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PoC harness for evaluating vision-language models (VLMs) served by **Ollama** for industrial inspection (mining / oil & gas). The VLM detects instruments/objects and returns structured JSON with normalized bounding boxes. The JSON contract is meant to feed a downstream VLA stage and is consumed in production by the "Silk AI Proxy Gateway / F1.9".

## Code conventions

- **Everything in this repo is in English — absolutely everything**: comments, docstrings, identifiers/function names, all user-facing strings (menu text, prints, argparse help, table headers), the README, the VLM prompts, and the JSON output keys. **The only file that must NOT be translated/touched is `FIX.txt`.** The user converses in Spanish (Rioplatense) — that's fine for chat only; never reintroduce Spanish into the code or docs.
- The VLM→VLA JSON contract was **also translated to English** (it used to be Spanish): output keys are now `objects`, `type`, `description`, `reading`, `confidence`, `bbox`; the `type` family enum values are `pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other`. The second detection scope was renamed `todo` → `all`. NOTE: this is an external contract consumed downstream by "Silk AI Proxy Gateway / F1.9" — the downstream consumer must be updated to match these English keys.
- **NEVER run `git commit` or `git push`.** The user reviews every change manually before committing themselves. Make edits, verify, report — leave committing to the user.

## Commands

```bash
pip install -r requirements.txt      # only dependency: requests

# Interactive menu (main entry point) — pick model/image/scope, persists to config.json
python3 menu.py

# Smoke test: one image, prints latency + parsed JSON
python3 src/smoke_test.py fotos/clean/1.jpeg
python3 src/smoke_test.py fotos/clean/1.jpeg --model qwen3-vl:4b --scope todo

# Benchmark: sweeps the cartesian product of models × prompts × max_tokens × num_ctx × think
python3 src/benchmark.py ./fotos/clean --runs 3
python3 src/benchmark.py ./fotos/clean --models qwen3-vl:8b qwen3-vl:4b
python3 src/benchmark.py ./fotos/clean --variants v1_original v2_antiloop   # A/B prompts
python3 src/benchmark.py ./fotos/clean --max-tokens 4096 8192 --num-ctx 8192 16384
python3 src/benchmark.py ./fotos/clean --think true false                   # compare reasoning on/off
```

The code lives in `src/`; `menu.py` (the entry point) stays at the repo root and
bootstraps `src/` onto `sys.path`. The `src/` scripts import each other as
siblings, so they also run directly (`python3 src/benchmark.py …`).

See `README.md` for the full flag reference.

Prerequisite: an Ollama server reachable at `http://localhost:11434` with the target models pulled. Check with `curl http://localhost:11434/api/version`. Override the endpoint on either script with `--url`. Use `ollama ps` to see whether a model loaded fully on GPU or split CPU/GPU.

### Model / hardware constraint (important)

The smoke-test default is `qwen3-vl:4b` (not 8b) because of VRAM. On an 8 GB GPU (e.g. RTX 5060), `qwen3-vl:8b` does **not** fit and Ollama splits it ~53% CPU / 47% GPU, pushing latency to ~85–110 s/image. `qwen3-vl:4b` (~3.3 GB) loads 100% on GPU and runs at ~18–25 s/image. Verify the split with `ollama ps`.

**Thinking is decided by the MODEL, not the `think` flag** (verified empirically against Ollama 0.30.6). The flag is *not* a reliable on/off; what matters is the model's `thinking` capability (see `ollama show <model>` / `/api/show`):
- `qwen3-vl:4b` / `:8b` — capability `thinking`, renderer `qwen3-vl-thinking`. **Always reasons**; passing `"think": false` is silently **ignored** on 0.30.6 (the renderer still emits a full `<think>` block — confirmed: `think:false` produced *more* reasoning than `think:true`, i.e. pure variance).
- `qwen2.5vl:7b` — **no** `thinking` capability. Never reasons, and sending `"think": true` returns **HTTP 400 `"does not support thinking"`**. So the flag must be **omitted** for such models or the request fails (this previously made the whole qwen2.5vl benchmark error out at 100%).

So `vlm_common.model_supports_thinking()` queries `/api/show` (cached) and `query_vlm` only includes `think` in the payload when the model supports it. `query_vlm` returns `think_requested` / `thinking_supported` / `did_think`; `describe_thinking()` turns those into an honest one-liner shown by the smoke test, and the benchmark prints which models reason and which don't. Since `qwen3-vl` thinking always reasons, its real failure mode is **token starvation**: if reasoning consumes the whole output budget, `content` comes back **empty** with `finish_reason: "length"`. The fix is budget — give it room with `--max-tokens` (`num_predict`) **and** `--num-ctx`.

**The real reasoning switch is model choice — and Qwen3-VL ships as two separate checkpoints, not a flag.** Qwen publishes `Qwen3-VL-*-Thinking` (always reasons) and `Qwen3-VL-*-Instruct` (never reasons) as different weights. Ollama's `qwen3-vl:4b` is the **Thinking** checkpoint (same blob id as `qwen3-vl:4b-thinking`, `1343d82ebee3`). The **Instruct** sibling is in Ollama's registry too. So to actually toggle reasoning within the same family, switch models — no Ollama upgrade needed:
- `qwen3-vl:4b-instruct` (= `-instruct-q4_K_M`, 3.3 GB) — no `thinking` capability → never reasons, fits 100% on the 8 GB GPU. **Recommended primary for the PoC** (faster, no token starvation, structured-JSON task doesn't benefit from chain-of-thought).
- `qwen3-vl:4b-thinking` (= `qwen3-vl:4b`) — keep for comparison / ambiguous images.
- Quantization on 8 GB: prefer `q4_K_M` (3.3 GB, 100% GPU). `q8_0` (5.1 GB) fits tightly — verify with `ollama ps`. `bf16` (8.9 GB) does NOT fit (CPU/GPU split → slow). `qwen2.5vl:7b` is a different (older) generation with no `thinking` capability.

Because the benchmark sweeps `--models`, comparing thinking-vs-instruct (and quantizations) is just `python3 src/benchmark.py fotos/clean --models qwen3-vl:4b-instruct qwen3-vl:4b-thinking`. See README "Modelos (Ollama)" for the full variant table.

Two knobs, and they interact:
- **`max_tokens` / `num_predict`** = ceiling on tokens the model **generates** (thinking + answer).
- **`num_ctx`** = the full context window (input + output), and the value `ollama ps` shows under *context*. The image+prompt input alone is ~1000–2600 tokens, so `num_ctx` must exceed input + desired output or the answer gets squeezed out. Bonus: a larger `num_ctx` also lets Ollama feed the image at **higher resolution** (more image tokens → more detail).

Reasoning is **streamed live**: the smoke test (and menu) print the model's thinking dimmed in real time (`verbose=True` in `query_vlm`). `--think`/`--no-think` toggle requesting reasoning; default is on so you can watch it.

## Architecture

All entry points share **`src/vlm_common.py`** — there is no longer any prompt/client duplication. Change prompts, the JSON schema, or the Ollama client in that one module and every entry point picks it up. Layout:

- `src/vlm_common.py` — the core. Holds `SCOPES` (the detection-mode prompts), `query_vlm()` (the streaming Ollama client), `extract_json()`/`render_result()`, `image_size()`/`normalize_bboxes()` (bbox post-processing), the `config.json` load/save (`DEFAULT_CONFIG`, `load_config`, `save_config`), and `PROJECT_ROOT`/`RESULTS_DIR`/`results_path()` (paths resolve relative to the repo root, not `src/`).
- `menu.py` — interactive launcher at the repo root (run with no args / IDE "Play"). Inserts `src/` on `sys.path`, then imports `run_smoke`/`run_benchmark` as normal modules (no more `importlib` numeric-name hack — the scripts dropped their `0N_` prefixes). Reads/writes `config.json`, lists Ollama models via `/api/tags`, lists images from the folder.
- `src/smoke_test.py` — single-image check. Exposes `run_smoke(...)`; `main()` just parses args (defaults pulled from `config.json`) and calls it.
- `src/benchmark.py` — latency/reliability benchmark **and** prompt A/B. Exposes `run_benchmark(...)`; sweeps the **full cartesian product of models × prompt-variants × max_tokens × num_ctx × think** (each combination is one row), computes P50/P95/mean latency, JSON-valid rate, length-truncations and avg objects per combination, prints a comparison table (with `ctx`/`maxtok`/`thk` columns, and the Model/Prompt columns auto-size to the longest value so long model names stay aligned) + a best-combination verdict that prints the full winning config. Each run is written to its OWN timestamped file `results/benchmark_<YYYYMMDD_HHMMSS>.json` (override with `--out`) so runs never overwrite each other. The saved payload includes both the metrics and, per combination, a `detections` map of `image -> [per-run {result/raw, ok, finish_reason, elapsed_s}]` capturing what the model actually returned (image file names are never sent to the model — only the bytes + fixed prompt). `max_tokens`/`num_ctx`/`think` accept a scalar or a list (helpers `as_list`/`parse_bool`/`dedup`); CLI flags `--max-tokens`/`--num-ctx`/`--think` are all `nargs="+"` (e.g. `--think true false`). (This subsumes the old `05_prompt_test.py`, which was removed: prompt variants are now compared the same way models are.)

Both scripts hit Ollama's **native** endpoint (`/api/chat`) with `stream: true`, **not** the OpenAI-compatible `/v1/chat/completions`. This was a deliberate switch: the OpenAI endpoint ignores `think` and mixes reasoning into `content` (so a thinking model can return empty `content`), whereas the native endpoint puts reasoning in a separate `thinking` field and the JSON answer cleanly in `content`. Streaming is what lets us print the reasoning live. Images go in the message's `images: [base64]` array (raw base64, no data-URI prefix). Requests use `format: "json"` and `options: {temperature: 0.1, num_predict: max_tokens, num_ctx}`. `host_of()` normalizes any `url` (including an old `…/v1/chat/completions` from a stale `config.json`) down to the base host before appending `/api/chat`.

**Bounding boxes:** qwen3-vl returns bbox in **absolute pixel** coordinates of the source image, not normalized. `query_vlm` normalizes them to 0–1 via `normalize_bboxes()`, using `image_size()` (a dependency-free JPEG/PNG header reader) — any bbox value > 1 is treated as pixels and divided by width/height. Callers must pass `size=image_size(path)`.

### Detection scopes

Two modes, both keyed in `SCOPES`:
- `industrial` — open detection of **any** industrial instrument. `type` is a coarse family (`pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other`) and `description` is free text for the specifics. The prompt lists per-family examples (pressure gauge, thermocouple, Coriolis flow meter, radar sensor, etc.) as *reference, not a closed list*, and tells the model not to over-deliberate on the category — this is what stops it from burning reasoning tokens deciding where a flow meter "belongs".
- `all` — any object, free-form `type`. Looser; output varies more between runs.

An empty `{"objects": []}` means the model genuinely saw nothing matching the scope — not a parsing bug. (Earlier `industrial` empties on thermometer/sensor images were the taxonomy being too narrow, since fixed.)

### config.json

Lives at the repo root (path computed from `PROJECT_ROOT`, not next to `vlm_common.py` inside `src/`). Created on first run from `DEFAULT_CONFIG`. The CLI scripts read it for their defaults; any flag overrides for that run only (does not write back). The menu writes choices back. Smoke-test keys: `model`, `image`, `folder`, `scope`, `variant`, `max_tokens`, `num_ctx`, `think`, `url`. Benchmark has its own independent block: `benchmark_models`, `benchmark_runs`, `benchmark_images` (`[]` = all in folder), `benchmark_scope`, `benchmark_variants`, `benchmark_max_tokens`, `benchmark_num_ctx`, `benchmark_think`. **The five swept dimensions are all lists** — `benchmark_models`, `benchmark_variants`, `benchmark_max_tokens`, `benchmark_num_ctx`, `benchmark_think` (e.g. `[4096, 8192]`); a single-element list disables sweeping that dimension. (Note the plurals: these replaced the old singular `benchmark_variant`/scalar `benchmark_max_tokens`/`benchmark_num_ctx`/`benchmark_think`; old scalar configs are still coerced to lists at runtime.)

### Evaluation targets (from internal doc "F1.8")

The benchmark exists to pick the primary VLM for the PoC against three criteria: detection precision, **P95 latency < 1.5 s on-prem**, and a high valid-JSON rate for the VLM→VLA contract. Default model lineup compared: `qwen3-vl:8b`, `qwen3-vl:4b`, `qwen2.5vl:7b`.

## Data

Image folders live under `fotos/`: `fotos/clean/` holds the lab test images (`*.jpeg`) and `fotos/ciudad/` an additional set. Supported extensions in the benchmark: jpg, jpeg, png, bmp, webp. Benchmark output (JSON) is written to `results/` (git-ignored except for the folder itself), not the repo root.
