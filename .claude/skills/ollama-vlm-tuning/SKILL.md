---
name: ollama-vlm-tuning
description: >-
  Deep reference for tuning the VLM/YOLO inference in AI-VL-core (iacore): which
  Ollama model to pick, VRAM/GPU-split math, the qwen3-vl Thinking-vs-Instruct
  checkpoints and the unreliable `think` flag, num_ctx/num_predict interaction and
  token starvation, the native /api/chat rationale, bbox normalization, detection
  scopes, the config.json key catalog, and the F1.8 evaluation targets. Read this
  BEFORE choosing a model, changing benchmark sweeps, debugging empty/truncated VLM
  output, adjusting max_tokens/num_ctx, or reasoning about latency/VRAM on the GPU.
---

# Ollama / VLM tuning for iacore

This is consult-on-demand knowledge that used to live in `CLAUDE.md`. The always-on
rules (English-only, never-commit, the app boundary, the JSON contract) stay in
`CLAUDE.md`; the deep tuning knowledge lives here.

## Model / hardware constraint (important)

The scan default is `qwen3-vl:4b` (not 8b) because of VRAM. On an 8 GB GPU (e.g.
RTX 5060), `qwen3-vl:8b` does **not** fit and Ollama splits it ~53% CPU / 47% GPU,
pushing latency to ~85–110 s/image. `qwen3-vl:4b` (~3.3 GB) loads 100% on GPU and
runs at ~18–25 s/image. Verify the split with `ollama ps`.

## Thinking is decided by the MODEL, not the `think` flag

Verified empirically against Ollama 0.30.6. The flag is *not* a reliable on/off;
what matters is the model's `thinking` capability (see `ollama show <model>` /
`/api/show`):

- `qwen3-vl:4b` / `:8b` — capability `thinking`, renderer `qwen3-vl-thinking`.
  **Always reasons**; passing `"think": false` is silently **ignored** on 0.30.6
  (the renderer still emits a full `<think>` block — confirmed: `think:false`
  produced *more* reasoning than `think:true`, i.e. pure variance).
- `qwen2.5vl:7b` — **no** `thinking` capability. Never reasons, and sending
  `"think": true` returns **HTTP 400 `"does not support thinking"`**. So the flag
  must be **omitted** for such models or the request fails (this previously made
  the whole qwen2.5vl benchmark error out at 100%).

So `vlm_common.model_supports_thinking()` queries `/api/show` (cached) and
`query_vlm` only includes `think` in the payload when the model supports it.
`query_vlm` returns `think_requested` / `thinking_supported` / `did_think`;
`describe_thinking()` turns those into an honest one-liner shown by the scan, and
the benchmark prints which models reason and which don't. Since `qwen3-vl` thinking
always reasons, its real failure mode is **token starvation**: if reasoning
consumes the whole output budget, `content` comes back **empty** with
`finish_reason: "length"`. The fix is budget — give it room with `--max-tokens`
(`num_predict`) **and** `--num-ctx`.

## The real reasoning switch is model choice — two separate checkpoints, not a flag

Qwen publishes `Qwen3-VL-*-Thinking` (always reasons) and `Qwen3-VL-*-Instruct`
(never reasons) as different weights. Ollama's `qwen3-vl:4b` is the **Thinking**
checkpoint (same blob id as `qwen3-vl:4b-thinking`, `1343d82ebee3`). The
**Instruct** sibling is in Ollama's registry too. To toggle reasoning within the
same family, switch models — no Ollama upgrade needed:

- `qwen3-vl:4b-instruct` (= `-instruct-q4_K_M`, 3.3 GB) — no `thinking` capability
  → never reasons, fits 100% on the 8 GB GPU. **Recommended primary for the PoC**
  (faster, no token starvation, structured-JSON task doesn't benefit from
  chain-of-thought).
- `qwen3-vl:4b-thinking` (= `qwen3-vl:4b`) — keep for comparison / ambiguous images.
- Quantization on 8 GB: prefer `q4_K_M` (3.3 GB, 100% GPU). `q8_0` (5.1 GB) fits
  tightly — verify with `ollama ps`. `bf16` (8.9 GB) does NOT fit (CPU/GPU split →
  slow). `qwen2.5vl:7b` is a different (older) generation with no `thinking`
  capability.

Because the benchmark sweeps `--models`, comparing thinking-vs-instruct (and
quantizations) is just
`python3 src/vlm_benchmark.py fotos/clean --models qwen3-vl:4b-instruct qwen3-vl:4b-thinking`.
See the README "Models (Ollama)" section for the full variant table.

## The two interacting budget knobs

- **`max_tokens` / `num_predict`** = ceiling on tokens the model **generates**
  (thinking + answer).
- **`num_ctx`** = the full context window (input + output), and the value
  `ollama ps` shows under *context*. The image+prompt input alone is ~1000–2600
  tokens, so `num_ctx` must exceed input + desired output or the answer gets
  squeezed out. Bonus: a larger `num_ctx` also lets Ollama feed the image at
  **higher resolution** (more image tokens → more detail).

Reasoning is **streamed live**: the scan (and menu) print the model's thinking
dimmed in real time (`verbose=True` in `query_vlm`). `--think`/`--no-think` toggle
requesting reasoning; default is on so you can watch it.

## Native /api/chat endpoint (not OpenAI-compat)

Both VLM scripts hit Ollama's **native** endpoint (`/api/chat`) with `stream:
true`, **not** the OpenAI-compatible `/v1/chat/completions`. Deliberate: the OpenAI
endpoint ignores `think` and mixes reasoning into `content` (so a thinking model
can return empty `content`), whereas the native endpoint puts reasoning in a
separate `thinking` field and the JSON answer cleanly in `content`. Streaming is
what lets us print the reasoning live. Images go in the message's `images:
[base64]` array (raw base64, no data-URI prefix). Requests use `format: "json"` and
`options: {temperature: 0.1, num_predict: max_tokens, num_ctx}`. `host_of()`
normalizes any `url` (including an old `…/v1/chat/completions` from a stale
`config.json`) down to the base host before appending `/api/chat`.

## Bounding boxes

qwen3-vl returns bbox in **absolute pixel** coordinates of the source image, not
normalized. `query_vlm` normalizes them to 0–1 via `normalize_bboxes()`, using
`image_size()` (a dependency-free JPEG/PNG header reader) — any bbox value > 1 is
treated as pixels and divided by width/height. Callers must pass
`size=image_size(path)`.

## Detection scopes

Two modes, both keyed in `SCOPES` (in `src/vlm_common.py`):

- `industrial` — open detection of **any** industrial instrument. `type` is a
  coarse family (`pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other`)
  and `description` is free text for the specifics. The prompt lists per-family
  examples (pressure gauge, thermocouple, Coriolis flow meter, radar sensor, etc.)
  as *reference, not a closed list*, and tells the model not to over-deliberate on
  the category — this is what stops it from burning reasoning tokens deciding where
  a flow meter "belongs".
- `all` — any object, free-form `type`. Looser; output varies more between runs.

An empty `{"objects": []}` means the model genuinely saw nothing matching the scope
— not a parsing bug. (Earlier `industrial` empties on thermometer/sensor images
were the taxonomy being too narrow, since fixed.) A `people`-oriented `variant`
also exists for people/PPE-focused prompts.

## config.json key catalog

Lives at the repo root (path computed from `PROJECT_ROOT`, not next to
`vlm_common.py` inside `src/`). Created on first run from `DEFAULT_CONFIG`. The CLI
scripts read it for their defaults; any flag overrides for that run only (does not
write back). The menu writes choices back.

- **Scan keys**: `model`, `image`, `folder`, `scope`, `variant`, `max_tokens`,
  `num_ctx`, `think`, `url`.
- **Benchmark block (independent)**: `benchmark_models`, `benchmark_runs`,
  `benchmark_images` (`[]` = all in folder), `benchmark_scope`,
  `benchmark_variants`, `benchmark_max_tokens`, `benchmark_num_ctx`,
  `benchmark_think`. **The five swept dimensions are all lists** —
  `benchmark_models`, `benchmark_variants`, `benchmark_max_tokens`,
  `benchmark_num_ctx`, `benchmark_think` (e.g. `[4096, 8192]`); a single-element
  list disables sweeping that dimension. (Old singular/scalar configs are coerced
  to lists at runtime.)
- **YOLO keys** (same file, added on first run by `yolo_common.load_config`, which
  wraps the VLM `load_config`): scan keys `yolo_model`, `yolo_image`,
  `yolo_folder`, `yolo_conf`, `yolo_imgsz`, `yolo_save`, `yolo_classes` (class
  filter, e.g. `["person"]`); benchmark keys `yolo_benchmark_models`,
  `yolo_benchmark_runs`, `yolo_benchmark_images`, `yolo_benchmark_conf` (list),
  `yolo_benchmark_imgsz` (list). No `url`/`think`/prompt for YOLO (runs in-process,
  no Ollama).

## Evaluation targets (from internal doc "F1.8")

The benchmark exists to pick the primary VLM for the PoC against three criteria:
detection precision, **P95 latency < 1.5 s on-prem**, and a high valid-JSON rate
for the VLM→VLA contract. Historical lineup compared: `qwen3-vl:8b`, `qwen3-vl:4b`,
`qwen2.5vl:7b`.
