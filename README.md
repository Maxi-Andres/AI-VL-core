# VLM PoC — Industrial inspection with Ollama

Detection of instruments/objects in images using a VLM (qwen3-vl) served
by Ollama, returning JSON with bounding boxes. Designed to validate the primary
VLM of the PoC (F1.8 criteria) and the VLM→VLA contract.

## Project structure

```
.
├── menu.py                # main entry point (asks VLM vs YOLO, then a menu)
├── src/                   # source code (VLM and YOLO paths, side by side)
│   ├── vlm_common.py      # VLM core: prompts, Ollama client, parsing, config
│   ├── vlm_scan.py        # VLM 1-image scan (CLI)
│   ├── vlm_benchmark.py   # VLM latency/JSON benchmark + prompt A/B (CLI)
│   ├── yolo_common.py     # YOLO core: ultralytics loader, detection, config
│   ├── yolo_scan.py       # YOLO 1-image scan (CLI; saves the boxed image)
│   └── yolo_benchmark.py  # YOLO latency benchmark (models × imgsz × conf)
├── fotos/                 # image folders
│   ├── clean/             # lab test set
│   └── ciudad/            # additional set
├── results/               # benchmark output (JSON)
├── config.json            # persistent configuration (created on its own)
└── requirements.txt
```

Two detection paths share one `config.json` and one `results/` folder:

- **VLM** (`vlm_*`): the Ollama vision-language model — open-vocabulary, reasons
  over a prompt, slower. Picks the primary VLM (F1.8 criteria).
- **YOLO** (`yolo_*`): an in-process Ultralytics detector — what the real
  deployment runs on the **live video stream** (here: single images only; video
  needs a WebSocket + front/back-end, out of scope). Same JSON contract as the
  VLM. Weights (`*.pt`) auto-download on first use and are git-ignored.
  **Full YOLO guide** (usage, tuning variables, training, model differences):
  [`docs/YOLO.md`](docs/YOLO.md).

## Live service (3-app architecture)

This repo is the **inference core**, one of three independent apps that talk over
the network **by port, never by file path** (each can run on a different machine):

```
frontend (browser UI)  ──HTTP/WS──▶  backend (gateway)  ──HTTP──▶  iacore (this repo)
```

The CLI (`menu.py`) still runs everything locally on single images. `service.py`
additionally exposes the **same** YOLO + VLM detection over HTTP so the backend
gateway can reach it — this is the live-video path that was previously out of
scope. Run the service:

```bash
source .venv/bin/activate
uvicorn service:app --host 0.0.0.0 --port 8001
```

Endpoints: `GET /health`, `GET /options`, `GET /classes?model=`,
`POST /detect` (raw image bytes → boxes), `POST /vlm` (base64 image → VLM JSON).
The browser never hits this service directly; only the backend does. The backend
and frontend live in their own repos (`../backend`, `../frontend`).

## Setup (clone & run)

After `git clone`, run the setup script once — it creates an isolated virtual
environment (`.venv/`) and installs everything. It does **not** touch the system
Python (which on many distros, including this one with Python 3.14, is
"externally-managed" and rejects `pip install`).

```bash
git clone <repo> && cd "VL test"
./setup.sh                 # creates .venv/ and installs requirements
source .venv/bin/activate  # then use plain `python` / `yolo`
python menu.py
```

The `.venv/` folder is **git-ignored on purpose** (it's machine-specific — compiled
binaries, absolute paths, ~GB). It is **not** in the repo; `./setup.sh` rebuilds it
from `requirements.txt` on any machine, so a fresh clone always works. `setup.sh`
even handles platforms whose `venv` lacks `ensurepip` (it bootstraps `pip` via
`get-pip.py`, no sudo/apt needed).

> Manual alternative (if you prefer): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

### Requirements

- The **YOLO path** needs `ultralytics` (pulls in torch/opencv); the VLM path does
  not. If you only use the VLM, `requests` alone is enough.

- **Ollama** running at `http://localhost:11434` with the models downloaded
  (`ollama pull qwen3-vl:4b`, etc.).
- Verify it's alive: `curl http://localhost:11434/api/version`
- Verify the model loads **100% on GPU**: `ollama ps` (see the VRAM note below).

## Models (Ollama)

Qwen3-VL comes in **two separate checkpoints** (it's not a flag): **Thinking**
(always reasons) and **Instruct** (doesn't reason). Ollama's `qwen3-vl:4b` **is the
Thinking one** (same blob as `qwen3-vl:4b-thinking`). To turn reasoning on/off you
**switch models**, not the flag (see "Reasoning" below).

**4B** variants in the Ollama registry (all Text+Image, 256K context):

| tag | size | quantization | on an 8 GB GPU |
|-----|-------:|--------------|----------------|
| **`qwen3-vl:4b-instruct`**  (= `-instruct-q4_K_M`) | 3.3 GB | Q4_K_M | ✅ 100% GPU — **recommended** |
| `qwen3-vl:4b-instruct-q8_0` | 5.1 GB | Q8_0  | ⚠ barely fits; check `ollama ps` |
| `qwen3-vl:4b-instruct-bf16` | 8.9 GB | BF16  | ❌ doesn't fit (splits CPU/GPU) |
| `qwen3-vl:4b-thinking`  (= `qwen3-vl:4b`, `-thinking-q4_K_M`) | 3.3 GB | Q4_K_M | ✅ 100% GPU |
| `qwen3-vl:4b-thinking-q8_0` | 5.1 GB | Q8_0  | ⚠ barely fits |
| `qwen3-vl:4b-thinking-bf16` | 8.9 GB | BF16  | ❌ doesn't fit |

Others that exist / are installed: `qwen3-vl:8b`, `qwen3-vl:8b-instruct`,
`qwen2.5vl:7b` (the latter **without** the `thinking` capability: it never reasons, but it's
another generation).

**Recommended for the PoC: `qwen3-vl:4b-instruct`** — it doesn't reason (faster, no
token-starvation) and fits 100% on GPU. Download it with:

```bash
ollama pull qwen3-vl:4b-instruct
```

To compare thinking vs not-thinking within the same family, run the benchmark with
both: `python3 src/vlm_benchmark.py fotos/clean --models qwen3-vl:4b-instruct qwen3-vl:4b-thinking`.

## Quick start: the menu (without typing commands)

```bash
python3 menu.py        # or just hit Play on the file in the IDE
```

The menu has two ways to analyze:

1. **Scan** (option 1): one image, prints the reasoning live + JSON.
2. **Benchmark** (option 2): opens a **submenu** where you choose **which images**
   (which ones and how many), **which models**, **which prompts** (one or several variants),
   how many **runs**, and **one or several** values of `max_tokens`, `num_ctx` and `think`.
   It runs the **Cartesian product** of all those dimensions with a **progress
   bar** and at the end reports **times** (per image, total, average,
   P50/P95), the **valid-JSON rate** and a verdict
   with the best combination. Comparing several prompts here replaces the old
   `prompt_test`: prompts are tested just like models, all together.

**Everything you choose is saved to `config.json`**, so next time it starts
with whatever you last used. The benchmark has its **own** config (the
`benchmark_*` keys), independent of the scan: you can run the benchmark with a
lighter context (faster) without lowering the scan's context.

---

## Use without the menu (command line)

Both scripts take their **defaults from `config.json`**; any flag overrides that
default only for that run (it doesn't modify the file).

### Scan — one image

```bash
python3 src/vlm_scan.py                          # uses everything from config.json
python3 src/vlm_scan.py fotos/clean/5.jpeg        # another image
python3 src/vlm_scan.py fotos/clean/5.jpeg --model qwen3-vl:8b
python3 src/vlm_scan.py fotos/clean/5.jpeg --scope all
python3 src/vlm_scan.py fotos/clean/5.jpeg --max-tokens 8192 --num-ctx 16384
python3 src/vlm_scan.py fotos/clean/5.jpeg --no-think     # request think=false
```

The scan **prints live what the model is thinking** (in gray) and at the
end the JSON, the latency and the input/output tokens.

| Flag            | Default (config.json) | What it does |
|-----------------|-----------------------|----------|
| `image` (pos.)  | `image`               | Path to the image. If you omit it, it uses the one from config. |
| `--model`       | `model`               | Ollama model (e.g. `qwen3-vl:4b`, `qwen3-vl:8b`, `qwen2.5vl:7b`). |
| `--scope`       | `scope`               | `industrial` (industrial instruments) or `all` (any object). |
| `--variant`     | `variant`             | Prompt variant (`v1_original`, `v2_antiloop`, …). See "Prompt variants". |
| `--max-tokens`  | `max_tokens`          | Cap on **output** tokens (`num_predict`; includes the reasoning). |
| `--num-ctx`     | `num_ctx`             | Context window (input+output); the one `ollama ps` shows. |
| `--think` / `--no-think` | `think`      | Request reasoning/no. **Only applies to models with the `thinking` capability** and is only sent if the model has it. On `qwen3-vl` (0.30.6) `--no-think` is ignored (it reasons anyway); to avoid reasoning, use `qwen2.5vl:7b`. |
| `--url`         | `url`                 | Ollama host. |

### Benchmark — set of images (times + P50/P95 + % valid JSON)

The benchmark **sweeps the Cartesian product of ALL these dimensions** —
models, prompts, images, `max_tokens`, `num_ctx` and `think` — and each
combination is a row of the report. All of them accept **several values**:

```bash
python3 src/vlm_benchmark.py                                    # uses everything from config.json (benchmark_* keys)
python3 src/vlm_benchmark.py fotos/clean --runs 5
python3 src/vlm_benchmark.py fotos/clean --models qwen3-vl:4b qwen3-vl:8b      # several models
python3 src/vlm_benchmark.py fotos/clean --variants v1_original v2_antiloop    # prompt A/B
python3 src/vlm_benchmark.py fotos/clean --max-tokens 4096 8192                # compare output caps
python3 src/vlm_benchmark.py fotos/clean --num-ctx 8192 16384                  # compare context windows
python3 src/vlm_benchmark.py fotos/clean --think true false                    # compare with/without reasoning
python3 src/vlm_benchmark.py fotos/clean --images 1.jpeg 14.jpeg 16.jpeg       # only those images
python3 src/vlm_benchmark.py fotos/clean --scope all --runs 1
```

> Beware the **combinatorial explosion**: the total number of calls is
> `images × runs × models × prompts × max_tokens × num_ctx × think`. The menu
> shows you the total before running.

| Flag            | Default (config.json)  | What it does |
|-----------------|------------------------|----------|
| `folder` (pos.) | `folder`               | Folder with images (jpg/jpeg/png/bmp/webp). |
| `--images`      | (all)                  | Specific names inside the folder (e.g. `1.jpeg 14.jpeg`). Without this, it uses all of them. |
| `--models`      | `benchmark_models`     | One or several models to compare (space-separated). |
| `--variants`    | `benchmark_variants`   | One or several prompt variants (`v1_original v2_antiloop …`). |
| `--runs`        | `benchmark_runs`       | Repetitions per image. |
| `--scope`       | `benchmark_scope`      | `industrial` or `all`. |
| `--max-tokens`  | `benchmark_max_tokens` | One or **several** output caps (`num_predict`) to compare (e.g. `4096 8192`). |
| `--num-ctx`     | `benchmark_num_ctx`    | One or **several** context windows to compare (e.g. `8192 16384`). |
| `--think`       | `benchmark_think`      | One or **several** `true`/`false` values to compare (e.g. `--think true false`). Only changes anything on models with the `thinking` capability; see the reasoning note. |
| `--out`         | (timestamped)          | Output file. Default `results/benchmark_<timestamp>.json` (each run gets its own file). |
| `--url`         | `url`                  | Ollama host. |

While running it shows a **progress bar** (current combination/image, %
and ETA). When it finishes it prints the **time per image** (avg/min/max) and a table with
**one row per combination** and columns `ctx`/`maxtok`/`thk` plus
P50/P95/mean/min/max/total + JSON% + `length` cutoffs + average objects; it closes
with a **verdict** that gives you the **full config** of the best combination
(model + variant + max_tokens + num_ctx + think), ready to paste into `config.json`.
Each run is saved to its **own timestamped file** — `results/benchmark_<YYYYMMDD_HHMMSS>.json`
(or pass `--out path.json`) — so separate runs **never overwrite each other** and can be
compared. The file holds both the metrics **and the per-image detections**: under each
combination, `detections` maps every image to what the model actually returned per run
(the full `{"objects": [...]}`), so you can inspect *what* it detected, not just the
aggregate numbers. The image **file names are never sent to the model** (only the image
bytes + the fixed prompt), so it can't "cheat" by reading the name.

> **Context note (speed):** the benchmark starts with `num_ctx=8192` /
> `max_tokens=4096` — **half** of what the scan uses (16384 / 8192).
> Less context = faster prefill (the image goes in at lower resolution). With
> the `v2_antiloop` prompt this does **not** truncate: tested on images 1, 14 and 16
> (including the one that previously ran out of context) → valid JSON and ~13–17 s.
> If a hard image comes back empty (`finish_reason: length`), raise `num_ctx` /
> `max_tokens` from the submenu (option 6).

> **Comparing prompts (A/B):** the A/B of prompt variants **is no longer a separate
> script** — it's inside the benchmark. Pass several with `--variants` (or pick them
> in the submenu, option 5) and compare speed / JSON% / objects keeping everything
> else constant. E.g.: `python3 src/vlm_benchmark.py fotos/clean --variants v1_original v2_antiloop`.

---

## Detection modes (`scope`)

| scope        | What it detects | Taxonomy (`type`) |
|--------------|-------------|--------------------|
| `industrial` | **Any** industrial instrument/equipment | general family: `pressure\|temperature\|flow\|level\|electrical\|analysis\|control\|vibration\|valve\|ppe\|other` (+ free `description` with the detail) |
| `all`        | Any visible object | free category (`person`, `vehicle`, …) |

The prompts for each mode are in `src/vlm_common.py` → `SCOPES`. The
`industrial` mode gives a list of typical instruments per family (pressure gauge,
thermocouple, flow meter, radar sensor, etc.) **as a reference, not a closed
list**: the model must be able to recognize any industrial instrument.

The **bounding boxes** are returned normalized 0–1. qwen3-vl delivers them in
pixels of the original file, so the code normalizes them on its own (reading the
real image size from the JPEG/PNG header).

## Prompt variants (interchangeable / A-B test)

The prompts live in `src/vlm_common.py` → `PROMPT_VARIANTS`, **one per variant**, and
are written in **English** (qwen3-vl reasons in English; less translation
overhead). The JSON *keys* and the `type` values are also in English because
they are the VLM→VLA contract.

| scope | variant | what it's like |
|-------|----------|---------|
| `industrial` | `v1_original` | The original short prompt. Gives the list of families and says "don't deliberate on the category". |
| `industrial` | `v2_antiloop` | Longer and more explicit: clarifies that **equipment counts** (not only instruments), expands `electrical` (transformer, bushing, disconnector…) and starts with an anti-deliberation RULE so it **doesn't get stuck choosing a category** (that was what emptied the `content`). |
| `all` | `default` | Single prompt for generic objects. |

**How to swap them** (3 ways, no need to touch code except the last one):
1. `config.json` → key `"variant"` (scan) or `"benchmark_variants"` (benchmark).
2. Flag `--variant v2_antiloop` in `src/vlm_scan.py`, or `--variants v1_original v2_antiloop` in `src/vlm_benchmark.py` (overrides the config for that run).
3. The **default active** variant is in `DEFAULT_VARIANT` (`src/vlm_common.py`); change it there if you want to move the global default.

To **add** a new variant: add an entry to `PROMPT_VARIANTS["industrial"]`
and compare with `python3 src/vlm_benchmark.py … --variants <old> <new>`.

> **Measurement note:** in tests on images 1, 14 and 16, `v2_antiloop`
> turned out **faster** than `v1_original` (e.g. on image 16: ~15 s vs ~97 s),
> because cutting the category deliberation saves a lot of reasoning tokens.
> The default active variant is `v1_original` (explicit request); change it to
> `v2_antiloop` if you want the faster one. Reproduce with
> `python3 src/vlm_benchmark.py fotos/clean --variants v1_original v2_antiloop`.

## config.json

It's created on its own the first time. The menu edits it, but you can tweak it by hand:

```json
{
  "model": "qwen3-vl:4b",
  "image": "fotos/clean/2.jpeg",
  "folder": "fotos/clean",
  "scope": "industrial",
  "variant": "v1_original",
  "max_tokens": 8192,
  "num_ctx": 16384,
  "think": true,
  "url": "http://localhost:11434",

  "benchmark_models": ["qwen3-vl:8b", "qwen3-vl:4b", "qwen2.5vl:7b"],
  "benchmark_runs": 3,
  "benchmark_images": [],
  "benchmark_scope": "industrial",
  "benchmark_variants": ["v2_antiloop"],
  "benchmark_max_tokens": [4096],
  "benchmark_num_ctx": [8192],
  "benchmark_think": [true]
}
```

The benchmark's sweep dimensions (`benchmark_models`, `benchmark_variants`,
`benchmark_max_tokens`, `benchmark_num_ctx`, `benchmark_think`) are **lists**:
put a single element to not sweep that dimension, or several to compare them
(e.g. `"benchmark_max_tokens": [4096, 8192]`). The benchmark also accepts scalar
values from old configs (it wraps them in a list on its own).

The `benchmark_*` keys are the benchmark's **own** config (independent
of the scan). `benchmark_images: []` means **all** the images in the
folder (if you add photos, they're included automatically); put a list of names
(`["1.jpeg", "16.jpeg"]`) to run only those. The scan (`max_tokens`,
`num_ctx`, `scope`, `variant`) stays intact.

## Important notes

- **VRAM / model choice:** on an 8 GB GPU (e.g. RTX 5060), `qwen3-vl:8b`
  (~10 GB loaded) **doesn't fit** and Ollama splits it ~53% CPU / 47% GPU →
  ~85–110 s per image. `qwen3-vl:4b` (~3.3 GB) loads **100% on GPU** →
  ~15–25 s. That's why the default is `4b`. Confirm with `ollama ps`.
- **Reasoning (thinking) — the flag is NOT an on/off, the MODEL decides it:**
  tested against Ollama 0.30.6, the reasoning control is **not** a simple
  `think: true/false` per request, but depends on the model's *capability*
  (you can see it with `ollama show <model>` or `/api/show`):

  | model | `thinking` capability | `--think` (ON) | `--no-think` (OFF) |
  |--------|:---------------------:|----------------|--------------------|
  | `qwen3-vl:4b` / `:8b` | yes | reasons | **reasons anyway** (the `qwen3-vl-thinking` renderer **ignores** `think:false` in 0.30.6) |
  | `qwen2.5vl:7b` | no | **HTTP 400 error** if the flag is sent | never reasons (real OFF) |

  In other words: **the real reasoning switch is choosing the model**.
  `qwen3-vl` always thinks; if you want it to **not** think, use a model without the
  `thinking` capability (e.g. `qwen2.5vl:7b`). That's why the code queries
  `/api/show` and **only sends `think` if the model supports it** (it isn't sent to
  `qwen2.5vl`: otherwise it returns `400 "does not support thinking"` — that used to
  make the benchmark of that model fail). The UI tells you the truth per model (whether it
  reasons, whether it was requested and whether it actually reasoned). If you need to turn off
  the reasoning of `qwen3-vl`, another version of Ollama is needed (in 0.30.6 it's not
  possible). And watch out: when it reasons, the risk is running out of tokens — if the
  reasoning eats up the budget, `content` comes back empty
  (`finish_reason: length`); it's mitigated with the anti-loop prompt (below) and giving it
  room with `max_tokens` and `num_ctx`. We use the **native** endpoint (`/api/chat`,
  not `/v1/...`) because it separates the reasoning (`thinking`) from the JSON (`content`) and
  allows **printing it live**.
- **Anti-loop prompt (why it ran out of context):** the typical case was an
  image where the model *recognized* the object (e.g. a bushing / transformer)
  but got into a loop **debating which family to put it in** (`electrical`? `other`?
  `control`?) until it exhausted the tokens → empty `content`. The `industrial`
  prompt now cuts that off at the root:
  - the `system` asks it to **reason in few steps, without repeating itself, and move to the JSON
    as soon as it recognizes the object** (not re-evaluate the category);
  - the `user` starts with an **explicit RULE**: identify at a glance, don't
    debate, and if it hesitates between two families pick one and put the detail in
    `description` (or use `other`);
  - it clarifies that **equipment also counts** (not only measuring instruments)
    and the families were expanded (`electrical` now includes transformer, bushing,
    disconnector, breaker, busbar, cell; examples of `valve` and `ppe` were added).

  Expected result: shorter reasoning (faster) and no truncation. Still, it's
  advisable to leave a token margin for the hard images.
- **`max_tokens` vs `num_ctx` (the difference that matters):**
  - **`num_ctx`** = the **full context window**: everything that goes in +
    everything that comes out. That is, `input (system + user + image tokens) +
    output (reasoning + answer)`. It's the number you see in `ollama ps` under
    *context*. Also, **more `num_ctx` lets Ollama send the image at higher
    resolution** (more image tokens → more detail).
  - **`max_tokens`** (= `num_predict`) = the **cap on what the model *generates***
    (reasoning + answer). When this cap is reached, it cuts off and returns
    `finish_reason: length` (what was happening to you: it cut off mid-reasoning).
  - **How they combine:** the real output budget is
    `min(max_tokens, num_ctx − input_tokens)`. That is, **both have to
    suffice**: if `max_tokens` is small, it cuts off even if `num_ctx` is plentiful; if
    `num_ctx` is small, the input (image included) eats into the output's room and
    it also cuts off. The input here is around ~1000–2600 tokens, so with
    `num_ctx 16384` and `max_tokens 8192` both stay comfortable.
  - **Current defaults:** `max_tokens 8192`, `num_ctx 16384` (previously 4096 / 8192,
    which fell short on hard images). Raise them further with `--max-tokens`
    / `--num-ctx` if an image keeps truncating; lower them if you want more speed
    and your images are simple.
- **Latency vs the F1.8 target:** the target of **P95 < 1.5 s** is not achievable
  with a VLM of this class on this hardware (best case ~15 s). To get closer
  you'd need another model/quantization, lower image resolution, or more VRAM.

## Structure

| Path                  | What it is |
|-----------------------|--------|
| `menu.py`             | Interactive menu (main entry point). |
| `src/vlm_scan.py`   | VLM scan of 1 image (CLI). |
| `src/vlm_benchmark.py`    | VLM latency/JSON benchmark + prompt A/B (CLI). |
| `src/yolo_common.py`  | YOLO core: ultralytics loader, detection, config. |
| `src/yolo_scan.py`  | YOLO scan of 1 image (CLI). |
| `src/yolo_benchmark.py`   | YOLO latency benchmark (models × imgsz × conf). |
| `src/vlm_common.py`   | Shared core: prompts (`PROMPT_VARIANTS`/`SCOPES`), Ollama client, config. |
| `config.json`         | Persistent configuration (created on its own, at the root). |
| `fotos/clean/`, `fotos/ciudad/` | Test images. |
| `results/`            | Benchmark output (JSON). |
