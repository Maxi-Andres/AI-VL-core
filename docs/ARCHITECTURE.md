# Architecture: from image-folder PoC to a live-video robot inspection system

> **Status:** design document (no code yet). This is the blueprint for evolving the current
> still-image PoC into a real-time system: robot/phone camera → backend inference (hybrid
> YOLO + Qwen3-VL) → React telemetry frontend, with a fast non-relational datastore.
>
> Everything in this repo (and therefore this doc) is in **English** by project convention.
> The conversation language is Spanish; the code and docs are not.

---

## 1. Overview & goals

### 1.1 Where we are

The PoC today is a **CLI harness over still images** in `fotos/`. It has two mirror-image
detection paths that already emit **one shared JSON contract**:

- **VLM path** — `src/vlm_common.py::query_vlm(img_b64, model, scope, max_tokens, think,
  url, timeout, num_ctx, verbose, size, variant)` streams an image to Ollama's native
  `/api/chat` endpoint and returns a dict with `parsed` (the `{"objects": [...]}` result),
  `elapsed`, `finish_reason`, reasoning, token counts, etc. Open-vocabulary, reasons over a
  prompt, **slow** (~9–25 s/image on an 8 GB GPU).
- **YOLO path** — `src/yolo_common.py::run_detection(model_name, image_path, conf, imgsz,
  device, warmup, annotate_path)` runs an in-process Ultralytics model and returns the
  **same** `{"objects": [...]}` shape. **Fast** (~30–100 ms/image).

Both produce `objects[]`, where each object is:

```jsonc
{
  "type":        "pressure|temperature|flow|level|electrical|analysis|control|vibration|valve|ppe|other",
  "description": "free text (VLM) or class label (YOLO)",
  "reading":     "120 PSI" ,        // VLM only when legible; YOLO -> null
  "confidence":  0.0,               // 0..1
  "bbox":        [x_min, y_min, x_max, y_max]   // normalized 0..1
}
```

This contract is the **anchor of the whole system** — every new component (WebSocket,
database, frontend overlay) speaks it. We do **not** invent a parallel shape.

### 1.2 Where we are going

Three new capabilities, layered on top of the existing inference core:

1. **Live video** — instead of a file path, frames arrive continuously from a camera. The
   *real* source is the **robot camera**; the *temporary* source is **your phone's camera,
   inside the same web page** (browser `getUserMedia` + a camera-permission prompt). Both
   sources land on the **same backend ingest interface**, so swapping phone → robot later is
   a one-component change.
2. **A web frontend (React + Tailwind + TypeScript)** that is a **single page** showing:
   the **live video the robot sees**, **detection overlays** drawn on top, **robot telemetry**
   widgets, and a **chat with the AI**.
3. **Persistence** — a **fast, non-relational** datastore for telemetry, (key)frames, and
   the user↔AI conversation.

### 1.3 Hard requirement: latency

The internal eval doc **F1.8** sets the target: detection precision, **P95 latency < 1.5 s
on-prem**, and a high valid-JSON rate for the VLM→VLA contract. The architecture is shaped
around keeping the **hot path (YOLO) well under that**, while the **slow path (VLM) runs off
the hot loop** so it never stalls the stream.

### 1.4 The cadence the system is built around

> **~30 frames → YOLO, ~1 frame → Qwen3-VL.**

YOLO sees (almost) every frame and gives real-time boxes. The VLM is expensive, so it runs
on a **throttled cadence** (e.g. roughly once per second / per ~30 frames) **and on demand**
(when the user asks a question about what the robot is seeing). The two never block each
other — see §5.

---

## 2. Target architecture (high level)

```
                              ┌──────────────────────────────────────────────────────────┐
                              │                     BACKEND (one host)                     │
                              │                                                            │
  ┌───────────────┐  frames  │  ┌────────────┐   every frame   ┌──────────────────────┐   │
  │ Phone camera  │ ───────► │  │            │ ──────────────► │ YOLO  (run_detection) │   │
  │ (getUserMedia │   (WS,   │  │  Ingest /  │                 │ in-proc, PyTorch/CUDA │   │
  │  in web page) │  binary) │  │  frame bus │                 └───────────┬──────────┘   │
  └───────────────┘          │  │ (asyncio   │                             │ objects[]    │
        ~later~              │  │  queues)   │  ~1/sec or on-demand         ▼              │
  ┌───────────────┐  frames  │  │            │ ──────────────► ┌──────────────────────┐   │
  │ Robot camera  │ ───────► │  │            │   latest frame  │ Qwen3-VL (query_vlm) │   │
  │ (RTSP/onboard)│          │  └─────┬──────┘                 │ Ollama /api/chat     │   │
  └───────────────┘          │        │                        └───────────┬──────────┘   │
                              │        │ telemetry + detections             │ objects[]+   │
                              │        ▼                                     │ reading/chat │
                              │  ┌────────────┐   pub/sub    ┌───────────────▼──────────┐  │
                              │  │   Redis    │ ◄──────────► │  Persistence writer       │  │
                              │  │ hot buffer │              │  - Mongo (docs/chat/dets) │  │
                              │  │ + pub/sub  │              │  - FS/MinIO (frames)      │  │
                              │  └─────┬──────┘              └───────────────────────────┘  │
                              │        │ fan-out                                            │
                              │        ▼                                                    │
                              │  ┌────────────┐                                             │
                              │  │ WebSocket  │                                             │
                              │  │  endpoint  │                                             │
                              └──┴─────┬──────┴─────────────────────────────────────────────┘
                                       │  detections (JSON) + telemetry (JSON) + chat (JSON)
                                       ▼
                              ┌─────────────────────────────────────────┐
                              │   FRONTEND  (React + Tailwind + TS)       │
                              │   one page:                               │
                              │   ┌─────────────┐  ┌───────────────────┐  │
                              │   │ <video> +   │  │ telemetry widgets │  │
                              │   │ canvas      │  │ (WS-driven)       │  │
                              │   │ overlay     │  └───────────────────┘  │
                              │   └─────────────┘  ┌───────────────────┐  │
                              │   camera permission │ chat with the AI │  │
                              │   (getUserMedia)    └───────────────────┘  │
                              └─────────────────────────────────────────┘
```

Key idea: **the same WebSocket connection carries frames *up* and telemetry/detections/chat
*down***. The video element shows the local camera (or the robot's stream); the **overlay is
drawn client-side from the detection JSON** the backend pushes back — we do *not* re-encode
and ship annotated video, which keeps the down-channel tiny and fast.

---

## 3. Backend stack: Python/FastAPI vs Kotlin+Spring (+ Python inference micro)

This is the decision you were unsure about. Both are presented as **viable**; the table
gives the trade-offs honestly, with no forced pick.

### 3.1 First, debunk "Python is slow" — because it matters here

Your worry is reasonable in general but **mostly does not apply to this workload**:

- **YOLO inference is not Python.** `run_detection` calls Ultralytics → PyTorch → CUDA
  kernels. The heavy math runs in compiled native/GPU code. Python only hands the frame in
  and reads the boxes out (~microseconds of Python around a ~30–100 ms GPU call).
- **The VLM is not even in this process.** `query_vlm` is an **HTTP client** to the Ollama
  server (written in Go). The model runs in Ollama on the GPU. The Python side is just
  streaming bytes over a socket — pure **I/O**.
- **The GIL is irrelevant for I/O-bound async.** FastAPI on `asyncio`/`uvicorn` handles
  thousands of concurrent socket reads/writes happily; the GIL only bites on **CPU-bound
  pure-Python loops**, which we don't have on the hot path. The few CPU-bound bits (JPEG
  decode/encode) are native (Pillow/OpenCV) and/or can be pushed to a worker process / thread
  pool.
- **Bottom line:** the latency budget is dominated by **GPU inference**, not by the language
  of the glue. Rewriting the glue in Kotlin would not make YOLO or Qwen faster; it would just
  move the same GPU calls behind a network hop.

So the real question is **operational/ecosystem fit**, not raw speed.

### 3.2 Option A — Python + FastAPI (single backend)

```
Camera ──WS──► [ FastAPI / uvicorn (asyncio) ]
                   ├── YOLO (run_detection, in-process, GPU)
                   ├── Ollama VLM (query_vlm, HTTP to localhost:11434)
                   ├── Redis + Mongo clients
                   └── WebSocket fan-out ──► React
```

One process, one language, **directly reuses `run_detection` and `query_vlm` as-is**.

### 3.3 Option B — Kotlin/Spring Boot gateway + Python inference microservice

```
Camera ──WS──► [ Python infer svc ]  ──gRPC/WS──►  [ Kotlin/Spring Boot ]
               (run_detection,                       (API, auth, telemetry,
                query_vlm)                             DB orchestration)
                                                          └── WS ──► React
```

The Python service still does the inference (you cannot escape Python here — Ultralytics and
the Ollama client are Python/HTTP). Kotlin owns the API surface, auth, business logic, DB.

### 3.4 Trade-off table

| Dimension | A. Python + FastAPI | B. Kotlin/Spring + Python micro |
|---|---|---|
| **Inference latency** | Best — frame → YOLO in the same process, zero extra hop | Slightly worse — frame crosses an extra service boundary (serialize + network) before/after inference |
| **Reuse of current code** | Total — `run_detection`/`query_vlm` called directly | Partial — Python kept for inference, Kotlin layer is net-new |
| **Moving parts / deploys** | One service | Two services, two runtimes (JVM + Python), more to orchestrate |
| **On-prem footprint** | Light (Python + venv, already set up via `setup.sh`) | Heavier (JVM + Python + IPC) |
| **Concurrency model** | `asyncio` (great for I/O fan-out; CPU work → workers) | JVM threads/coroutines (mature, strong for CPU-bound business logic) |
| **Type safety / large-team API** | Pydantic gives runtime validation; less compile-time rigor | Kotlin static types + Spring ecosystem shine for big, long-lived APIs |
| **Enterprise integration** | Fine, smaller ecosystem for "enterprise" middleware | Spring is the gold standard (security, messaging, transactions) |
| **Team skills** | Whole inference core is already Python | Need Kotlin/Spring expertise on the team |
| **Time-to-PoC** | Fastest | Slowest |

### 3.5 When each wins

- **Choose A (FastAPI)** when: you want the fastest path to a working live demo, minimal
  latency, minimal ops, and to reuse the existing Python core unchanged. This is the natural
  fit for a PoC and an on-prem single-GPU box.
- **Choose B (Kotlin/Spring + Python)** when: this becomes a **large, multi-team, long-lived
  product** that needs Spring's auth/security/messaging/transaction ecosystem, where keeping
  inference as a thin Python sidecar behind a Kotlin gateway is worth the extra hop and
  operational weight.

> Practical note: even if you eventually want B, **start with A and split out the Kotlin
> gateway later** — the WebSocket/JSON contracts in §9 are the same either way, so the
> Python inference core is reusable behind both.

---

## 4. Real-time transport: WebSocket vs REST (and the video channel)

### 4.1 Why not plain REST request/response

REST is **request → response, client-initiated, one shot**. For a live stream you'd be
**polling** ("any new detection? any new telemetry?") which is high-latency, wasteful, and
can't *push*. For a continuous robot feed it's the wrong tool.

### 4.2 Why WebSocket for telemetry / detections / chat

- **Server push** — the backend sends a detection the instant it's computed; no polling.
- **Bidirectional** — same socket carries chat questions up and answers down.
- **Low per-message overhead** — no HTTP headers per message after the handshake.
- **Persistent** — one connection for the whole session.

### 4.3 The video channel needs care

Three ways to get **frames into** the backend, with trade-offs:

| Transport | How | Pros | Cons |
|---|---|---|---|
| **Frames over WebSocket (binary)** | `getUserMedia` → draw to `<canvas>` → `toBlob()` JPEG → send as binary WS message at throttled FPS | Simplest; one connection for everything; full control of FPS/quality; trivial to feed straight into `run_detection` | You hand-roll FPS/quality control; not as bandwidth-efficient as a real codec at high FPS |
| **WebRTC** | Browser ↔ backend media track | Lowest latency, adaptive bitrate, built for video | Heavy: signaling server, ICE/STUN, server-side decode (aiortc); overkill for a PoC at a few FPS |
| **MJPEG / RTSP** | Phone IP-camera app or robot camera exposes a stream URL; backend reads with `cv2.VideoCapture(url)` | Zero browser code; matches how the **robot camera** will likely expose its feed | Not in-browser (needs an external app for the phone); MJPEG is bandwidth-hungry |

### 4.4 Recommended split for this project

Because you explicitly want the **same web page** with a camera permission prompt, the
PoC uses the **WebSocket binary-frames** path:

- **Up (client → backend):** `getUserMedia` → `<canvas>` → JPEG blob → **binary WS frame**,
  throttled (e.g. 10–15 FPS to start; the VLM never needs more than ~1 FPS anyway).
- **Down (backend → client):** **JSON WS messages** — detections (`objects[]`), telemetry,
  chat answers.
- **Overlay** is drawn **client-side** on a canvas over the `<video>`, from the detection
  JSON. We never ship annotated video back — keeps the down-channel tiny and the overlay
  perfectly in sync with the UI.
- **Robot camera later:** swap the ingest source to read RTSP/onboard frames with
  `cv2.VideoCapture` and feed the **same frame bus** (§5). Nothing downstream changes.

---

## 5. The 30-YOLO / 1-Qwen pipeline (the heart of the system)

The constraint: **YOLO must keep up with the stream; the VLM must never block it.** Solution
is a classic **async producer/consumer** with two consumers running at different rates.

```
                       ┌──────────────────────────────────────────────┐
 incoming frames ────► │  frame bus (asyncio.Queue, maxsize=1..N)       │
 (from WS / RTSP)      └───┬───────────────────────────────────┬───────┘
                           │ EVERY frame                        │ "latest frame" slot
                           ▼                                     │ (overwrite, drop stale)
                  ┌─────────────────┐                            ▼
                  │ YOLO consumer    │                   ┌─────────────────────┐
                  │ run_detection()  │                   │ VLM consumer         │
                  │ ~30-100 ms       │                   │ query_vlm()          │
                  │ → push objects[] │                   │ ~1/sec OR on-demand  │
                  └────────┬─────────┘                   │ ~9-25 s              │
                           │                             │ → push objects[]+    │
                           ▼                             │   reading/answer     │
                   Redis pub/sub ──► WebSocket ──► UI    └──────────┬──────────┘
                                                                    ▼
                                                          Redis pub/sub ──► WS ──► UI
```

Design rules:

- **YOLO consumer** pulls from the frame bus and runs `run_detection` per frame (or every Nth
  frame if the camera outruns the GPU). It publishes `objects[]` immediately → real-time
  overlay.
- **VLM consumer** does **not** drain the queue. It reads the **latest frame** on a timer
  (e.g. once per second) **or** when a chat question arrives, calls `query_vlm`, and publishes
  the richer result (with `reading` and/or a natural-language answer). Because it always grabs
  the *newest* frame and old frames are dropped, a slow VLM call never creates a backlog.
- **Backpressure:** the frame bus is bounded (`maxsize` small); if YOLO falls behind, oldest
  frames are dropped — we want *fresh* detections, not a growing lag.
- **Reuse:** both consumers call the **existing** functions unchanged — `run_detection`
  (`src/yolo_common.py:212`) and `query_vlm` (`src/vlm_common.py:440`). The model cache in
  `yolo_common` already keeps the YOLO weights loaded in-process; Ollama keeps the VLM warm
  (see the warm-up notes in `FIX.txt`).

### 5.1 ⚠️ GPU / VRAM contention (the real risk)

YOLO (PyTorch/CUDA) **and** Qwen3-VL (via Ollama) both want the **same 8 GB GPU**. Per the
project hardware notes, `qwen3-vl:4b-instruct` (~3.3 GB) already fills a good chunk and only
fits because it's the 4B instruct quant; the 8B model does **not** fit and spills to CPU
(~85–110 s/frame). Adding YOLO's footprint on the same card can:

- push the VLM into a CPU/GPU split (catastrophic for latency), or
- starve YOLO of memory/compute, raising its latency above the F1.8 budget.

Mitigations to document and measure:

- Keep YOLO **nano** (`yolov8n.pt` / `yolo11n.pt`) — tiny VRAM, ~30–100 ms.
- Verify the split with `ollama ps` (must show 100% GPU for the VLM).
- Run the VLM **infrequently** (≈1/sec) so the two rarely contend simultaneously; serialize
  if needed (don't fire a VLM call while a YOLO batch is mid-flight on a tight card).
- If contention is unacceptable, the long-term fix is a **second GPU** (one for the real-time
  detector, one for the VLM) — but prove the bottleneck with measurements first.

---

## 6. Database design (polyglot: Redis + MongoDB + filesystem/MinIO)

The data is heterogeneous and **non-relational**, so we use the right store for each shape
rather than forcing one DB. This matches your "fast, doesn't need to be relational" ask.

```
   live telemetry (high-rate, ephemeral)   ──►  Redis      (in-memory ring buffer + pub/sub)
   chat + detections + events (documents)  ──►  MongoDB    (JSON-native document store)
   frames / keyframes (binary blobs)       ──►  Filesystem or MinIO  (refs stored in Mongo)
```

### 6.1 Redis — hot path & fan-out

- **Live telemetry buffer:** latest robot telemetry (battery, pose, speed, sensor values)
  held in memory; optionally a short capped stream (`XADD`/Redis Streams) for the last N
  seconds. In-memory = microsecond reads, perfect for the live dashboard.
- **Pub/sub fan-out:** the inference consumers `PUBLISH` detections/telemetry; the WebSocket
  layer subscribes and pushes to every connected client. This decouples producers from the
  socket layer and lets you scale to multiple frontend clients cleanly.

### 6.2 MongoDB — durable documents

Document store because the data is **already JSON documents** (the `objects[]` contract maps
1:1 to a Mongo document — no schema gymnastics, no JOINs). Sketch of collections:

```jsonc
// detections — one doc per inference result
{
  "_id": "...",
  "ts": "2026-06-15T12:00:00Z",
  "source": "yolo" | "vlm",
  "model": "yolov8n.pt" | "qwen3-vl:4b-instruct",
  "frame_ref": "frames/2026-06-15/abc123.jpg",   // pointer into FS/MinIO (see 6.3)
  "objects": [ { "type": "...", "description": "...", "reading": null,
                 "confidence": 0.0, "bbox": [x_min, y_min, x_max, y_max] } ],
  "elapsed_s": 0.042
}

// chat — the user <-> AI conversation
{
  "_id": "...",
  "ts": "2026-06-15T12:00:05Z",
  "role": "user" | "assistant",
  "text": "what is that gauge reading?",
  "frame_ref": "frames/2026-06-15/abc123.jpg",   // the frame the question was about
  "detection_id": "..."                           // link to the detections doc used
}

// events — robot/system events for the telemetry timeline
{ "_id": "...", "ts": "...", "kind": "alarm|state_change|...", "payload": { } }
```

Mongo also has **native time-series collections**, so moderate-rate telemetry can be persisted
here too without adding another database for the PoC.

### 6.3 Filesystem / MinIO — frames (never in the DB)

Frames are **large binary blobs**. Storing them inside Redis (memory pressure) or Mongo
(document bloat, slow queries, 16 MB doc limit) is an anti-pattern. Instead:

- Write frames/keyframes to the **filesystem** (PoC) or **MinIO** (S3-compatible object
  store, drop-in for later) under a dated path, reuse the existing `results/` convention.
- Store only the **reference** (`frame_ref`) in the Mongo document.
- Don't keep every frame — store **keyframes** (e.g. the frame each VLM call ran on, or
  frames tied to events/chat), not all 15 FPS, or storage explodes.

### 6.4 Future option

If telemetry analytics becomes serious (long-range dashboards, downsampling, retention
policies), add a dedicated time-series DB — **TimescaleDB** (Postgres extension) or
**InfluxDB**. Not needed for the PoC; Redis + Mongo time-series cover it initially.

---

## 7. Frontend (React + Tailwind + TypeScript)

A **single page**. Components:

```
<App>
 ├── <CameraPanel>        // getUserMedia + permission prompt; <video> + <canvas> overlay
 │     └── <DetectionOverlay objects={...}/>   // draws bbox (0..1 * canvas size) + label
 ├── <TelemetryPanel>     // WS-driven widgets: battery, pose, speed, sensor gauges
 ├── <ChatPanel>          // ask the AI about the current frame; shows VLM answers
 └── <ConnectionStatus>   // WS state, current model, FPS, VLM cadence
```

Behaviour:

- On load, `<CameraPanel>` calls `navigator.mediaDevices.getUserMedia({ video: true })` →
  browser **asks permission** → stream attached to `<video>`.
- A capture loop draws `<video>` → offscreen `<canvas>` → `canvas.toBlob('image/jpeg')` →
  sends the blob as a **binary WS message**, throttled to a target FPS.
- Incoming detection JSON updates `<DetectionOverlay>`: each `bbox` (normalized 0..1) is
  multiplied by the canvas size and drawn as a rectangle + `type`/`description`/`confidence`
  label. Because bboxes are already normalized, the overlay is resolution-independent.
- `<TelemetryPanel>` and `<ChatPanel>` subscribe to their WS message types.
- Tailwind for layout/styling; TypeScript types **mirror the WS contracts in §9** so the
  frontend and backend share one source of truth for shapes.

---

## 8. Phone-as-camera (temporary), robot-camera later

You want the phone feed in the **same telemetry page**, the easy way:

1. Open the web app on the phone (same LAN as the backend).
2. `<CameraPanel>` requests camera permission via `getUserMedia`.
3. Frames are captured to a canvas and pushed over the WS at a throttled FPS (§7).
4. Backend ingest feeds them into the **frame bus** (§5) exactly like any other source.

**Swapping to the robot camera** is isolated to the ingest layer: replace the WS-frame source
with `cv2.VideoCapture(rtsp_url)` (or the robot's onboard SDK) feeding the **same** frame bus.
Everything downstream — YOLO/VLM consumers, Redis, Mongo, the WebSocket down-channel, the
React overlay — is unchanged, because they all speak the §9 contracts.

---

## 9. Data / message contracts (WebSocket)

One source of truth, reused by backend (Pydantic) and frontend (TS types). All down-messages
are JSON with a `type` discriminator; the up-frame is binary.

**Up (client → backend):**

- **Frame:** *binary* WS message = JPEG bytes. (Optionally a tiny JSON header before it with
  a frame id / timestamp if needed for correlation.)
- **Chat question (JSON):**
  ```jsonc
  { "type": "chat", "text": "what is that gauge reading?", "frame_id": "abc123" }
  ```

**Down (backend → client), all JSON:**

```jsonc
// real-time YOLO (or VLM) detections — wraps the existing objects[] contract
{ "type": "detections", "ts": "...", "source": "yolo", "frame_id": "abc123",
  "objects": [ { "type": "pressure", "description": "pressure gauge",
                 "reading": null, "confidence": 0.91,
                 "bbox": [0.12, 0.34, 0.20, 0.45] } ] }

// telemetry tick
{ "type": "telemetry", "ts": "...",
  "battery": 0.87, "speed": 0.4, "pose": { "x": 1.2, "y": 0.3, "yaw": 90 },
  "sensors": { "temp_c": 41.2 } }

// chat answer from the VLM
{ "type": "chat", "ts": "...", "role": "assistant",
  "text": "The gauge reads about 120 PSI.", "frame_id": "abc123",
  "objects": [ /* optional: the detection the answer is grounded on */ ] }

// status / health
{ "type": "status", "model": "qwen3-vl:4b-instruct", "fps": 14, "vlm_cadence_s": 1.0 }
```

The `objects[]` element is **byte-for-byte the existing contract** from `vlm_common` /
`yolo_common` — the backend just forwards `parsed["objects"]`.

> ⚠️ The `objects[]` keys were translated to English (`type/description/reading/confidence/
> bbox`) and the scope `todo`→`all`. The downstream **"Silk AI Proxy Gateway / F1.9"**
> consumer must already be aligned to these English keys; the WS contract inherits them.

---

## 10. Phased implementation roadmap (no code yet — this is the order to build)

| Phase | Goal | What gets added |
|---|---|---|
| **0. HTTP wrapper** | Inference reachable over the network | FastAPI app wrapping `run_detection` and `query_vlm` behind `POST /detect` / `POST /ask` (single image in, `objects[]` out). Proves the core works as a service. |
| **1. Single-frame WS loop** | Browser camera → backend → overlay | `getUserMedia` page (React+Tailwind+TS), binary-frame WS up, run YOLO per frame, detections JSON down, canvas overlay. Phone-as-camera works end to end. |
| **2. Hybrid 30/1 pipeline** | Real-time + slow path together | `asyncio` frame bus + two consumers (§5); YOLO every frame, VLM ~1/sec; backpressure + latest-frame slot. Measure VRAM contention with `ollama ps`. |
| **3. Persistence + telemetry** | Store + show robot state | Redis (hot telemetry + pub/sub fan-out), MongoDB (detections/chat/events), frames to FS/MinIO with refs. `<TelemetryPanel>` driven by WS. |
| **4. Chat + on-demand VLM** | Ask about what the robot sees | `<ChatPanel>`; chat question triggers a VLM call on the latest frame; answer streamed back; conversation stored in Mongo. |
| **5. Robot camera swap** | Replace phone with the real feed | Ingest reads RTSP/onboard frames via `cv2.VideoCapture` into the same frame bus; everything downstream unchanged. |

Each phase is independently demoable; phase 0–2 already give a working live demo on your
phone.

---

## 11. Open questions & risks

- **VRAM contention (highest risk):** YOLO + Ollama on one 8 GB GPU. Must be measured early
  (phase 2). Mitigations in §5.1; worst case needs a second GPU.
- **VLM FPS realism:** at ~9–25 s/inference, the VLM is **~0.05–0.1 FPS**, not 1 FPS, on the
  current box. "1 in 30 frames" is aspirational on this hardware — treat the VLM as
  **on-demand + best-effort cadence**, not a guaranteed per-second tick. Faster hardware or a
  smaller/quantized model is the lever.
- **F1.8 latency target (P95 < 1.5 s on-prem):** achievable for the **YOLO hot path**;
  the VLM will not meet it and is explicitly the slow/async path, not the real-time one.
- **Auth / security:** the WS endpoint and camera page need access control before anything
  leaves the lab; `getUserMedia` requires a **secure context** (HTTPS or `localhost`) —
  plan TLS for LAN/phone use.
- **Backend choice timing:** start with FastAPI (Option A) to get a demo fast; the §9
  contracts keep the door open to a Kotlin/Spring gateway (Option B) later without touching
  the Python inference core.
- **Frame storage growth:** store keyframes only (§6.3), or storage balloons at 10–15 FPS.

---

## Appendix — symbols this design reuses

| Symbol | Location | Role |
|---|---|---|
| `query_vlm(...)` | `src/vlm_common.py:440` | Streaming Ollama `/api/chat` client → `objects[]` |
| `run_detection(...)` | `src/yolo_common.py:212` | In-process Ultralytics detector → `objects[]` |
| `run_vlm_scan(...)` | `src/vlm_scan.py:36` | Single-image VLM entrypoint (reference) |
| `run_yolo_scan(...)` | `src/yolo_scan.py:36` | Single-image YOLO entrypoint (reference) |
| `objects[]` contract | `vlm_common` / `yolo_common` | `{type, description, reading, confidence, bbox(0..1)}` |
| `config.json` | repo root | Shared VLM+YOLO config (model, scope, conf, imgsz, url, …) |
| `results/` | repo root | Existing output convention; reuse for frame storage |
```
