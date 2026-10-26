# FASHN VTON v1.5 — End-to-End Knowledge Transfer (KT)

> **Audience:** Human developers and AI coding tools.  
> **Purpose:** One file that explains what this repo is, how folders map, how inference works, how to run locally, how weights work, and how to deploy.  
> **Package:** `fashn-vton` version `1.5.0` | **Python:** `>=3.10` | **License:** Apache-2.0

---

## 1. What this project is (in one paragraph)

FASHN VTON v1.5 is a **virtual try-on** AI system. You give it:

1. A **person / customer photo**
2. A **garment photo** (worn by a model, or a flat-lay product shot)
3. A **category** (`tops` | `bottoms` | `one-pieces`)

It generates a photorealistic image of that person wearing that garment. It works in **pixel space** and is **maskless** by default (segmentation-free mode), so you usually do not need to draw clothing masks yourself.

Upstream project: [fashn-AI/fashn-vton-1.5](https://github.com/fashn-AI/fashn-vton-1.5)  
Model weights: [HuggingFace `fashn-ai/fashn-vton-1.5`](https://huggingface.co/fashn-ai/fashn-vton-1.5)

---

## 2. Critical rules (read first)

| Rule | Detail |
|------|--------|
| **Do NOT commit weights to Git** | `weights/` is ~2.1 GB and is already in `.gitignore` |
| **GitHub = code only** | Clone code → download weights on the machine → run app |
| **Weights come from HuggingFace** | Via `scripts/download_weights.py` |
| **CPU is very slow** | ~550–580 seconds per image observed locally with ~10–20 diffusion steps |
| **GPU is strongly preferred for deploy** | CUDA + bf16 on Ampere+ GPUs when available |
| **Use the project venv** | System `python` may be missing; use `.venv/bin/python` or `source .venv/bin/activate` |
| **Local UI is Gradio** | `examples/web_app.py` serves at `http://127.0.0.1:7860` by default |

---

## 3. Repository folder map

```text
fashn-vton-1.5-main/
├── KT_OVERVIEW.md              ← THIS FILE (full KT for humans + AI tools)
├── README.md                   ← Upstream project readme
├── LICENSE                     ← Apache-2.0
├── pyproject.toml              ← Package metadata + dependencies
├── .gitignore                  ← Ignores weights/, .venv/, outputs/, images, etc.
│
├── src/fashn_vton/             ← MAIN PYTHON PACKAGE (installable as fashn-vton)
│   ├── __init__.py             ← Exports TryOnPipeline, PipelineOutput, TryOnModel
│   ├── pipeline.py             ← END-TO-END ORCHESTRATION (most important file)
│   ├── tryon_mmdit.py          ← Neural network (MMDiT-style TryOnModel)
│   ├── dwpose/                 ← Pose detection (ONNX DWPose + YOLOX)
│   ├── preprocessing/          ← Resize/pad, masks, clothing-agnostic image
│   └── utils/                  ← Checkpoint load, sampling schedule, tensors, logger
│
├── examples/
│   ├── web_app.py              ← Gradio local web UI
│   ├── basic_inference.py      ← CLI one-shot inference
│   ├── colab_inference_server.py ← FastAPI GPU server for Colab (async jobs + API key)
│   └── data/                   ← Sample person + garment images (local; often gitignored as images)
│
├── scripts/
│   ├── download_weights.py     ← Download model + DWPose weights from HuggingFace
│   └── debug_masks.py          ← Debug helper for masks / preprocessing
│
├── tests/
│   └── test_pose_shapes.py     ← Unit test for pose-related shapes
│
├── weights/                    ← NOT IN GIT (~2.1 GB). Created by download script
│   ├── model.safetensors       ← Main try-on weights (~1.8 GB)
│   └── dwpose/
│       ├── yolox_l.onnx        ← Person detector (~207 MB)
│       └── dw-ll_ucoco_384.onnx← Pose estimator (~128 MB)
│
├── outputs/                    ← Generated images (gitignored)
└── .venv/                      ← Local virtualenv (gitignored)
```

### What each major package piece does

| Path | Role |
|------|------|
| `pipeline.py` | Loads models, preprocesses images, runs diffusion sampling, returns PIL images. Also logs IST `startTime` / `endTime` / `duration` around sampling (local customization). |
| `tryon_mmdit.py` | The generative model (`TryOnModel`) + CFG forward helpers. |
| `dwpose/` | Detects body keypoints for person and garment so the model can align pose. |
| `preprocessing/` | Aspect-preserving resize, pad/unpad, clothing-agnostic image creation, garment prep. |
| `utils/sampling.py` | Flow-matching / diffusion timestep schedule (`get_rf_schedule`). |
| `utils/checkpoint.py` | Loads `.safetensors` into the model. |
| `examples/web_app.py` | Gradio UI wrapping `TryOnPipeline`. |
| `examples/basic_inference.py` | CLI wrapping `TryOnPipeline`. |
| `examples/colab_inference_server.py` | FastAPI server wrapping `TryOnPipeline` on Colab GPU (async `/v1/tryon` jobs). |
| `scripts/download_weights.py` | Downloads required weights; also triggers human-parser cache download. |

---

## 4. Models involved (3 components)

When `TryOnPipeline(weights_dir=...)` starts, it loads:

1. **TryOnModel** — main generative model from `weights/model.safetensors`
2. **DWPose** — pose from `weights/dwpose/*.onnx` (ONNX Runtime)
3. **FashnHumanParser** — human parsing for clothing-agnostic prep  
   - Weights are **auto-downloaded to HuggingFace cache** on first use (~244 MB)  
   - Not stored inside `./weights` by default

Device / dtype behavior:

- Device: `cuda` if available, else `cpu` (or forced via `--device`)
- Dtype: `bfloat16` on CUDA with bf16 support; otherwise `float32` (CPU uses float32)

---

## 5. End-to-end inference flow

```text
Person image + Garment image + category
        │
        ▼
[1] Set seed (torch / numpy / cuda)
        │
        ▼
[2] Pre-resize (aspect preserve) for pose quality
        │
        ▼
[3] DWPose → person pose (+ garment pose if garment_photo_type=model)
        │
        ▼
[4] FashnHumanParser + preprocessing
    → clothing-agnostic person image
    → processed garment image
    (segmentation_free controls masking aggressiveness)
        │
        ▼
[5] Resize/pad to model input shape, convert to tensors [-1, 1]
        │
        ▼
[6] Diffusion / flow sampling for `num_timesteps` steps
    → each step runs the TryOnModel (this is the slow part)
    → classifier-free guidance uses `guidance_scale`
        │
        ▼
[7] Unpad → list of PIL images in PipelineOutput.images
```

**Runtime reality:** Almost all wall-clock time is spent in step [6]. On CPU, ~10–20 steps can take ~9–10 minutes. Guidance scale and seed do not materially change runtime; diffusion steps do.

---

## 6. Advanced settings explained (UI + API)

These appear in Gradio Advanced settings and map to `TryOnPipeline.__call__` args.

### Diffusion steps (`num_timesteps`)

- How many iterative refinement steps from noise → final image.
- Higher → usually better quality, **much slower**.
- Lower → faster, may look softer / less stable.
- Practical guide from CLI help: `20=fast`, `30=balanced`, `50=quality`.
- UI slider range in `web_app.py`: 10–50 (default shown in UI is often 20).

### Guidance scale (`guidance_scale`)

- Classifier-free guidance strength.
- Conceptually: `guided = uncond + scale * (cond - uncond)`.
- `1.0` → mild / no extra push.
- `1.5` → default / balanced garment adherence.
- Higher (2–3) → stronger garment match; can oversaturate if too high.
- Little effect on runtime.

### Seed (`seed`)

- Fixes random noise initialization.
- Same seed + same inputs + same settings → reproducible (or near-reproducible) output.
- Change seed → different random variation.
- No effect on runtime.

### Segmentation-free mode (`segmentation_free`)

- `True` (default): less aggressive masking of the person; often better body preservation and garment volume.
- `False`: more traditional mask-based clothing removal before try-on.
- Affects look more than speed.

### Other related API knobs

| Arg | Default | Meaning |
|-----|---------|---------|
| `category` | required | `tops` / `bottoms` / `one-pieces` |
| `garment_photo_type` | `model` | `model` if garment worn by person; `flat-lay` for product shot |
| `num_samples` | `1` | Number of outputs (1–4) |
| `skip_cfg_last_n_steps` | `1` | Skip CFG on last N steps to reduce color saturation |

---

## 7. Local setup (developer machine)

### 7.1 Create venv and install

```bash
cd /path/to/fashn-vton-1.5-main
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
pip install -e ".[web]"            # needed for Gradio UI
```

If only CPU / no CUDA for ONNX:

```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

### 7.2 Download weights (needs internet once)

```bash
python scripts/download_weights.py --weights-dir ./weights
```

Creates:

- `./weights/model.safetensors`
- `./weights/dwpose/yolox_l.onnx`
- `./weights/dwpose/dw-ll_ucoco_384.onnx`
- Plus HuggingFace cache for FashnHumanParser on first pipeline load

Disk: keep ~3 GB free.

### 7.3 Run Gradio web app (recommended for demos)

```bash
source .venv/bin/activate
python examples/web_app.py --weights-dir ./weights --device cpu --host 127.0.0.1 --port 7860
```

Or without activating venv:

```bash
.venv/bin/python examples/web_app.py --weights-dir ./weights --device cpu --host 127.0.0.1 --port 7860
```

Open: `http://127.0.0.1:7860`

Public temporary Gradio link:

```bash
python examples/web_app.py --weights-dir ./weights --device cpu --share
```

### 7.4 Run CLI inference

```bash
python examples/basic_inference.py \
  --weights-dir ./weights \
  --person-image examples/data/model.webp \
  --garment-image examples/data/garment.webp \
  --category tops \
  --device cpu \
  --num-timesteps 20 \
  --output-dir outputs
```

### 7.5 Use as a Python library

```python
from fashn_vton import TryOnPipeline
from PIL import Image

pipeline = TryOnPipeline(weights_dir="./weights", device="cpu")  # or "cuda"

result = pipeline(
    person_image=Image.open("person.jpg").convert("RGB"),
    garment_image=Image.open("garment.jpg").convert("RGB"),
    category="tops",                 # tops | bottoms | one-pieces
    garment_photo_type="model",      # model | flat-lay
    num_samples=1,
    num_timesteps=20,
    guidance_scale=1.5,
    seed=42,
    segmentation_free=True,
)

result.images[0].save("output.png")
```

---

## 8. Logging notes (local customization)

In `src/fashn_vton/pipeline.py`, around sampling, logs include **IST** timestamps:

- Start: `Running inference with N timesteps... startTime=YYYY-MM-DD HH:MM:SS IST`
- End: `Generated K images | startTime=... | endTime=... | duration=XX.XXs`

Timezone used: `Asia/Kolkata` via `zoneinfo.ZoneInfo`.

Observed local CPU example: ~20 steps ≈ ~9–10 minutes (`duration` around 550–580s for lower step counts also reported depending on settings/hardware).

---

## 9. Git / GitHub strategy

### Push to GitHub

**YES — push code:**

- `src/`, `examples/*.py`, `scripts/`, `tests/`, `pyproject.toml`, `README.md`, `LICENSE`, `KT_OVERVIEW.md`

**NO — do not push:**

- `weights/` (~2.1 GB)
- `.venv/`
- `outputs/`
- Large binary images (many image extensions are gitignored)
- Secrets / `.env` if you add any later

`.gitignore` already excludes `weights/`, `*.safetensors`, `*.onnx`, `*.pt`, `*.pth`, `outputs/`, `.venv/`.

### Mental model

```text
GitHub  ----------->  code + scripts
HuggingFace ------->  model weights
Deploy machine ---->  git clone + pip install + download_weights + run
```

---

## 10. Deployment playbook (server / cloud)

### 10.1 Recommended shape

1. Machine with enough disk (~3+ GB for weights/cache)
2. Prefer **NVIDIA GPU + CUDA** for production latency
3. Python 3.10–3.12
4. Outbound internet for first-time weight download (can run offline after cache is warm)

### 10.2 Fresh server commands

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <repo-folder>

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[web]"

# CPU-only ONNX if needed:
# pip uninstall -y onnxruntime-gpu && pip install onnxruntime

python scripts/download_weights.py --weights-dir ./weights

# Bind on all interfaces for remote access
python examples/web_app.py \
  --weights-dir ./weights \
  --device cuda \
  --host 0.0.0.0 \
  --port 7860
```

If no GPU:

```bash
python examples/web_app.py --weights-dir ./weights --device cpu --host 0.0.0.0 --port 7860
```

Expect multi-minute latency per image on CPU.

### 10.3 Production hardening checklist (suggested)

- Put Gradio behind a reverse proxy (nginx / Caddy) with HTTPS
- Restrict auth if the UI is public (Gradio share links are temporary and not a security boundary)
- Persist `./weights` and HuggingFace cache across redeploys (volume / disk mount) so you do not re-download every time
- Set `HF_HOME` if you want a custom cache location
- Use a process manager (`systemd`, Docker, etc.)
- Monitor disk, GPU memory, and request queue (`web_app.py` uses `demo.queue(max_size=2)`)
- Do not bake 2 GB weights into every CI image unless you have a private artifact store; prefer download-on-boot or mounted volume

### 10.4 Offline inference after weights exist

Once `weights/` and HF parser cache exist on disk, generation can run without internet. Download step itself needs internet.

---

## 11. Categories reference

| Category | Use for |
|----------|---------|
| `tops` | T-shirts, shirts, blouses, jackets |
| `bottoms` | Pants, skirts, shorts |
| `one-pieces` | Dresses, jumpsuits |

Garment photo type:

| Value | Meaning |
|-------|---------|
| `model` | Garment worn by a person in the garment image |
| `flat-lay` | Product / flat garment on plain background |

---

## 12. Key entry points for AI tools / code navigation

If an AI tool must change behavior, start here:

| Goal | File |
|------|------|
| Change inference orchestration / logging | `src/fashn_vton/pipeline.py` |
| Change model architecture | `src/fashn_vton/tryon_mmdit.py` |
| Change Gradio UI / defaults / host-port | `examples/web_app.py` |
| Change CLI flags | `examples/basic_inference.py` |
| Change Colab GPU HTTP API | `examples/colab_inference_server.py` |
| Change weight download sources | `scripts/download_weights.py` |
| Change dependencies / extras | `pyproject.toml` |
| Change pose detection | `src/fashn_vton/dwpose/` |
| Change masks / agnostic person prep | `src/fashn_vton/preprocessing/` |
| Change timestep schedule | `src/fashn_vton/utils/sampling.py` |

Public Python API surface:

```python
from fashn_vton import TryOnPipeline, PipelineOutput, TryOnModel
```

---

## 13. Common failures and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `zsh: command not found: python` | No system `python` / venv not active | `source .venv/bin/activate` or use `.venv/bin/python` |
| `Missing model weights` / `FileNotFoundError` | `weights/` not downloaded | `python scripts/download_weights.py --weights-dir ./weights` |
| Very slow (~10 min/image) | Running on CPU | Expected on CPU; use GPU or lower `num_timesteps` |
| ONNX / CUDA errors on Mac or CPU box | `onnxruntime-gpu` installed without GPU | `pip uninstall onnxruntime-gpu && pip install onnxruntime` |
| Gradio not found | Web extra not installed | `pip install -e ".[web]"` |
| First run downloads extra files | Human parser cache | Normal; needs internet once |
| Port already in use | Another Gradio/process on 7860 | Change `--port` |

---

## 14. Colab GPU inference server

Use this when the Mac UI should stay local but inference runs on a Colab NVIDIA GPU.

**Dependency extra:** `pip install -e ".[server]"` → installs `fastapi`, `uvicorn`, `pydantic`.

**Auth:** set `FASHN_API_KEY` (never hardcode). Clients send `Authorization: Bearer <key>`.

**Endpoints:**

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | no | Model loaded + CUDA status |
| `POST` | `/v1/tryon` | yes | Submit job → `202 {job_id, status}` |
| `GET` | `/v1/tryon/{job_id}` | yes | Poll status; on success includes `images_b64` |

Request body fields match `TryOnPipeline.__call__`: `category`, `garment_photo_type`, `num_samples`, `num_timesteps`, `guidance_scale`, `seed`, `segmentation_free`, `skip_cfg_last_n_steps`, plus `person_image_b64` / `garment_image_b64`.

Concurrency: one inference at a time (async queue). Local Gradio remote client is a later step (`web_app.py` not wired yet).

### Colab start (exact)

```python
# 1) Runtime → Change runtime type → GPU
# 2) Mount Drive (weights already at MyDrive/fashn-vton/weights/)
from google.colab import drive
drive.mount("/content/drive")

# 3) Get the repo onto the Colab VM (clone or upload), then:
%cd /content/fashn-vton-1.5-main   # adjust to your folder name

!pip install -e ".[server]"

import os
os.environ["FASHN_API_KEY"] = "replace-with-a-long-random-secret"
os.environ["FASHN_WEIGHTS_DIR"] = "/content/drive/MyDrive/fashn-vton/weights"

!python examples/colab_inference_server.py --host 0.0.0.0 --port 8000
```

Expose port 8000 with a tunnel (`ngrok`, `cloudflared`, etc.), then test from another cell or from your Mac.

### Colab / local test curls

```bash
export FASHN_API_KEY="replace-with-a-long-random-secret"
export URL="http://127.0.0.1:8000"   # or your tunnel URL

curl -s "$URL/health"

# Submit (images as base64)
PERSON_B64=$(base64 < person.jpg | tr -d '\n')
GARMENT_B64=$(base64 < garment.jpg | tr -d '\n')

JOB=$(curl -s -X POST "$URL/v1/tryon" \
  -H "Authorization: Bearer $FASHN_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"person_image_b64\":\"$PERSON_B64\",\"garment_image_b64\":\"$GARMENT_B64\",\"category\":\"tops\",\"num_timesteps\":20}")
echo "$JOB"
JOB_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])" <<<"$JOB")

# Poll until succeeded / failed
curl -s "$URL/v1/tryon/$JOB_ID" -H "Authorization: Bearer $FASHN_API_KEY"
```

---

## 15. Quick command cheat sheet

```bash
# activate
source .venv/bin/activate

# install
pip install -e .
pip install -e ".[web]"
pip install -e ".[server]"   # Colab / FastAPI inference server

# weights
python scripts/download_weights.py --weights-dir ./weights

# UI
python examples/web_app.py --weights-dir ./weights --device cpu --host 127.0.0.1 --port 7860

# CLI
python examples/basic_inference.py \
  --weights-dir ./weights \
  --person-image examples/data/model.webp \
  --garment-image examples/data/garment.webp \
  --category tops \
  --device cpu \
  --num-timesteps 20 \
  --output-dir outputs

# Colab GPU API server
export FASHN_API_KEY="your-secret"
export FASHN_WEIGHTS_DIR="/content/drive/MyDrive/fashn-vton/weights"
python examples/colab_inference_server.py --host 0.0.0.0 --port 8000
```

---

## 16. One-line summary for AI tools

**This repo is a Python package (`fashn_vton`) that runs maskless virtual try-on via `TryOnPipeline`; code lives under `src/`, demos under `examples/`, weights must be downloaded with `scripts/download_weights.py` into `./weights` (never committed), Gradio UI is `examples/web_app.py`, Colab GPU API is `examples/colab_inference_server.py`, and almost all latency is diffusion steps on the chosen device (CPU ≈ many minutes per image; GPU preferred).**

---

*Document generated as project KT overview for developers and AI assistants. Keep this file updated when entry points, deploy steps, or weight handling change.*
