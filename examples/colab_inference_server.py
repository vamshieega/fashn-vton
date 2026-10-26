#!/usr/bin/env python3
"""
FASHN VTON v1.5 — GPU inference HTTP server (intended for Google Colab).

Loads TryOnPipeline once and serves authenticated async try-on jobs.

Environment:
  FASHN_API_KEY   Required. Bearer token clients must send.
  FASHN_WEIGHTS_DIR  Optional. Default: ./weights
                     Colab Drive example: /content/drive/MyDrive/fashn-vton/weights

Start (Colab):
  !pip install -e ".[server]"
  import os
  os.environ["FASHN_API_KEY"] = "choose-a-long-random-secret"
  os.environ["FASHN_WEIGHTS_DIR"] = "/content/drive/MyDrive/fashn-vton/weights"
  !python examples/colab_inference_server.py --host 0.0.0.0 --port 8000

Expose publicly (pick one tunnel), then call from Mac:
  curl -s http://127.0.0.1:8000/health
  curl -s -X POST "$URL/v1/tryon" \\
    -H "Authorization: Bearer $FASHN_API_KEY" \\
    -H "Content-Type: application/json" \\
    -d '{"person_image_b64":"...","garment_image_b64":"...","category":"tops"}'
  curl -s "$URL/v1/tryon/$JOB_ID" -H "Authorization: Bearer $FASHN_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from fashn_vton import TryOnPipeline

# ---------------------------------------------------------------------------
# Limits (aligned with examples/web_app.py + examples/basic_inference.py)
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = frozenset({"tops", "bottoms", "one-pieces"})
ALLOWED_GARMENT_PHOTO_TYPES = frozenset({"model", "flat-lay"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # decoded payload size before PIL open
MAX_IMAGE_PIXELS = 25_000_000  # width * height
MAX_EDGE_PX = 4096
MIN_TIMESTEPS, MAX_TIMESTEPS = 10, 50
MIN_GUIDANCE, MAX_GUIDANCE = 1.0, 3.0
MIN_SAMPLES, MAX_SAMPLES = 1, 4
JOB_RETENTION_SECONDS = 3600
MAX_STORED_JOBS = 64

logger = logging.getLogger("colab_inference_server")
security = HTTPBearer(auto_error=False)

PIPELINE: Optional[TryOnPipeline] = None
API_KEY: str = ""
WEIGHTS_DIR: str = ""


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.queued
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    images_b64: List[str] = field(default_factory=list)
    duration_s: Optional[float] = None
    # Request payload kept only until the worker finishes (not logged).
    request: Optional["TryOnRequest"] = None
    person_image: Optional[Image.Image] = None
    garment_image: Optional[Image.Image] = None


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()
_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TryOnRequest(BaseModel):
    person_image_b64: str = Field(..., description="Base64-encoded person image (PNG/JPEG/WEBP)")
    garment_image_b64: str = Field(..., description="Base64-encoded garment image (PNG/JPEG/WEBP)")
    category: Literal["tops", "bottoms", "one-pieces"]
    garment_photo_type: Literal["model", "flat-lay"] = "model"
    num_samples: int = Field(1, ge=MIN_SAMPLES, le=MAX_SAMPLES)
    num_timesteps: int = Field(30, ge=MIN_TIMESTEPS, le=MAX_TIMESTEPS)
    guidance_scale: float = Field(1.5, ge=MIN_GUIDANCE, le=MAX_GUIDANCE)
    seed: int = 42
    segmentation_free: bool = True
    skip_cfg_last_n_steps: int = Field(1, ge=0, le=10)

    @field_validator("category")
    @classmethod
    def _category_ok(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(ALLOWED_CATEGORIES)}")
        return v

    @field_validator("garment_photo_type")
    @classmethod
    def _photo_type_ok(cls, v: str) -> str:
        if v not in ALLOWED_GARMENT_PHOTO_TYPES:
            raise ValueError(f"garment_photo_type must be one of {sorted(ALLOWED_GARMENT_PHOTO_TYPES)}")
        return v


class SubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    images_b64: Optional[List[str]] = None
    num_images: Optional[int] = None


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    cuda_available: bool
    device: Optional[str] = None
    weights_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfigured: FASHN_API_KEY is not set.",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use: Bearer <FASHN_API_KEY>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _decode_image(b64_data: str, field_name: str) -> Image.Image:
    raw = b64_data.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is not valid base64.",
        ) from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} exceeds max size of {MAX_IMAGE_BYTES} bytes.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is empty.",
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is not a recognized image.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} could not be decoded as an image.",
        ) from exc

    fmt = (img.format or "").upper()
    if fmt and fmt not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} format '{fmt}' is not allowed. Use PNG, JPEG, or WEBP.",
        )

    width, height = img.size
    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} has invalid dimensions.",
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} exceeds max pixel count ({MAX_IMAGE_PIXELS}).",
        )
    if max(width, height) > MAX_EDGE_PX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} longest edge must be <= {MAX_EDGE_PX}px.",
        )

    return img.convert("RGB")


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _purge_old_jobs() -> None:
    """Drop finished jobs older than retention; also cap total stored jobs."""
    now = time.time()
    with _jobs_lock:
        expired = [
            jid
            for jid, job in _jobs.items()
            if job.status in (JobStatus.succeeded, JobStatus.failed)
            and job.finished_at is not None
            and (now - job.finished_at) > JOB_RETENTION_SECONDS
        ]
        for jid in expired:
            del _jobs[jid]

        if len(_jobs) > MAX_STORED_JOBS:
            # Remove oldest finished jobs first.
            finished = sorted(
                (
                    (jid, job)
                    for jid, job in _jobs.items()
                    if job.status in (JobStatus.succeeded, JobStatus.failed)
                ),
                key=lambda item: item[1].finished_at or item[1].created_at,
            )
            overflow = len(_jobs) - MAX_STORED_JOBS
            for jid, _ in finished[:overflow]:
                del _jobs[jid]


def _run_inference(job: Job) -> None:
    """Blocking inference — runs in a worker thread. One job at a time via queue."""
    global PIPELINE
    assert PIPELINE is not None
    assert job.request is not None
    assert job.person_image is not None
    assert job.garment_image is not None

    req = job.request
    logger.info(
        "Job %s running inference (category=%s, timesteps=%s, seed=%s)",
        job.job_id,
        req.category,
        req.num_timesteps,
        req.seed,
    )
    start = time.perf_counter()
    result = PIPELINE(
        person_image=job.person_image,
        garment_image=job.garment_image,
        category=req.category,
        garment_photo_type=req.garment_photo_type,
        num_samples=req.num_samples,
        num_timesteps=req.num_timesteps,
        guidance_scale=req.guidance_scale,
        skip_cfg_last_n_steps=req.skip_cfg_last_n_steps,
        seed=req.seed,
        segmentation_free=req.segmentation_free,
    )
    duration = time.perf_counter() - start
    job.images_b64 = [_pil_to_b64_png(img) for img in result.images]
    job.duration_s = round(duration, 2)
    logger.info("Job %s succeeded (num_images=%s, duration_s=%.2f)", job.job_id, len(job.images_b64), duration)


async def _worker_loop() -> None:
    """Single consumer: guarantees at most one concurrent TryOnPipeline call."""
    while True:
        job_id = await _queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue

        job.status = JobStatus.running
        job.started_at = time.time()
        try:
            await asyncio.to_thread(_run_inference, job)
            job.status = JobStatus.succeeded
        except Exception as exc:
            # Do not expose internals / secrets to clients; log a short message only.
            logger.exception("Job %s failed: %s", job_id, type(exc).__name__)
            job.status = JobStatus.failed
            job.error = "Inference failed. Check server logs for details."
        finally:
            job.finished_at = time.time()
            # Drop image tensors from memory; keep only encoded outputs / error.
            job.person_image = None
            job.garment_image = None
            job.request = None
            _purge_old_jobs()
            _queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, API_KEY, WEIGHTS_DIR, _worker_task

    API_KEY = os.environ.get("FASHN_API_KEY", "").strip()
    if not API_KEY:
        raise RuntimeError(
            "FASHN_API_KEY environment variable is required. "
            "Set a long random secret before starting the server."
        )

    WEIGHTS_DIR = os.environ.get("FASHN_WEIGHTS_DIR", "./weights").strip() or "./weights"
    if not os.path.isdir(WEIGHTS_DIR):
        raise RuntimeError(
            f"Weights directory not found: {WEIGHTS_DIR}. "
            "Mount Drive and set FASHN_WEIGHTS_DIR, e.g. "
            "/content/drive/MyDrive/fashn-vton/weights"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab: Runtime → Change runtime type → GPU."
        )

    logger.info("Loading TryOnPipeline from %s (device=cuda)...", WEIGHTS_DIR)
    PIPELINE = TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
    logger.info("TryOnPipeline loaded on %s", PIPELINE.device)

    _worker_task = asyncio.create_task(_worker_loop())
    yield

    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    PIPELINE = None


app = FastAPI(
    title="FASHN VTON v1.5 Inference Server",
    version="1.5.0",
    description="Authenticated async try-on API wrapping TryOnPipeline (Colab GPU).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = PIPELINE is not None
    cuda = torch.cuda.is_available()
    device = str(PIPELINE.device) if PIPELINE is not None else None
    return HealthResponse(
        ok=loaded and cuda,
        model_loaded=loaded,
        cuda_available=cuda,
        device=device,
        weights_dir=WEIGHTS_DIR or None,
    )


@app.post(
    "/v1/tryon",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_require_api_key)],
)
async def submit_tryon(body: TryOnRequest) -> SubmitResponse:
    if PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet.",
        )

    person = _decode_image(body.person_image_b64, "person_image_b64")
    garment = _decode_image(body.garment_image_b64, "garment_image_b64")

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status=JobStatus.queued,
        request=body,
        person_image=person,
        garment_image=garment,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    await _queue.put(job_id)
    logger.info(
        "Job %s queued (category=%s, garment_photo_type=%s, num_timesteps=%s)",
        job_id,
        body.category,
        body.garment_photo_type,
        body.num_timesteps,
    )
    return SubmitResponse(job_id=job_id, status=JobStatus.queued)


@app.get(
    "/v1/tryon/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(_require_api_key)],
)
def get_tryon_job(job_id: str) -> JobResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    include_images = job.status == JobStatus.succeeded
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=_iso(job.created_at) or "",
        started_at=_iso(job.started_at),
        finished_at=_iso(job.finished_at),
        duration_s=job.duration_s,
        error=job.error if job.status == JobStatus.failed else None,
        images_b64=job.images_b64 if include_images else None,
        num_images=len(job.images_b64) if include_images else None,
    )


if __name__ == "__main__":
    # Pass the app object directly so Colab/script launch does not depend on
    # module import path quirks. Single process keeps the in-memory job queue.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="FASHN VTON v1.5 Colab GPU inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Override FASHN_WEIGHTS_DIR for this process",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    cli = parser.parse_args()
    if cli.weights_dir:
        os.environ["FASHN_WEIGHTS_DIR"] = cli.weights_dir

    logging.getLogger().setLevel(getattr(logging, cli.log_level.upper()))
    uvicorn.run(app, host=cli.host, port=cli.port, log_level=cli.log_level)
