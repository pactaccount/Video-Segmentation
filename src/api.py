"""FastAPI service for the Hybrid Video Segmentation and VLM Chaptering Engine.
Exposes:
  POST /process-video  -> enqueue pipeline job (returns job id + status endpoint)
  GET  /jobs/{id}      -> retrieve manifest / status
Includes structured logging, error handling, and input validation.
"""

import logging
import uuid
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# Import from src modules (assume running from repo root or installed package)
import sys
sys.path.insert(0, "src")

from pipeline import process_video

logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Video Segmentation + VLM Chaptering Engine")

# Simple in-memory job store (production would use Redis / DB)
_JOBS = {}


class ProcessResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Pipeline job started"


class JobStatus(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    manifest_path: Optional[str] = None
    chapters: Optional[int] = None
    errors: list = []


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("FastAPI server starting — device: MPS available=%s", __import__("torch").backends.mps.is_available())


@app.post("/process-video", response_model=ProcessResponse)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="MP4 video file"),
    threshold: float = Query(0.70, ge=0.0, le=1.0),
    sample_seconds: float = Query(1.0, gt=0.0),
):
    """Upload video and start async pipeline."""
    # Input validation
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files accepted.")

    job_id = str(uuid.uuid4())[:8]
    logger.info("Job %s: received upload %s (threshold=%.2f, sample=%.1f)", job_id, file.filename, threshold, sample_seconds)

    # Write to temp file (cleaned by pipeline / caller; here we manage)
    temp_dir = tempfile.mkdtemp(prefix="upload_")
    temp_path = Path(temp_dir) / file.filename
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            # Safety: limit upload size (e.g., 500MB rough guard via len)
            if len(content) > 500_000_000:
                raise HTTPException(status_code=413, detail="File exceeds 500MB limit.")
            f.write(content)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    manifest_path = Path(temp_dir) / "manifest.json"

    # Enqueue background task
    _JOBS[job_id] = {
        "status": "processing",
        "video_path": str(temp_path),
        "manifest_path": str(manifest_path),
        "errors": [],
    }

    def run_pipeline():
        try:
            manifest = process_video(
                video_path=temp_path,
                threshold=threshold,
                sample_seconds=sample_seconds,
                keyframe_dir=str(Path(temp_dir) / "keyframes"),
                output_manifest_path=str(manifest_path),
            )
            _JOBS[job_id].update({
                "status": manifest.get("status", "unknown"),
                "chapters": len(manifest.get("chapters", [])),
                "errors": manifest.get("errors", []),
            })
            logger.info("Job %s completed: %s chapters, status=%s", job_id, _JOBS[job_id]["chapters"], _JOBS[job_id]["status"])
        except Exception as exc:
            logger.exception("Job %s pipeline crashed: %s", job_id, exc)
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["errors"].append(str(exc))
        finally:
            # Cleanup source video temp (manifest and keyframes kept for retrieval)
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    logger.info("Cleaned temp upload %s", temp_path)
            except Exception as exc:
                logger.warning("Cleanup error for %s: %s", temp_path, exc)

    background_tasks.add_task(run_pipeline)
    return ProcessResponse(job_id=job_id, status="processing", message="Pipeline job queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    info = _JOBS[job_id]
    return JobStatus(
        job_id=job_id,
        status=info["status"],
        video_path=info.get("video_path"),
        manifest_path=info.get("manifest_path"),
        chapters=info.get("chapters"),
        errors=info.get("errors", []),
    )
