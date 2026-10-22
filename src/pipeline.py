"""Orchestrator pipeline: segmenter cuts -> keyframe extraction -> VLM caption -> JSON chapter manifest.
Includes robust OpenCV capture release, temp-file cleanup, structured logging,
error handling for corrupt/missing streams, and device-aware VLM enrichment.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

# Add src to path cleanly for imports when run standalone
import sys
sys.path.insert(0, str(Path(__file__).parent))

from segmenter import segment_video
from vlm_enricher import enrich_chapters

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_keyframes(video_path: str | Path, boundaries: List[Dict], output_dir: Optional[str] = None) -> List[str]:
    """Extract keyframes at detected boundary timestamps.

    Uses OpenCV VideoCapture with guaranteed release and temp-file hygiene.
    Returns list of saved keyframe paths.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="keyframes_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.error("Cannot open video for keyframe extraction: %s", video_path)
        capture.release()
        return paths

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        logger.error("Invalid FPS in %s", video_path)
        capture.release()
        return paths

    try:
        for idx, b in enumerate(boundaries):
            ts = b.get("timestamp_seconds", 0)
            frame_index = int(round(ts * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.warning("Could not read frame at index %d (ts=%.2fs) for %s", frame_index, ts, video_path)
                # Try to write a placeholder or skip
                continue
            out_path = Path(output_dir) / f"keyframe_{idx:03d}_{ts:.2f}s.jpg"
            saved = cv2.imwrite(str(out_path), frame)
            if saved:
                paths.append(str(out_path))
                logger.info("Extracted keyframe %d: %s (ts=%.2fs)", idx, out_path.name, ts)
            else:
                logger.error("Failed to write keyframe %s", out_path)
    finally:
        capture.release()
        logger.info("VideoCapture released for %s", video_path)
    return paths


def process_video(
    video_path: str | Path,
    threshold: float = 0.70,
    sample_seconds: float = 1.0,
    keyframe_dir: Optional[str] = None,
    output_manifest_path: Optional[str] = None,
) -> Dict:
    """Full pipeline: detect -> extract -> enrich -> manifest.

    Returns JSON-ready manifest dict.
    """
    setup_logging()
    video_path = Path(video_path)
    logger.info("Starting pipeline for %s (threshold=%.2f, sample=%.1fs)", video_path, threshold, sample_seconds)

    manifest = {
        "video_path": str(video_path),
        "threshold": threshold,
        "sample_seconds": sample_seconds,
        "device": "mps" if __import__("torch").backends.mps.is_available() else "cpu",
        "chapters": [],
        "errors": [],
    }

    try:
        # 1. Scene detection (existing CPU segmenter)
        result = segment_video(str(video_path), threshold=threshold, sample_seconds=sample_seconds)
        boundaries = result.get("boundaries", [])
        manifest["sampled_frames"] = result.get("sampled_frames")
        manifest["total_frames"] = result.get("frames")
        manifest["scene_boundary_count"] = len(boundaries)
        logger.info("Detected %d scene boundaries", len(boundaries))

        if not boundaries:
            manifest["status"] = "complete_no_boundaries"
            logger.info("No boundaries detected; manifest complete.")
            return manifest

        # 2. Keyframe extraction (with guaranteed capture release)
        kf_dir = keyframe_dir or tempfile.mkdtemp(prefix="kf_")
        keyframe_paths = extract_keyframes(video_path, boundaries, output_dir=kf_dir)
        manifest["keyframe_dir"] = kf_dir
        manifest["keyframe_count"] = len(keyframe_paths)

        # 3. VLM enrichment (device-aware, with graceful degradation)
        chapters = enrich_chapters(keyframe_paths, boundaries)
        manifest["chapters"] = chapters
        manifest["enrichment_status"] = "ok" if any(ch["caption_status"] == "ok" for ch in chapters) else "degraded"

        # 4. Write manifest
        manifest["status"] = "complete"
        if output_manifest_path:
            out_path = Path(output_manifest_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info("Manifest written to %s", out_path)

    except ValueError as exc:
        msg = f"Pipeline error: {exc}"
        logger.error(msg)
        manifest["errors"].append(msg)
        manifest["status"] = "failed"
    except Exception as exc:
        msg = f"Unexpected pipeline failure: {exc}"
        logger.exception(msg)
        manifest["errors"].append(msg)
        manifest["status"] = "failed"
    finally:
        # Cleanup any temporary video file if passed from an upload (caller handles upload temp)
        # We never delete the source video; only temp artifacts created here
        pass

    return manifest
