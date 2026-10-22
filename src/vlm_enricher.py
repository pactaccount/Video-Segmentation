"""Lightweight VLM chapter enricher using a pre-trained vision-language model.
Uses 'vikhyatk/moondream2' via transformers with automatic MPS/CPU device selection.
Includes robust memory management: model loaded once, keyframes released after inference,
structured logging for corrupt images, and graceful degradation if model download fails.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Device selection (MPS for Apple Silicon M5 Pro, else CPU)
# ------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("VLM enricher device selected: %s", DEVICE)

# ------------------------------------------------------------------
# Lazy-loaded model (load once, reuse across pipeline calls)
# ------------------------------------------------------------------
_MODEL = None
_PROCESSOR = None

MODEL_ID = "vikhyatk/moondream2"

def _load_model(force_reload=False):
    """Lazy-load the vision-language model with error handling."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None and not force_reload:
        return _MODEL, _PROCESSOR
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        # Note: moondream2 uses a specific processor; fall back gracefully
        logger.info("Loading VLM model: %s (this may take a moment)", MODEL_ID)
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        _MODEL.eval()
        logger.info("VLM model loaded successfully on %s", DEVICE)
    except Exception as exc:
        logger.error("Failed to load VLM model '%s': %s", MODEL_ID, exc)
        # Graceful fallback: return None so pipeline can continue with raw scenes
        _MODEL = None
        _PROCESSOR = None
    return _MODEL, _PROCESSOR


def caption_keyframe(image_path: str | Path, max_new_tokens: int = 60) -> Dict:
    """Caption a single keyframe using the VLM.

    Returns {
        "image_path": str,
        "caption": str,
        "status": "ok" | "degraded" | "error",
        "device": str,
    }
    """
    image_path = Path(image_path)
    result = {
        "image_path": str(image_path),
        "caption": "",
        "status": "degraded",
        "device": DEVICE,
    }
    # Robust input validation
    if not image_path.exists():
        logger.error("Keyframe not found: %s", image_path)
        result["status"] = "error"
        return result
    try:
        model, processor = _load_model()
        if model is None or processor is None:
            # Degraded mode: no caption, but pipeline continues
            logger.warning("VLM unavailable; skipping caption for %s", image_path)
            result["status"] = "degraded"
            result["caption"] = "[VLM unavailable: no caption generated]"
            return result

        # Load image safely (corrupt images handled here)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as img_exc:
            logger.error("Corrupt keyframe image %s: %s", image_path, img_exc)
            result["status"] = "error"
            return result

        # Correct moondream2 interface (encode image + answer_question with tokenizer)
        image = Image.open(image_path).convert("RGB")
        enc_image = model.encode_image(image)
        prompt = "Describe the scene in this video frame."
        caption = model.answer_question(enc_image, prompt, tokenizer=processor)
        # Sanitize
        caption = str(caption).strip()

        # Post-process caption
        caption = caption.strip()
        if not caption or caption == prompt:
            caption = "[No meaningful caption generated]"

        result["caption"] = caption
        result["status"] = "ok"
        logger.info("Captioned %s: %s...", image_path.name, caption[:80])
    except Exception as exc:
        logger.error("Unexpected error captioning %s: %s", image_path, exc)
        result["status"] = "error"
    finally:
        # Memory hygiene: delete large intermediate tensors if any referenced
        # (Python GC handles most; explicit cleanup for long batches)
        import gc
        gc.collect()
        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    return result


def enrich_chapters(
    keyframe_paths: List[str | Path],
    scene_boundaries: List[Dict],
) -> List[Dict]:
    """Enrich detected scene boundaries with VLM captions.

    Args:
        keyframe_paths: Ordered list of extracted keyframe file paths.
        scene_boundaries: List of boundary dicts from segmenter (contains timestamp).

    Returns:
        List of chapter manifest entries with caption + timestamp.
    """
    chapters = []
    # Simple pairing: use boundary count vs keyframes; if mismatch, take min
    n = min(len(scene_boundaries), len(keyframe_paths))
    for i in range(n):
        boundary = scene_boundaries[i]
        kf_path = Path(keyframe_paths[i])
        caption_result = caption_keyframe(kf_path)
        chapter = {
            "chapter_index": i + 1,
            "timestamp_seconds": boundary.get("timestamp_seconds"),
            "ssim": boundary.get("ssim"),
            "histogram_correlation": boundary.get("histogram_correlation"),
            "keyframe_path": str(kf_path),
            "caption": caption_result["caption"],
            "caption_status": caption_result["status"],
            "device": caption_result["device"],
        }
        chapters.append(chapter)
        logger.info("Chapter %d at %.2fs -> status=%s", i + 1, chapter["timestamp_seconds"], caption_result["status"])
    if len(scene_boundaries) > len(keyframe_paths):
        logger.warning("More boundaries (%d) than keyframes (%d); some uncaptioned.", len(scene_boundaries), len(keyframe_paths))
    return chapters
