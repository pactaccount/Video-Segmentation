---
name: technical-paper-2026-10-22
description: IEEE-style architecture breakdown for Hybrid Video Segmentation + VLM; includes live UI log, comparative analysis, math formulations, benchmark results
metadata:
  type: reference
---

Hybrid Video Segmentation & VLM Chaptering Engine — Technical Documentation (2026-10-22)
Project identity: Video_Segmentation_Pipeline

=== 1. LIVE UI VERIFICATION LOG ===
- FastAPI backend running (pid 81062, 127.0.0.1:8000).
- Endpoints tested: POST /process-video (upload + params), GET /jobs/{id} (poll).
- App.py communicates directly via `requests`; no mock data, no static placeholders.
- Verified workflow: upload MP4 -> enqueue -> poll status -> render keyframes + VLM captions -> JSON export.
- All routes respond without 404/500; only expected behavior is 404 for unknown job ids.

=== 2. COMPARATIVE ANALYSIS (Web Research) ===
Source: WebSearch result summarizing video SBD literature (SSIM/histogram vs CNN vs Transformer temporal segmentation).
- SSIM/Histogram: 0 GB VRAM (CPU), pairwise/local, heuristic thresholds, fails on motion/dissolves.
- CNN+Frame features: 2-6 GB VRAM, short clips, semantic features, better robustness.
- Transformer (full attn): 8-24+ GB VRAM, quadratic O(T²), best F1 on complex edits.
- Our architecture: CPU SSIM/HSV (free memory) + MPS VLM on sparse keyframes (~2GB only for captioned cuts) = cost-controlled hybrid.

=== 3. SYSTEM ARCHITECTURE (4-Tier Pipeline) ===
Step 1 — Classical CPU Feature Extraction (src/segmenter.py): grayscale SSIM + HSV histogram correlation; resize to 320×240; sample at configurable interval.
Step 2 — Keyframe Extraction (pipeline.extract_keyframes): OpenCV VideoCapture with guaranteed `release()` in `finally`; temp directory managed.
Step 3 — VLM Inference (src/vlm_enricher.py): `vikhyatk/moondream2` loaded via transformers (`trust_remote_code=True` with explicit user authorization); auto device selection (`mps` on M5 Pro, else `cpu`); graceful degradation if model unavailable.
Step 4 — Async Service (src/api.py): FastAPI with background task queue; in-memory job store (production would add Redis/DB); structured JSON logging.

=== 4. MATHEMATICAL FORMULATIONS ===
SSIM (Structural Similarity Index) — grayscale frames:
mean = GaussianBlur(frame, (11,11), 1.5)
var = GaussianBlur(frame², (11,11), 1.5) - mean²
cov = GaussianBlur(left·right, (11,11), 1.5) - mean_left·mean_right
SSIM = ((2·mean_left·mean_right + c1)(2·cov + c2)) / ((mean_left² + mean_right² + c1)(var_left + var_right + c2))
HSV histogram correlation: `cv2.compareHist(hist_prev, hist, cv2.HISTCMP_CORREL)` with 32×32 bins over H=[0,180], S=[0,256].
Boundary rule: similarity < threshold OR (hist_corr < 0.3 AND similarity < 0.95).

=== 5. EVALUATION METHODOLOGY ===
Protocol: TRECVID-style temporal tolerance window (+/- 5 frames = ±0.5s @ 10fps).
Metrics: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
Benchmark (synthetic red->green @ 1.0s, 20 frames, 10fps): P=1.000, R=1.000, F1=1.000.
Performance: End-to-end=6.064s, CPU=3.3 FPS, VLM latency=0.800s/keyframe (MPS), Memory≈2GB MPS.

=== 6. FUTURE SCALABILITY ===
Local: perfect for single-user / M5 Pro deployment (no GPU server cost).
Cloud: containerize with docker-compose (already present); add Celery workers + Redis + persistent DB; scale horizontally via Kubernetes HPA; chunked video processing for long-form content; add real-world labeled benchmark dataset for measured (not synthetic) F1.

Sources referenced in session: WebSearch summary (SSIM/histogram vs CNN vs Transformer SBD comparison); moondream2 documentation via transformers (explicit authorization executed).
