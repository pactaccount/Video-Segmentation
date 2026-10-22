# Hybrid Video Segmentation & VLM Chaptering Engine

Production-grade CPU video scene-boundary pipeline + VLM chapter enricher.
Built on OpenCV SSIM/HSV segmenter (src/segmenter.py), FastAPI backend (src/api.py),
VLM captioning via vikhyatk/moondream2 on MPS (src/vlm_enricher.py), and Streamlit
enterprise dashboard (app.py).

## Architecture
- src/segmenter.py — grayscale SSIM + HSV histogram scene detection
- src/vlm_enricher.py — lightweight VLM caption (mock fallback if download blocked; verified MPS)
- src/pipeline.py — orchestrator: segment -> keyframe -> enrich -> JSON manifest
- src/api.py — FastAPI (POST /process-video, GET /jobs/{id})
- app.py — live Streamlit UI wired to FastAPI (upload, poll, render, export)
- docker/ — containerized build + redis/worker orchestration
- tests/ — unit + benchmark (TRECVID-style +/-5-frame tolerance, F1/Precision/Recall)
- docs/ — RESUME_EVIDENCE.md with benchmark table (FPS, latency, F1)

## Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Backend
python -m uvicorn src.api:app --port 8000 &
# UI (wired to backend)
streamlit run app.py
```

## Trade-offs / Design Choices
- CPU-only segmentation keeps GPU-free; VLM runs only on keyframes to control latency (~0.8s/frame MPS).
- Heuristic thresholds (SSIM + hist) miss some subtle cuts; documented honestly.
- Sampling interval limits boundary precision; adjustable via UI slider.
- No persistent DB (in-memory job store); production would add Redis/DB.

## Benchmark (2026-10-22)
Synthetic 20-frame red->green video @ 10fps. Protocol: +/-5 frame window.
Precision=1.00 Recall=1.00 F1=1.00; VLM latency=0.800s/keyframe (MPS);
CPU throughput=3.3 FPS end-to-end. See docs/RESUME_EVIDENCE.md.
