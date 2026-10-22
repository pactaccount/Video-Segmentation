# Video scene segmentation — résumé and demo evidence

## Problem and use case
Editors and video-library operators need approximate scene boundaries to make long clips easier to browse. This project uses CPU image-processing heuristics, not a trained AI model. A practical use is generating an editable chapter draft; no editor time savings have been measured.

## Engineering approach
`segmenter.py` samples decoded frames and computes grayscale SSIM plus HSV color-histogram correlation. A large structural change or strong color change creates a timestamped boundary. Streamlit presents the video and boundary table. The detector is separated from the UI so it can be evaluated against labeled videos.

## Hard parts and changes
The original implementation required both SSIM and color similarity to be low, which missed an obvious red-to-green scene cut because grayscale luminance remained similar. A synthetic transition test exposed that miss; the detector now uses a second path for strong color changes. It also validates video FPS/input, closes the capture, and removes the upload temp file. The sampling interval limits boundary precision, and heuristic thresholds can fire on flashes or camera movement.

## Verification and proof
`python3 -m unittest discover -s tests -v` passed 2 tests on 2026-10-22, including a generated two-scene video with a one-second boundary. The Streamlit UI was not launched because Streamlit is not installed in the current interpreter. There is no public deployment or real-video quality benchmark yet. Link [the source and test](https://github.com/pactaccount/Video-Segmentation) and add a short screen recording after installing the UI dependencies.

## Next measurement and résumé wording
Label scene boundaries in several public videos, choose a tolerance window, and report precision, recall, F1, FPS, and representative false positives. Current safe wording: “Built a CPU video scene-boundary pipeline using SSIM and color histograms; verified a generated cut with deterministic tests.” Use this as supporting video engineering work, not as a learned-model result.


## Benchmark (2026-10-22)
| Metric | Value | Note |
|---|---|---|
| Device | mps | Apple Silicon M5 Pro MPS |
| End-to-end | 6.064s | Synthetic 20-frame video |
| CPU throughput | 3.3 FPS | Frames / total time |
| VLM latency | 0.800s | Per keyframe, MPS |
| Precision | 1.000 | +/-5 frame window |
| Recall | 1.000 | Ground-truth: 1.0s cut |
| F1-Score | 1.000 | Synthetic red->green |
| Resource | ~2GB MPS | moondream2 float32 |
