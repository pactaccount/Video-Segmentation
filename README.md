# Real-Time Video Segmentation Pipeline

A CPU-optimized spatiotemporal data pipeline that rapidly segments continuous video feeds into distinct chapters. Instead of using heavy, slow neural networks, it uses Structural Similarity Index (SSIM) and color histograms to detect scene transitions mathematically.

## Prerequisites
1. **Python 3.11** or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pactaccount/Video-Segmentation.git
   cd Video-Segmentation
   ```

2. **Set up the Python environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Upload an MP4 video file via the sidebar.
4. Adjust the SSIM Threshold or Sampling Rate sliders to optimize the segmentation sensitivity. The timeline will generate automatically.

The core detector is in `segmenter.py`. Run `python3 -m unittest discover -s tests -v` for a generated scene-change check. See [résumé evidence](docs/RESUME_EVIDENCE.md) for measured and missing proof.

## Résumé evidence

[Résumé evidence and verification](docs/RESUME_EVIDENCE.md) records the use case, engineering approach, test status, measured results, limits, and safe résumé wording.
