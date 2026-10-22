"""Reusable, deterministic video scene-boundary detector."""

import cv2
import numpy as np


def _ssim(left, right):
    """SSIM on grayscale uint8 frames using the standard local-window formula."""
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    var_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left**2
    var_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right**2
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_left * mean_right
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    )
    return float(np.clip(score.mean(), -1, 1))


def _histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def segment_video(path, threshold=0.70, sample_seconds=1.0):
    if not 0 <= threshold <= 1 or sample_seconds <= 0:
        raise ValueError("Threshold must be 0–1 and sampling interval must be positive.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open this video. Use a valid MP4 file.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video frame rate is unavailable.")
    interval = max(1, round(fps * sample_seconds))
    previous_gray = previous_hist = None
    frame_index = sampled = 0
    boundaries = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hist = _histogram(small)
                sampled += 1
                if previous_gray is not None:
                    similarity = _ssim(previous_gray, gray)
                    hist_corr = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
                    # Grayscale SSIM can miss a red-to-green cut with similar
                    # luminance; the second clause catches large color changes.
                    if similarity < threshold or (hist_corr < 0.3 and similarity < 0.95):
                        boundaries.append({
                            "timestamp_seconds": round(frame_index / fps, 2),
                            "ssim": round(similarity, 3),
                            "histogram_correlation": round(hist_corr, 3),
                        })
                previous_gray, previous_hist = gray, hist
            frame_index += 1
    finally:
        capture.release()
    return {"frames": frame_index, "sampled_frames": sampled, "boundaries": boundaries}
