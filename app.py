import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Video Scene Segmentation", layout="wide")
st.title("Real-Time Video Segmentation")
st.markdown("A CPU-optimized spatiotemporal data pipeline detecting structural scene transitions.")

st.sidebar.header("Pipeline Configurations")
uploaded_file = st.sidebar.file_uploader("Upload Video File (MP4)", type=["mp4"])
ssim_threshold = st.sidebar.slider("SSIM Threshold", 0.0, 1.0, 0.70, 0.05, help="Lower value detects fewer changes.")
sample_rate_sec = st.sidebar.slider("Sampling Rate (sec)", 1, 5, 1, help="Extract 1 frame every N seconds to reduce compute overhead.")

def compute_color_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def process_video(video_path, ssim_thresh, sample_rate):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    frame_skip = int(fps * sample_rate)
    chapters = []
    prev_frame_gray = None
    prev_hist = None
    current_frame_idx = 0
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame_idx % frame_skip == 0:
            timestamp_sec = current_frame_idx / fps
            
            # Downsample for faster CPU processing
            small_frame = cv2.resize(frame, (320, 240))
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            hist = compute_color_histogram(small_frame)
            
            if prev_frame_gray is not None:
                # Calculate Structural Similarity Index
                similarity, _ = ssim(prev_frame_gray, gray_frame, full=True)
                
                # Calculate Histogram Correlation
                hist_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # Scene change detection heuristics
                if similarity < ssim_thresh and hist_corr < 0.8:
                    chapters.append({
                        "timestamp": round(timestamp_sec, 2),
                        "similarity": round(similarity, 3),
                        "hist_corr": round(hist_corr, 3)
                    })
            
            prev_frame_gray = gray_frame
            prev_hist = hist
            
        current_frame_idx += 1
        
        # Safely update progress bar
        if total_frames > 0 and current_frame_idx % (frame_skip * 5) == 0:
            progress = min(current_frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)
            
    cap.release()
    progress_bar.progress(1.0)
    return chapters

if uploaded_file:
    with st.spinner("Processing video stream..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(tmp_path)
            
        with col2:
            st.subheader("Segmentation Timeline")
            chapters = process_video(tmp_path, ssim_threshold, sample_rate_sec)
            
            if not chapters:
                st.info("No distinct scene transitions detected based on current thresholds.")
            else:
                for idx, chap in enumerate(chapters):
                    st.write(f"**Chapter {idx + 1}**: `[{chap['timestamp']}s]` | SSIM: {chap['similarity']} | Hist: {chap['hist_corr']}")
                    
        os.remove(tmp_path)
elif not uploaded_file:
    st.info("Upload an MP4 file to initialize segmentation.")
