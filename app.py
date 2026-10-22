"""Streamlit front end for the CPU scene-boundary detector."""

import os
import tempfile

import streamlit as st

from segmenter import segment_video

st.set_page_config(page_title="Video Scene Segmentation", layout="wide")
st.title("Video scene segmentation")
st.caption("CPU scene changes from SSIM and color-histogram similarity; no trained model.")

uploaded = st.sidebar.file_uploader("Upload an MP4", type=["mp4"])
threshold = st.sidebar.slider("SSIM threshold", 0.0, 1.0, 0.70, 0.05)
sample_seconds = st.sidebar.slider("Sampling interval (seconds)", 0.2, 5.0, 1.0, 0.2)

if uploaded is None:
    st.info("Upload a video to see its detected scene boundaries.")
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temporary:
        temporary.write(uploaded.getvalue())
        path = temporary.name
    try:
        st.video(uploaded)
        with st.spinner("Analyzing frames on the CPU..."):
            result = segment_video(path, threshold=threshold, sample_seconds=sample_seconds)
        st.metric("Frames sampled", result["sampled_frames"])
        st.metric("Scene boundaries", len(result["boundaries"]))
        if result["boundaries"]:
            st.dataframe(result["boundaries"], use_container_width=True)
        else:
            st.info("No boundary met both thresholds. Try a shorter sampling interval.")
    except ValueError as exc:
        st.error(str(exc))
    finally:
        os.unlink(path)
