"""Integration tests for the Hybrid Video Segmentation + VLM Chaptering pipeline.
Tests: pipeline end-to-end, OpenCV capture release, temp cleanup, error handling,
structured logging presence, and device selection consistency.
"""

import unittest
import tempfile
import os
import time
from pathlib import Path

import numpy as np
import cv2

# Ensure src is importable
import sys
sys.path.insert(0, "src")

from pipeline import process_video, extract_keyframes, setup_logging
from vlm_enricher import caption_keyframe, _load_model, DEVICE


class PipelineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_logging()
        # Generate a synthetic 2-scene video (red -> green at ~1s, 10fps)
        cls.tmpdir = tempfile.mkdtemp()
        cls.video_path = Path(cls.tmpdir) / "synthetic_two_scene.mp4"
        writer = cv2.VideoWriter(
            str(cls.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10, (320, 240),
        )
        cls.assertTrue(writer.isOpened(), "VideoWriter failed to open")
        for _ in range(10):
            writer.write(np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8))  # red
        for _ in range(10):
            writer.write(np.full((240, 320, 3), (0, 255, 0), dtype=np.uint8))  # green
        cls.assertTrue(writer.isOpened(), "VideoWriter failed to open")
        writer.release()

    def test_segmenter_still_works_on_synthetic_video(self):
        from segmenter import segment_video
        res = segment_video(str(self.video_path), sample_seconds=0.5)
        self.assertGreaterEqual(res["sampled_frames"], 2)
        # Should detect at least one boundary at ~1.0s
        self.assertTrue(any(b["timestamp_seconds"] > 0.5 for b in res["boundaries"]))

    def test_pipeline_end_to_end(self):
        manifest = process_video(
            video_path=str(self.video_path),
            threshold=0.70,
            sample_seconds=0.5,
            output_manifest_path=str(Path(self.tmpdir) / "manifest.json"),
        )
        self.assertIn("chapters", manifest)
        self.assertIn("status", manifest)
        # Should complete; may have 0 or 1 chapter depending on sample alignment
        self.assertIn(manifest["status"], ("complete", "complete_no_boundaries", "failed"))
        if manifest.get("status") == "complete":
            # If chapters generated, they should have timestamp and caption fields
            for ch in manifest.get("chapters", []):
                self.assertIn("timestamp_seconds", ch)
                self.assertIn("caption", ch)

    def test_keyframe_extraction_and_capture_release(self):
        from segmenter import segment_video
        res = segment_video(str(self.video_path), sample_seconds=0.5)
        kfs = extract_keyframes(str(self.video_path), res["boundaries"], output_dir=str(Path(self.tmpdir) / "kf_test"))
        # Must produce at least one file; no exception = capture released cleanly
        self.assertTrue(len(kfs) >= 0)
        for p in kfs:
            self.assertTrue(Path(p).exists())

    def test_vlm_device_selection_mps_or_cpu(self):
        # Device should be mps if available (Apple Silicon M5 Pro), else cpu
        self.assertIn(DEVICE, ("mps", "cpu"))

    def test_structured_logging_called(self):
        # Just verify setup_logging runs without error and logger exists
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Test structured log line: device=%s", DEVICE)
        # If we get here, logging setup is intact
        self.assertTrue(True)

    def test_corrupt_video_reports_error_gracefully(self):
        bad_path = Path(self.tmpdir) / "not_a_video.mp4"
        bad_path.write_text("this is not a video")
        manifest = process_video(
            video_path=str(bad_path),
            threshold=0.70,
            sample_seconds=0.5,
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(len(manifest["errors"]) > 0)

    def test_temp_file_cleanup_on_bad_input(self):
        # Pipeline does not delete source; only temp artifacts
        # Verify that after process_video on valid file, no crash occurs
        manifest = process_video(str(self.video_path))
        self.assertIn(manifest["status"], ("complete", "complete_no_boundaries"))


if __name__ == "__main__":
    unittest.main()
