import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segmenter import segment_video


class SegmenterTest(unittest.TestCase):
    def test_known_scene_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for color in [(0, 0, 255)] * 10 + [(0, 255, 0)] * 10:
                writer.write(np.full((64, 64, 3), color, dtype=np.uint8))
            writer.release()
            result = segment_video(path, sample_seconds=0.5)
            self.assertEqual(len(result["boundaries"]), 1)
            self.assertAlmostEqual(result["boundaries"][0]["timestamp_seconds"], 1.0, places=1)

    def test_missing_video_reports_error(self):
        with self.assertRaisesRegex(ValueError, "Could not open"):
            segment_video("not-a-video.mp4")


if __name__ == "__main__":
    unittest.main()
