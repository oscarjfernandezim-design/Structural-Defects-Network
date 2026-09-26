import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "detectors", ROOT / "src" / "02_edge_detection.py"
)
detectors = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(detectors)


class DetectorTests(unittest.TestCase):
    def test_detectors_return_binary_masks(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[30:34, 10:54] = 255
        for name, detector in detectors.EDGE_DETECTORS.items():
            with self.subTest(operator=name):
                mask = detector(image)
                self.assertEqual(mask.shape, image.shape)
                self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_cleanup_preserves_shape(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[15, 5:25] = 255
        cleaned = detectors.clean_mask(image)
        self.assertEqual(cleaned.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
