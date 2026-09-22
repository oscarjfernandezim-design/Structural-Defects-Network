import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate", ROOT / "src" / "07_evaluate.py"
)
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate)


class EvaluateTests(unittest.TestCase):
    def test_perfect_masks_have_perfect_scores(self):
        target = np.array([[0, 1], [1, 0]], dtype=bool)
        metrics = evaluate.calculate_metrics(target, target)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_empty_prediction_on_positive_target_is_not_correct(self):
        target = np.array([[0, 1], [0, 0]], dtype=bool)
        prediction = np.zeros_like(target)
        metrics = evaluate.calculate_metrics(prediction, target)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
