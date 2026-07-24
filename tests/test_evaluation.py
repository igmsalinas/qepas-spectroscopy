from __future__ import annotations

import tempfile
import unittest

import numpy as np
from pathlib import Path

from qepas_spectroscopy.evaluation import (
    compute_metrics,
    save_prediction_artifacts,
    save_seed_group_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_rejects_mismatched_prediction_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_metrics(np.zeros((2, 2)), np.zeros((2, 1)))

    def test_rejects_non_finite_predictions(self) -> None:
        predictions = np.zeros((2, 2))
        predictions[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            compute_metrics(np.zeros((2, 2)), predictions)

    def test_persists_oof_predictions_residuals_and_group_metrics(self) -> None:
        actual = np.array([[0.0, 0.0], [1.0, 10.0]])
        predicted = actual + np.array([[0.1, 1.0], [-0.1, -1.0]])
        result = compute_metrics(actual, predicted)
        result.name = "Model"

        with tempfile.TemporaryDirectory() as tmp:
            predictions, groups = save_prediction_artifacts(
                [result],
                actual,
                np.array(["a/0", "b/0"]),
                np.array(["a", "b"]),
                tmp,
            )

            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(groups), 4)
            self.assertTrue((Path(tmp) / "predictions.csv").is_file())
            self.assertTrue((Path(tmp) / "group_metrics.csv").is_file())

    def test_persists_seed_group_metrics(self) -> None:
        actual = np.array([[0.0, 0.0], [1.0, 10.0]])
        seed_predictions = {
            "Ensemble": np.stack(
                [actual, actual + np.array([[0.1, 1.0], [-0.1, -1.0]])]
            )
        }

        with tempfile.TemporaryDirectory() as tmp:
            metrics = save_seed_group_metrics(
                seed_predictions,
                actual,
                np.array(["a", "b"]),
                tmp,
            )

            self.assertEqual(len(metrics), 8)
            self.assertEqual(set(metrics["seed"]), {1, 2})
            self.assertTrue(
                (Path(tmp) / "deep_seed_group_metrics.csv").is_file()
            )

    def test_computes_each_target_metric(self) -> None:
        actual = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])

        result = compute_metrics(actual, actual.copy())

        self.assertEqual(set(result.per_target), {"13CO2", "12CO2"})
        self.assertEqual(result.overall_rmse, 0.0)
        self.assertEqual(result.per_target["13CO2"].r2, 1.0)


if __name__ == "__main__":
    unittest.main()
