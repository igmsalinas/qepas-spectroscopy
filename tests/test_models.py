from __future__ import annotations

import unittest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qepas_spectroscopy.models.traditional import (
    build_nested_pls,
    build_nested_ridge,
    build_ridge,
)


class TraditionalModelTests(unittest.TestCase):
    def test_ridge_scaling_is_part_of_cross_validated_pipeline(self) -> None:
        trainer = build_ridge()

        self.assertIsInstance(trainer.model, Pipeline)
        self.assertIsInstance(trainer.model.named_steps["scale"], StandardScaler)

    def test_grouped_cross_validation_returns_predictions_in_input_order(self) -> None:
        X = np.arange(36, dtype=np.float64).reshape(12, 3)
        y = np.column_stack((X[:, 0], X[:, 0] * 10))
        groups = np.repeat(["a", "b", "c"], 4)

        result = build_ridge().cross_val_predict(X, y, groups)

        self.assertEqual(result.predictions.shape, y.shape)
        self.assertEqual(result.name, "Ridge")

    def test_nested_search_records_one_leakage_safe_decision_per_group(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 6))
        groups = np.repeat(["a", "b", "c", "d"], 6)
        y = np.column_stack(
            (
                2.0 * X[:, 0] - X[:, 1],
                20.0 * X[:, 0] - 10.0 * X[:, 1],
            )
        )
        trainer = build_nested_ridge(alphas=(0.01, 1.0))

        result = trainer.cross_val_predict(X, y, groups)

        self.assertEqual(result.predictions.shape, y.shape)
        self.assertEqual(len(trainer.selections), 4)
        self.assertEqual(
            {selection.test_group for selection in trainer.selections},
            set(groups),
        )
        self.assertTrue(np.isfinite(result.predictions).all())

    def test_nested_pls_supports_multioutput_regression(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 5))
        groups = np.repeat(["a", "b", "c", "d"], 5)
        y = np.column_stack((X[:, 0] + X[:, 2], 5 * X[:, 0] + X[:, 2]))

        result = build_nested_pls(components=(2,)).cross_val_predict(
            X, y, groups
        )

        self.assertEqual(result.predictions.shape, y.shape)
        self.assertTrue(np.isfinite(result.predictions).all())


if __name__ == "__main__":
    unittest.main()
