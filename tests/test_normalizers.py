from __future__ import annotations

import unittest

import numpy as np

from qepas_spectroscopy.data.normalization import (
    NormalizationBundle,
)


class NormalizerTests(unittest.TestCase):
    def test_bundle_round_trip_preserves_values_and_serialization(self) -> None:
        signals = np.arange(48, dtype=np.float64).reshape(3, 4, 4)
        scalars = np.arange(15, dtype=np.float64).reshape(3, 5)
        targets = np.arange(6, dtype=np.float64).reshape(3, 2)

        bundle = NormalizationBundle.fit(signals, scalars, targets)
        restored = NormalizationBundle.from_dict(bundle.to_dict())

        np.testing.assert_allclose(
            restored.inverse_transform_target(restored.transform_target(targets)),
            targets,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            restored.transform_signals(signals),
            bundle.transform_signals(signals),
        )

    def test_fit_rejects_non_finite_values(self) -> None:
        signals = np.zeros((2, 4, 4))
        signals[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            NormalizationBundle.fit(
                signals,
                np.zeros((2, 5)),
                np.zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
