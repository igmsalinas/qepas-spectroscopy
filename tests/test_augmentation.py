from __future__ import annotations

import unittest

import numpy as np

from qepas_spectroscopy.data import (
    SpectralAugmentationConfig,
    augment_training_partition,
)


class SpectralAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        phase = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        modulus = np.linspace(0.1, 0.3, 16, dtype=np.float32)
        sample = np.column_stack((modulus, np.sin(phase), np.cos(phase)))
        self.signals = np.stack([sample + [0.01 * i, 0.0, 0.0] for i in range(4)])
        self.scalars = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.targets = np.arange(8, dtype=np.float32).reshape(4, 2)

    def test_augmentation_is_deterministic_and_preserves_contracts(self) -> None:
        original = self.signals.copy()

        first = augment_training_partition(
            self.signals,
            self.scalars,
            self.targets,
            seed=42,
        )
        second = augment_training_partition(
            self.signals,
            self.scalars,
            self.targets,
            seed=42,
        )

        self.assertEqual(first.signals.shape, (8, 16, 3))
        self.assertEqual(first.scalars.shape, (8, 3))
        self.assertEqual(first.targets.shape, (8, 2))
        np.testing.assert_array_equal(self.signals, original)
        np.testing.assert_allclose(first.signals, second.signals)
        np.testing.assert_allclose(first.signals[:4], self.signals)
        np.testing.assert_allclose(first.scalars[4:], self.scalars)
        np.testing.assert_allclose(first.targets[4:], self.targets)
        self.assertFalse(np.array_equal(first.signals[4:], self.signals))
        phase_norm = np.hypot(
            first.signals[4:, :, 1], first.signals[4:, :, 2]
        )
        np.testing.assert_allclose(phase_norm, 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(first.signals).all())

    def test_augmentation_rejects_incompatible_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly modulus"):
            augment_training_partition(
                np.zeros((4, 16, 5)),
                self.scalars,
                self.targets,
                seed=42,
            )

    def test_augmentation_config_rejects_negative_magnitudes(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            SpectralAugmentationConfig(phase_offset_std=-0.1)


if __name__ == "__main__":
    unittest.main()
