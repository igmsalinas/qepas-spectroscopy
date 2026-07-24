from __future__ import annotations

import unittest

import numpy as np

from qepas_spectroscopy.features import (
    COMPACT_SPECTROSCOPY_FEATURES,
    EngineeredFeatureExtractor,
    RawSignalPreprocessingConfig,
    RawSignalPreprocessor,
    feature_indices,
    resample_signal,
)


def _scan_data() -> dict[str, np.ndarray]:
    phase = np.linspace(-4.0, 4.0, num=32)
    modulus = np.linspace(1.0, 2.0, num=32)
    return {
        "modulo_remuestreado": modulus,
        "fase_remuestreada": phase,
        "X_remuestreada": modulus * np.sin(phase),
        "Y_remuestreada": modulus * np.cos(phase),
        "vector_med_presion": np.array(0.9),
        "vector_cons_presion": np.array(0.9),
        "vector_med_flujo": np.array(110.0),
        "vector_cons_flujo": np.array(1000.0),
        "vector_temp_Vflujo": np.array(30.3),
    }


class SignalPreprocessingTests(unittest.TestCase):
    def test_linear_resampling_has_exact_length_and_uses_full_signal(self) -> None:
        source = np.arange(10, dtype=np.float64)

        result = resample_signal(source, target=4, method="linear")

        self.assertEqual(result.shape, (4,))
        self.assertEqual(result[0], source[0])
        self.assertEqual(result[-1], source[-1])

    def test_polyphase_resampling_preserves_constant_signal(self) -> None:
        result = resample_signal(np.ones(100), target=17)

        self.assertEqual(result.shape, (17,))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_invalid_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            resample_signal(np.arange(4), target=0)

    def test_raw_pipeline_encodes_phase_on_unit_circle(self) -> None:
        preprocessor = RawSignalPreprocessor(
            RawSignalPreprocessingConfig(length=8, resampling="linear")
        )

        prepared = preprocessor.transform(_scan_data())

        self.assertEqual(prepared.signals.shape, (8, 3))
        self.assertEqual(
            prepared.signal_names,
            ("modulus", "phase_sin", "phase_cos"),
        )
        phase_norm = np.hypot(
            prepared.signals[:, 1], prepared.signals[:, 2]
        )
        np.testing.assert_allclose(phase_norm, 1.0, atol=1e-6)

    def test_cartesian_channels_are_an_explicit_option(self) -> None:
        preprocessor = RawSignalPreprocessor(
            RawSignalPreprocessingConfig(
                length=8,
                resampling="linear",
                include_cartesian=True,
            )
        )

        prepared = preprocessor.transform(_scan_data())

        self.assertEqual(prepared.signals.shape, (8, 5))
        self.assertEqual(prepared.signal_names[-2:], ("x", "y"))

    def test_engineered_features_are_finite_for_constant_signals(self) -> None:
        data = _scan_data()
        for name in (
            "modulo_remuestreado",
            "fase_remuestreada",
            "X_remuestreada",
            "Y_remuestreada",
        ):
            data[name] = np.zeros(32)

        features = EngineeredFeatureExtractor().transform(data)

        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertIn("phase_resultant_length", features)

    def test_compact_feature_set_matches_engineered_schema(self) -> None:
        available = EngineeredFeatureExtractor().feature_names

        indices = feature_indices(
            available,
            COMPACT_SPECTROSCOPY_FEATURES,
        )

        self.assertEqual(len(indices), len(COMPACT_SPECTROSCOPY_FEATURES))
        self.assertEqual(
            tuple(available[index] for index in indices),
            COMPACT_SPECTROSCOPY_FEATURES,
        )

    def test_feature_set_rejects_schema_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            feature_indices(("known",), ("missing",))


if __name__ == "__main__":
    unittest.main()
