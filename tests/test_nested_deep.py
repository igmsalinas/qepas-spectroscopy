from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qepas_spectroscopy.models.deep_learning.nested import (
    _load_completed_fold,
    _save_completed_fold,
)


class NestedDeepArtifactTests(unittest.TestCase):
    def test_completed_fold_round_trip_is_signature_and_shape_safe(self) -> None:
        predictions = {
            "InceptionNestedNoAug": np.arange(12, dtype=np.float32).reshape(
                2, 3, 2
            )
        }
        metadata = {"signature": {"protocol_version": 1}, "fold": 0}

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_completed_fold(directory, 0, predictions, metadata)

            loaded = _load_completed_fold(
                directory,
                0,
                metadata["signature"],
                tuple(predictions),
                seeds=2,
                test_size=3,
                output_count=2,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            loaded_predictions, loaded_metadata = loaded
            np.testing.assert_array_equal(
                loaded_predictions["InceptionNestedNoAug"],
                predictions["InceptionNestedNoAug"],
            )
            self.assertEqual(loaded_metadata, metadata)

            self.assertIsNone(
                _load_completed_fold(
                    directory,
                    0,
                    {"protocol_version": 2},
                    tuple(predictions),
                    seeds=2,
                    test_size=3,
                    output_count=2,
                )
            )
            self.assertIsNone(
                _load_completed_fold(
                    directory,
                    0,
                    metadata["signature"],
                    tuple(predictions),
                    seeds=3,
                    test_size=3,
                    output_count=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
