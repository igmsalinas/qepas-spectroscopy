from __future__ import annotations

import unittest

import keras_tuner as kt
import tensorflow as tf

from qepas_spectroscopy.models.deep_learning.hypermodel import (
    ARCHITECTURES,
    build_qepas_hypermodel,
)
from qepas_spectroscopy.models.deep_learning.small_data import (
    build_small_inception_hypermodel,
)


class ArchitectureContractTests(unittest.TestCase):
    def test_every_registered_architecture_builds_the_common_io_contract(self) -> None:
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                tf.keras.backend.clear_session()
                hypermodel = build_qepas_hypermodel(
                    input_shape=(16, 4),
                    scalar_dim=5,
                    num_outputs=2,
                    max_epochs=2,
                    allowed_architectures=[architecture],
                )

                model = hypermodel(kt.HyperParameters())

                self.assertEqual(model.input[0].shape[1:], (16, 4))
                self.assertEqual(model.input[1].shape[1:], (5,))
                self.assertEqual(model.output_shape, (None, 2))

    def test_architecture_search_dimensions_are_conditional(self) -> None:
        hyperparameters = kt.HyperParameters()
        hyperparameters.values["architecture"] = "dilated_resnet1d"
        hypermodel = build_qepas_hypermodel(
            input_shape=(32, 3),
            scalar_dim=5,
            max_epochs=2,
            allowed_architectures=[
                "inception_spectra",
                "dilated_resnet1d",
            ],
        )

        hypermodel(hyperparameters)

        names = {parameter.name for parameter in hyperparameters.space}
        self.assertIn("dilated_blocks", names)
        self.assertNotIn("inception_modules", names)


    def test_small_data_profile_caps_inception_capacity(self) -> None:
        hyperparameters = kt.HyperParameters()
        hypermodel = build_small_inception_hypermodel(
            input_shape=(32, 3),
            scalar_dim=5,
            max_epochs=4,
        )

        model = hypermodel(hyperparameters)

        self.assertEqual(model.output_shape, (None, 2))
        self.assertIn(hyperparameters.values["inception_modules"], {2, 3})
        self.assertIn(hyperparameters.values["inception_filters"], {8, 16})
        self.assertLessEqual(hyperparameters.values["dense_units"], 64)


if __name__ == "__main__":
    unittest.main()
