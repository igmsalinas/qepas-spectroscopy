"""Rich KerasTuner hypermodel for QEPAS deep learning."""

from __future__ import annotations

from typing import Callable

import keras_tuner as kt
from keras import losses, metrics as keras_metrics

from .architectures import (
    build_simple_cnn,
    build_resnet1d,
    build_tcn,
    build_lstm,
    build_multiscale_cnn,
    build_transformer1d,
)
from .schedules import make_lr_schedule, make_optimizer


ARCHITECTURES = {
    "simple_cnn": build_simple_cnn,
    "resnet1d": build_resnet1d,
    "tcn": build_tcn,
    "lstm": build_lstm,
    "multiscale_cnn": build_multiscale_cnn,
    "transformer1d": build_transformer1d,
}


def _sample_loss(hp):
    loss_name = hp.Choice("loss", ["mse", "mae", "huber"])
    if loss_name == "mse":
        return losses.MeanSquaredError()
    elif loss_name == "mae":
        return losses.MeanAbsoluteError()
    elif loss_name == "huber":
        delta = hp.Float("huber_delta", 0.1, 1.0, step=0.1)
        return losses.Huber(delta=delta)
    raise ValueError(loss_name)


DEFAULT_FAST_ARCHITECTURES = ["simple_cnn", "resnet1d", "multiscale_cnn"]


def build_qepas_hypermodel(
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int = 2,
    max_epochs: int = 100,
    allowed_architectures: list[str] | None = None,
) -> Callable:
    """Return a KerasTuner hypermodel function."""
    allowed = allowed_architectures or DEFAULT_FAST_ARCHITECTURES
    invalid = set(allowed) - set(ARCHITECTURES.keys())
    if invalid:
        raise ValueError(f"Unknown architectures: {invalid}. Available: {list(ARCHITECTURES.keys())}")

    def hypermodel(hp):
        architecture = hp.Choice("architecture", allowed)
        builder = ARCHITECTURES[architecture]

        # Common hyperparameters
        normalization = hp.Choice("normalization", ["batch", "layer", "none"])
        activation = hp.Choice("activation", ["relu", "gelu", "swish"])
        dropout = hp.Float("dropout", 0.0, 0.6, step=0.1)
        l2 = hp.Float("l2", 1e-6, 1e-3, sampling="log")
        scalar_units = hp.Int("scalar_units", 4, 64, step=4)
        dense_units = hp.Int("dense_units", 16, 256, step=16)

        common_kwargs = {
            "input_shape": input_shape,
            "scalar_dim": scalar_dim,
            "scalar_units": scalar_units,
            "dense_units": dense_units,
            "dropout": dropout,
            "l2": l2,
            "activation": activation,
            "num_outputs": num_outputs,
        }

        if architecture == "simple_cnn":
            model = builder(
                conv_blocks=hp.Int("conv_blocks", 1, 4),
                initial_filters=hp.Int("initial_filters", 8, 64, step=8),
                kernel_size=hp.Int("kernel_size", 3, 11, step=2),
                strides=hp.Int("strides", 1, 4),
                dense_blocks=hp.Int("dense_blocks", 1, 3),
                normalization=normalization,
                **common_kwargs,
            )
        elif architecture == "resnet1d":
            model = builder(
                blocks=hp.Int("resnet_blocks", 1, 4),
                base_filters=hp.Int("base_filters", 8, 64, step=8),
                kernel_size=hp.Int("kernel_size", 3, 11, step=2),
                normalization=normalization,
                **common_kwargs,
            )
        elif architecture == "tcn":
            model = builder(
                nb_filters=hp.Int("nb_filters", 16, 64, step=16),
                kernel_size=hp.Int("kernel_size", 3, 9, step=2),
                nb_stacks=hp.Int("nb_stacks", 1, 3),
                **common_kwargs,
            )
        elif architecture == "lstm":
            model = builder(
                lstm_units=hp.Int("lstm_units", 16, 128, step=16),
                lstm_layers=hp.Int("lstm_layers", 1, 3),
                bidirectional=hp.Boolean("bidirectional"),
                **common_kwargs,
            )
        elif architecture == "multiscale_cnn":
            model = builder(
                filters=hp.Int("filters", 8, 64, step=8),
                normalization=normalization,
                **common_kwargs,
            )
        elif architecture == "transformer1d":
            model = builder(
                embed_dim=hp.Int("embed_dim", 32, 128, step=32),
                num_heads=hp.Int("num_heads", 2, 8, step=2),
                ff_dim=hp.Int("ff_dim", 64, 256, step=64),
                num_blocks=hp.Int("num_blocks", 1, 4),
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Optimizer and learning rate schedule
        optimizer_name = hp.Choice("optimizer", ["adam", "adamw"])
        initial_lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        schedule_name = hp.Choice("lr_schedule", ["constant", "exponential_decay", "cosine_decay"])
        warmup_epochs = hp.Int("warmup_epochs", 0, 10, step=2)
        weight_decay = hp.Float("weight_decay", 0.0, 0.1, step=0.01) if optimizer_name == "adamw" else 0.0

        lr = make_lr_schedule(
            schedule_name=schedule_name,
            initial_lr=initial_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
        )
        optimizer = make_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=lr,
            weight_decay=weight_decay,
            clipnorm=hp.Float("clipnorm", 0.5, 5.0, step=0.5) if hp.Boolean("use_clipnorm") else None,
        )

        loss = _sample_loss(hp)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[keras_metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    return hypermodel
