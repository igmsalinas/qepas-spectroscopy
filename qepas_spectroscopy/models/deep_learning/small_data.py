"""Constrained neural search space for very small grouped datasets."""

from __future__ import annotations

from collections.abc import Callable

import keras_tuner as kt
from keras import losses, metrics as keras_metrics

from .architectures import build_inception_spectra
from .schedules import make_lr_schedule, make_optimizer


def _loss(hp: kt.HyperParameters):
    name = hp.Choice("loss", ["mae", "huber", "logcosh"])
    if name == "mae":
        return losses.MeanAbsoluteError()
    if name == "huber":
        with hp.conditional_scope("loss", ["huber"]):
            delta = hp.Float("huber_delta", 0.1, 1.0, sampling="log")
        return losses.Huber(delta=delta)
    return losses.LogCosh()


def build_small_inception_hypermodel(
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int = 2,
    max_epochs: int = 40,
    steps_per_epoch: int = 1,
) -> Callable:
    """Return a deliberately low-capacity Inception regression search."""

    def hypermodel(hp: kt.HyperParameters):
        hp.Fixed("architecture", "inception_spectra")
        normalization = hp.Choice(
            "normalization", ["batch", "layer", "group"]
        )
        activation = hp.Choice("activation", ["relu", "gelu", "swish"])
        dropout = hp.Choice("dropout", [0.2, 0.3, 0.4, 0.5])
        l2 = hp.Float("l2", 1e-6, 1e-3, sampling="log")
        model = build_inception_spectra(
            input_shape=input_shape,
            scalar_dim=scalar_dim,
            modules=hp.Int("inception_modules", 2, 3),
            filters=hp.Choice("inception_filters", [8, 16]),
            bottleneck_filters=hp.Choice(
                "inception_bottleneck_filters", [8, 16]
            ),
            kernel_size=hp.Choice("inception_kernel_size", [31, 63]),
            stem_stride=hp.Choice("inception_stem_stride", [4, 8]),
            residual_interval=2,
            use_residual=hp.Boolean("inception_use_residual"),
            scalar_units=hp.Choice("scalar_units", [8, 16, 24, 32]),
            dense_units=hp.Choice("dense_units", [16, 32, 48, 64]),
            dropout=dropout,
            l2=l2,
            normalization=normalization,
            activation=activation,
            num_outputs=num_outputs,
        )

        optimizer_name = hp.Choice("optimizer", ["adam", "adamw"])
        initial_lr = hp.Float(
            "learning_rate", 1e-4, 3e-3, sampling="log"
        )
        schedule_name = hp.Choice(
            "lr_schedule", ["constant", "exponential_decay", "cosine_decay"]
        )
        warmup_epochs = hp.Int(
            "warmup_epochs", 0, min(3, max(0, max_epochs - 1))
        )
        weight_decay = 0.0
        if optimizer_name == "adamw":
            with hp.conditional_scope("optimizer", ["adamw"]):
                weight_decay = hp.Float(
                    "weight_decay", 1e-6, 1e-3, sampling="log"
                )
        use_clipnorm = hp.Boolean("use_clipnorm")
        clipnorm = None
        if use_clipnorm:
            with hp.conditional_scope("use_clipnorm", [True]):
                clipnorm = hp.Choice("clipnorm", [0.5, 1.0])
        learning_rate = make_lr_schedule(
            schedule_name=schedule_name,
            initial_lr=initial_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        model.compile(
            optimizer=make_optimizer(
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
            ),
            loss=_loss(hp),
            metrics=[keras_metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    return hypermodel
