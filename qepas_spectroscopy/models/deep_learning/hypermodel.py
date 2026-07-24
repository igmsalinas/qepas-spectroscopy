"""Literature-guided KerasTuner search space for QEPAS regression."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import keras_tuner as kt
from keras import losses, metrics as keras_metrics

from .architectures import (
    build_dilated_resnet1d,
    build_inception_spectra,
    build_lstm,
    build_multiscale_cnn,
    build_resnet1d,
    build_simple_cnn,
    build_tcn,
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
    "inception_spectra": build_inception_spectra,
    "dilated_resnet1d": build_dilated_resnet1d,
}
DEFAULT_FAST_ARCHITECTURES = [
    "simple_cnn",
    "inception_spectra",
    "dilated_resnet1d",
]


def _sample_loss(hp: kt.HyperParameters):
    loss_name = hp.Choice("loss", ["mse", "mae", "huber", "logcosh"])
    if loss_name == "mse":
        return losses.MeanSquaredError()
    if loss_name == "mae":
        return losses.MeanAbsoluteError()
    if loss_name == "huber":
        with hp.conditional_scope("loss", ["huber"]):
            delta = hp.Float(
                "huber_delta",
                0.05,
                2.0,
                sampling="log",
            )
        return losses.Huber(delta=delta)
    if loss_name == "logcosh":
        return losses.LogCosh()
    raise ValueError(loss_name)


def _build_architecture(
    hp: kt.HyperParameters,
    architecture: str,
    builder: Callable[..., Any],
    normalization: str,
    common_kwargs: dict[str, Any],
):
    """Build one model while registering only its active search dimensions."""
    if architecture == "simple_cnn":
        with hp.conditional_scope("architecture", ["simple_cnn"]):
            return builder(
                conv_blocks=hp.Int("simple_conv_blocks", 1, 5),
                initial_filters=hp.Int(
                    "simple_initial_filters", 8, 96, step=8
                ),
                kernel_size=hp.Int("simple_kernel_size", 3, 21, step=2),
                strides=hp.Choice("simple_stride", [1, 2, 4, 8]),
                dense_blocks=hp.Int("simple_dense_blocks", 1, 3),
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "resnet1d":
        with hp.conditional_scope("architecture", ["resnet1d"]):
            return builder(
                blocks=hp.Int("resnet_blocks", 1, 5),
                base_filters=hp.Int("resnet_base_filters", 8, 64, step=8),
                kernel_size=hp.Int("resnet_kernel_size", 3, 15, step=2),
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "tcn":
        with hp.conditional_scope("architecture", ["tcn"]):
            return builder(
                nb_filters=hp.Int("tcn_filters", 16, 96, step=16),
                kernel_size=hp.Int("tcn_kernel_size", 3, 15, step=2),
                nb_stacks=hp.Int("tcn_stacks", 1, 3),
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "lstm":
        with hp.conditional_scope("architecture", ["lstm"]):
            return builder(
                lstm_units=hp.Int("lstm_units", 16, 128, step=16),
                lstm_layers=hp.Int("lstm_layers", 1, 3),
                bidirectional=hp.Boolean("lstm_bidirectional"),
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "multiscale_cnn":
        with hp.conditional_scope("architecture", ["multiscale_cnn"]):
            family = hp.Choice(
                "multiscale_kernel_family",
                ["small", "medium", "large"],
            )
            kernel_sizes = {
                "small": [3, 7, 15],
                "medium": [5, 15, 31],
                "large": [7, 31, 63],
            }[family]
            return builder(
                filters=hp.Int("multiscale_filters", 8, 64, step=8),
                kernel_sizes=kernel_sizes,
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "transformer1d":
        with hp.conditional_scope("architecture", ["transformer1d"]):
            return builder(
                embed_dim=hp.Int("transformer_embed_dim", 32, 128, step=32),
                num_heads=hp.Choice("transformer_heads", [2, 4, 8]),
                ff_dim=hp.Int("transformer_ff_dim", 64, 256, step=64),
                num_blocks=hp.Int("transformer_blocks", 1, 4),
                **common_kwargs,
            )
    if architecture == "inception_spectra":
        with hp.conditional_scope("architecture", ["inception_spectra"]):
            return builder(
                modules=hp.Int("inception_modules", 2, 6),
                filters=hp.Int("inception_filters", 8, 32, step=8),
                bottleneck_filters=hp.Int(
                    "inception_bottleneck_filters", 8, 32, step=8
                ),
                kernel_size=hp.Choice(
                    "inception_kernel_size", [15, 31, 63, 127]
                ),
                stem_stride=hp.Choice("inception_stem_stride", [2, 4, 8]),
                residual_interval=hp.Choice(
                    "inception_residual_interval", [2, 3]
                ),
                use_residual=hp.Boolean("inception_use_residual"),
                normalization=normalization,
                **common_kwargs,
            )
    if architecture == "dilated_resnet1d":
        with hp.conditional_scope("architecture", ["dilated_resnet1d"]):
            return builder(
                blocks=hp.Int("dilated_blocks", 2, 8),
                filters=hp.Int("dilated_filters", 8, 64, step=8),
                kernel_size=hp.Int("dilated_kernel_size", 3, 11, step=2),
                dilation_depth=hp.Int("dilated_depth", 2, 5),
                stem_stride=hp.Choice("dilated_stem_stride", [2, 4, 8]),
                normalization=normalization,
                **common_kwargs,
            )
    raise ValueError(f"Unknown architecture: {architecture}")


def build_qepas_hypermodel(
    input_shape: tuple[int, int],
    scalar_dim: int,
    num_outputs: int = 2,
    max_epochs: int = 100,
    steps_per_epoch: int = 1,
    allowed_architectures: list[str] | None = None,
) -> Callable:
    """Return a conditional, architecture-specific KerasTuner model factory."""
    allowed = allowed_architectures or DEFAULT_FAST_ARCHITECTURES
    invalid = set(allowed) - set(ARCHITECTURES)
    if invalid:
        raise ValueError(
            f"Unknown architectures: {invalid}. Available: {list(ARCHITECTURES)}"
        )

    def hypermodel(hp: kt.HyperParameters):
        architecture = hp.Choice("architecture", allowed)
        normalization = hp.Choice(
            "normalization",
            ["batch", "layer", "group", "none"],
        )
        activation = hp.Choice("activation", ["relu", "gelu", "swish"])
        dropout = hp.Float("dropout", 0.0, 0.6, step=0.1)
        l2 = hp.Float("l2", 1e-7, 1e-2, sampling="log")
        common_kwargs = {
            "input_shape": input_shape,
            "scalar_dim": scalar_dim,
            "scalar_units": hp.Int("scalar_units", 4, 64, step=4),
            "dense_units": hp.Int("dense_units", 16, 256, step=16),
            "dropout": dropout,
            "l2": l2,
            "activation": activation,
            "num_outputs": num_outputs,
        }
        model = _build_architecture(
            hp,
            architecture,
            ARCHITECTURES[architecture],
            normalization,
            common_kwargs,
        )

        optimizer_name = hp.Choice("optimizer", ["adam", "adamw"])
        initial_lr = hp.Float(
            "learning_rate", 1e-5, 1e-2, sampling="log"
        )
        schedule_name = hp.Choice(
            "lr_schedule",
            [
                "constant",
                "exponential_decay",
                "cosine_decay",
                "cosine_decay_restarts",
            ],
        )
        max_warmup = min(10, max(0, max_epochs - 1))
        warmup_epochs = hp.Int("warmup_epochs", 0, max_warmup, step=1)
        weight_decay = 0.0
        if optimizer_name == "adamw":
            with hp.conditional_scope("optimizer", ["adamw"]):
                weight_decay = hp.Float(
                    "weight_decay", 1e-7, 1e-2, sampling="log"
                )
        use_clipnorm = hp.Boolean("use_clipnorm")
        clipnorm = None
        if use_clipnorm:
            with hp.conditional_scope("use_clipnorm", [True]):
                clipnorm = hp.Choice("clipnorm", [0.5, 1.0, 2.0, 5.0])
        learning_rate = make_lr_schedule(
            schedule_name=schedule_name,
            initial_lr=initial_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        optimizer = make_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )
        model.compile(
            optimizer=optimizer,
            loss=_sample_loss(hp),
            metrics=[keras_metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    return hypermodel
