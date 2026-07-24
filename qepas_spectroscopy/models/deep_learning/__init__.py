"""Deep-learning model family."""

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
from .hypermodel import ARCHITECTURES, build_qepas_hypermodel
from .schedules import make_lr_schedule, make_optimizer
from .nested import FullyNestedDeepOutcome, train_fully_nested_inception
from .small_data import build_small_inception_hypermodel
from .trainer import save_tuner_results, train_deep_model, tune_deep_model

__all__ = [
    "build_simple_cnn",
    "build_inception_spectra",
    "build_dilated_resnet1d",
    "build_resnet1d",
    "build_tcn",
    "build_lstm",
    "build_multiscale_cnn",
    "build_transformer1d",
    "build_qepas_hypermodel",
    "ARCHITECTURES",
    "make_lr_schedule",
    "make_optimizer",
    "tune_deep_model",
    "train_deep_model",
    "save_tuner_results",
    "build_small_inception_hypermodel",
    "FullyNestedDeepOutcome",
    "train_fully_nested_inception",
]
