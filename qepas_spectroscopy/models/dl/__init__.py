"""Advanced deep-learning subpackage."""

from .architectures import (
    build_simple_cnn,
    build_resnet1d,
    build_tcn,
    build_lstm,
    build_multiscale_cnn,
    build_transformer1d,
)
from .hypermodel import build_qepas_hypermodel, ARCHITECTURES
from .normalizers import NormalizationBundle, SignalNormalizer, ScalarNormalizer, TargetNormalizer
from .schedules import make_lr_schedule, make_optimizer
from .trainer import tune_deep_model, train_deep_model, save_tuner_results

__all__ = [
    "build_simple_cnn",
    "build_resnet1d",
    "build_tcn",
    "build_lstm",
    "build_multiscale_cnn",
    "build_transformer1d",
    "build_qepas_hypermodel",
    "ARCHITECTURES",
    "NormalizationBundle",
    "SignalNormalizer",
    "ScalarNormalizer",
    "TargetNormalizer",
    "make_lr_schedule",
    "make_optimizer",
    "tune_deep_model",
    "train_deep_model",
    "save_tuner_results",
]
