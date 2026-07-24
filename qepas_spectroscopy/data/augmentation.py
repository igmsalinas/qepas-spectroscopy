"""Fit-only, physics-conservative augmentation for QEPAS signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_GAUSSIAN_MAD = 0.6744897501960817


@dataclass(frozen=True, slots=True)
class SpectralAugmentationConfig:
    """Controls small perturbations that preserve the signal representation."""

    copies: int = 1
    modulus_gain_std: float = 0.01
    modulus_baseline_fraction: float = 0.002
    noise_multiplier: float = 0.25
    modulus_noise_fraction_cap: float = 0.01
    phase_offset_std: float = 0.003
    phase_noise_cap: float = 0.005

    def __post_init__(self) -> None:
        values = {
            "copies": self.copies,
            "modulus_gain_std": self.modulus_gain_std,
            "modulus_baseline_fraction": self.modulus_baseline_fraction,
            "noise_multiplier": self.noise_multiplier,
            "modulus_noise_fraction_cap": self.modulus_noise_fraction_cap,
            "phase_offset_std": self.phase_offset_std,
            "phase_noise_cap": self.phase_noise_cap,
        }
        if self.copies < 1:
            raise ValueError("augmentation copies must be positive")
        if any(value < 0 for name, value in values.items() if name != "copies"):
            raise ValueError("augmentation magnitudes cannot be negative")


@dataclass(frozen=True, slots=True)
class AugmentedTrainingBatch:
    signals: np.ndarray
    scalars: np.ndarray
    targets: np.ndarray
    diagnostics: dict[str, float]


def _robust_sigma(values: np.ndarray) -> float:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    median = np.median(flattened)
    mad = np.median(np.abs(flattened - median))
    return float(mad / _GAUSSIAN_MAD)


def augment_training_partition(
    signals: np.ndarray,
    scalars: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    config: SpectralAugmentationConfig | None = None,
) -> AugmentedTrainingBatch:
    """Return original plus augmented fit samples without mutating inputs.

    Channel order must be modulus, phase-sine, phase-cosine. Modulus receives
    small gain, baseline, and locally estimated noise perturbations. Phase is
    perturbed in angular space and re-encoded on the unit circle.
    """
    selected = config or SpectralAugmentationConfig()
    signal_values = np.asarray(signals, dtype=np.float32)
    scalar_values = np.asarray(scalars, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    if signal_values.ndim != 3 or signal_values.shape[2] != 3:
        raise ValueError(
            "augmentation requires exactly modulus, phase_sin, phase_cos"
        )
    if scalar_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("scalars and targets must be two-dimensional")
    if not (
        len(signal_values) == len(scalar_values) == len(target_values)
    ):
        raise ValueError("augmentation inputs must have equal sample counts")
    if not (
        np.isfinite(signal_values).all()
        and np.isfinite(scalar_values).all()
        and np.isfinite(target_values).all()
    ):
        raise ValueError("augmentation inputs must be finite")

    modulus = signal_values[:, :, 0]
    phase = np.arctan2(signal_values[:, :, 1], signal_values[:, :, 2])
    modulus_scale = float(np.std(modulus))
    modulus_differences = np.diff(modulus, axis=1)
    modulus_noise = (
        _robust_sigma(modulus_differences)
        / np.sqrt(2.0)
        * selected.noise_multiplier
    )
    modulus_noise = min(
        modulus_noise,
        modulus_scale * selected.modulus_noise_fraction_cap,
    )
    phase_differences = np.angle(
        np.exp(1j * np.diff(phase.astype(np.float64), axis=1))
    )
    phase_noise = min(
        _robust_sigma(phase_differences)
        / np.sqrt(2.0)
        * selected.noise_multiplier,
        selected.phase_noise_cap,
    )

    rng = np.random.default_rng(seed)
    signal_batches = [signal_values.copy()]
    for _ in range(selected.copies):
        augmented = signal_values.copy()
        gain = rng.normal(
            1.0,
            selected.modulus_gain_std,
            size=(len(signal_values), 1),
        )
        baseline = rng.normal(
            0.0,
            modulus_scale * selected.modulus_baseline_fraction,
            size=(len(signal_values), 1),
        )
        noise = rng.normal(0.0, modulus_noise, size=modulus.shape)
        augmented[:, :, 0] = np.maximum(
            0.0,
            modulus * gain + baseline + noise,
        )

        phase_offset = rng.normal(
            0.0,
            selected.phase_offset_std,
            size=(len(signal_values), 1),
        )
        angular_noise = rng.normal(0.0, phase_noise, size=phase.shape)
        augmented_phase = phase + phase_offset + angular_noise
        augmented[:, :, 1] = np.sin(augmented_phase)
        augmented[:, :, 2] = np.cos(augmented_phase)
        signal_batches.append(augmented)

    repeats = selected.copies + 1
    augmented_signals = np.concatenate(signal_batches, axis=0)
    augmented_scalars = np.concatenate([scalar_values] * repeats, axis=0)
    augmented_targets = np.concatenate([target_values] * repeats, axis=0)
    return AugmentedTrainingBatch(
        signals=augmented_signals,
        scalars=augmented_scalars,
        targets=augmented_targets,
        diagnostics={
            "modulus_noise_sigma": float(modulus_noise),
            "phase_noise_sigma": float(phase_noise),
            "modulus_scale": modulus_scale,
        },
    )
