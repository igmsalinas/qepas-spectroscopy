"""Deterministic preprocessing for deep-learning signal inputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Literal, Mapping

import numpy as np
from scipy import signal

from ..core.schema import SCALAR_ARRAY_NAMES

ResamplingMethod = Literal["polyphase", "linear"]


def _as_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def resample_signal(
    values: np.ndarray,
    target: int,
    *,
    method: ResamplingMethod = "polyphase",
) -> np.ndarray:
    """Resample a signal to an exact length with optional anti-aliasing."""
    array = _as_finite_vector(values, name="signal")
    if target <= 0:
        raise ValueError("target must be positive")
    if method not in {"polyphase", "linear"}:
        raise ValueError(f"Unknown resampling method: {method}")
    if array.size == target:
        return array.copy()

    if method == "linear":
        source_positions = np.arange(array.size, dtype=np.float64)
        target_positions = np.linspace(0, array.size - 1, num=target)
        return np.interp(target_positions, source_positions, array)

    divisor = gcd(array.size, target)
    result = signal.resample_poly(
        array,
        up=target // divisor,
        down=array.size // divisor,
        padtype="line",
    )
    if result.size != target:
        result = np.interp(
            np.linspace(0, result.size - 1, num=target),
            np.arange(result.size, dtype=np.float64),
            result,
        )
    return result


@dataclass(frozen=True, slots=True)
class RawSignalPreprocessingConfig:
    """Configuration for deterministic, label-independent transforms."""

    length: int = 4096
    resampling: ResamplingMethod = "polyphase"
    include_cartesian: bool = False

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.resampling not in {"polyphase", "linear"}:
            raise ValueError(
                f"Unknown resampling method: {self.resampling}"
            )

    @property
    def signal_names(self) -> tuple[str, ...]:
        names = ("modulus", "phase_sin", "phase_cos")
        if self.include_cartesian:
            names += ("x", "y")
        return names

    @property
    def fingerprint(self) -> str:
        channels = "+".join(self.signal_names)
        return (
            f"qepas-raw-v1:length={self.length}:"
            f"resampling={self.resampling}:channels={channels}"
        )


@dataclass(frozen=True, slots=True)
class PreparedSignal:
    """One scan transformed into tensors consumed by a deep model."""

    signals: np.ndarray
    scalars: np.ndarray
    signal_names: tuple[str, ...]
    scalar_names: tuple[str, ...] = SCALAR_ARRAY_NAMES


@dataclass(frozen=True, slots=True)
class RawSignalPreprocessor:
    """Extract, anti-alias, and circularly encode a QEPAS scan."""

    config: RawSignalPreprocessingConfig = RawSignalPreprocessingConfig()

    def transform(
        self,
        scan_data: Mapping[str, np.ndarray],
    ) -> PreparedSignal:
        length = self.config.length
        method = self.config.resampling

        modulus = resample_signal(
            scan_data["modulo_remuestreado"],
            length,
            method=method,
        )
        phase = _as_finite_vector(
            scan_data["fase_remuestreada"],
            name="fase_remuestreada",
        )
        phase_sin = resample_signal(np.sin(phase), length, method=method)
        phase_cos = resample_signal(np.cos(phase), length, method=method)

        # Filtering circular components avoids interpolation across phase wraps.
        phase_norm = np.hypot(phase_sin, phase_cos)
        phase_norm = np.where(phase_norm < 1e-8, 1.0, phase_norm)
        channels = [
            modulus,
            phase_sin / phase_norm,
            phase_cos / phase_norm,
        ]
        if self.config.include_cartesian:
            channels.extend(
                [
                    resample_signal(
                        scan_data["X_remuestreada"],
                        length,
                        method=method,
                    ),
                    resample_signal(
                        scan_data["Y_remuestreada"],
                        length,
                        method=method,
                    ),
                ]
            )

        scalar_values: list[float] = []
        for name in SCALAR_ARRAY_NAMES:
            value = np.asarray(scan_data[name])
            if value.size != 1 or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{name} must be a numeric scalar")
            scalar = float(value.reshape(-1)[0])
            if not np.isfinite(scalar):
                raise ValueError(f"{name} must be finite")
            scalar_values.append(scalar)

        return PreparedSignal(
            signals=np.stack(channels, axis=-1).astype(np.float32),
            scalars=np.asarray(scalar_values, dtype=np.float32),
            signal_names=self.config.signal_names,
        )
