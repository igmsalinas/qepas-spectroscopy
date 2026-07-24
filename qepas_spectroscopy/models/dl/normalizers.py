"""Input and target normalization utilities for deep learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SignalNormalizer:
    """Per-channel standardization for signal inputs."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "SignalNormalizer":
        """Fit on (N, T, C) array."""
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "SignalNormalizer":
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


@dataclass
class ScalarNormalizer:
    """Per-feature standardization for scalar auxiliary inputs."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "ScalarNormalizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "ScalarNormalizer":
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


@dataclass
class TargetNormalizer:
    """Per-target standardization (zero mean, unit variance)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray, eps: float = 1e-8) -> "TargetNormalizer":
        mean = y.mean(axis=0)
        std = y.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "TargetNormalizer":
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


class NormalizationBundle:
    """Holds all normalizers for a deep-learning experiment."""

    def __init__(
        self,
        signal: SignalNormalizer | None = None,
        scalar: ScalarNormalizer | None = None,
        target: TargetNormalizer | None = None,
    ):
        self.signal = signal
        self.scalar = scalar
        self.target = target

    @classmethod
    def fit(
        cls,
        X_signals: np.ndarray,
        X_scalars: np.ndarray,
        y: np.ndarray,
    ) -> "NormalizationBundle":
        return cls(
            signal=SignalNormalizer.fit(X_signals),
            scalar=ScalarNormalizer.fit(X_scalars),
            target=TargetNormalizer.fit(y),
        )

    def transform_signals(self, X: np.ndarray) -> np.ndarray:
        return self.signal.transform(X) if self.signal else X

    def transform_scalars(self, X: np.ndarray) -> np.ndarray:
        return self.scalar.transform(X) if self.scalar else X

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.target.transform(y) if self.target else y

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.target.inverse_transform(y) if self.target else y

    def to_dict(self) -> Dict[str, Dict[str, list]]:
        return {
            "signal": self.signal.to_dict() if self.signal else {},
            "scalar": self.scalar.to_dict() if self.scalar else {},
            "target": self.target.to_dict() if self.target else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, list]]) -> "NormalizationBundle":
        return cls(
            signal=SignalNormalizer.from_dict(data["signal"]) if data.get("signal") else None,
            scalar=ScalarNormalizer.from_dict(data["scalar"]) if data.get("scalar") else None,
            target=TargetNormalizer.from_dict(data["target"]) if data.get("target") else None,
        )
