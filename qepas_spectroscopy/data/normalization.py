"""Input and target normalization utilities for deep learning."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _validated_numeric_array(
    values: np.ndarray,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class SignalNormalizer:
    """Per-channel standardization for signal inputs."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "SignalNormalizer":
        values = _validated_numeric_array(X, name="signals", ndim=3)
        mean = values.mean(axis=(0, 1))
        std = values.std(axis=(0, 1))
        return cls(mean=mean, std=np.where(std < eps, 1.0, std))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "SignalNormalizer":
        return cls(mean=np.asarray(data["mean"]), std=np.asarray(data["std"]))


@dataclass(frozen=True, slots=True)
class ScalarNormalizer:
    """Per-feature standardization for scalar auxiliary inputs."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "ScalarNormalizer":
        values = _validated_numeric_array(X, name="scalars", ndim=2)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return cls(mean=mean, std=np.where(std < eps, 1.0, std))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "ScalarNormalizer":
        return cls(mean=np.asarray(data["mean"]), std=np.asarray(data["std"]))


@dataclass(frozen=True, slots=True)
class TargetNormalizer:
    """Per-target standardization (zero mean, unit variance)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray, eps: float = 1e-8) -> "TargetNormalizer":
        values = _validated_numeric_array(y, name="targets", ndim=2)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return cls(mean=mean, std=np.where(std < eps, 1.0, std))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "TargetNormalizer":
        return cls(mean=np.asarray(data["mean"]), std=np.asarray(data["std"]))


@dataclass(frozen=True, slots=True)
class NormalizationBundle:
    """The preprocessing state required to reproduce model predictions."""

    signal: SignalNormalizer | None = None
    scalar: ScalarNormalizer | None = None
    target: TargetNormalizer | None = None

    @classmethod
    def fit(
        cls,
        X_signals: np.ndarray,
        X_scalars: np.ndarray,
        y: np.ndarray,
    ) -> "NormalizationBundle":
        sample_counts = {len(X_signals), len(X_scalars), len(y)}
        if len(sample_counts) != 1:
            raise ValueError(
                "signals, scalars, and targets must have the same sample count"
            )
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

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "signal": self.signal.to_dict() if self.signal else {},
            "scalar": self.scalar.to_dict() if self.scalar else {},
            "target": self.target.to_dict() if self.target else {},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, dict[str, list[float]]],
    ) -> "NormalizationBundle":
        return cls(
            signal=(
                SignalNormalizer.from_dict(data["signal"])
                if data.get("signal")
                else None
            ),
            scalar=(
                ScalarNormalizer.from_dict(data["scalar"])
                if data.get("scalar")
                else None
            ),
            target=(
                TargetNormalizer.from_dict(data["target"])
                if data.get("target")
                else None
            ),
        )
