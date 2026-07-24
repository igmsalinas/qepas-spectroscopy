"""Hand-engineered QEPAS feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew


def _stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    standardized_shape = spread > 1e-12
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": spread,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_skew": (
            float(skew(values)) if standardized_shape else 0.0
        ),
        f"{prefix}_kurt": (
            float(kurtosis(values)) if standardized_shape else 0.0
        ),
    }


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64)))


def _circular_stats(phase: np.ndarray) -> dict[str, float]:
    phasors = np.exp(1j * phase)
    resultant = np.mean(phasors)
    return {
        "phase_circular_mean": float(np.angle(resultant)),
        "phase_resultant_length": float(np.abs(resultant)),
        "phase_sin_mean": float(np.sin(phase).mean()),
        "phase_cos_mean": float(np.cos(phase).mean()),
    }


def _phase_bins(
    modulus: np.ndarray,
    phase: np.ndarray,
    n_bins: int,
) -> dict[str, float]:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    indices = np.digitize(phase, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    features: dict[str, float] = {}
    for index in range(n_bins):
        mask = indices == index
        features[f"phase_bin_{index:02d}_mean"] = (
            float(modulus[mask].mean()) if mask.any() else 0.0
        )
    return features


def _spectral(
    arr: np.ndarray,
    prefix: str,
    n_peaks: int,
) -> dict[str, float]:
    nperseg = min(4096, len(arr))
    frequencies, psd = signal.welch(arr, fs=1.0, nperseg=nperseg)
    features = {
        f"{prefix}_psd_total": float(psd.sum()),
        f"{prefix}_psd_max": float(psd.max()),
        f"{prefix}_psd_freq_max": float(frequencies[np.argmax(psd)]),
        f"{prefix}_psd_mean": float(psd.mean()),
    }
    peak_indices = np.argsort(psd)[-n_peaks:][::-1]
    for rank, peak in enumerate(peak_indices):
        features[f"{prefix}_peak{rank}_freq"] = float(frequencies[peak])
        features[f"{prefix}_peak{rank}_pow"] = float(psd[peak])
    return features


@dataclass(frozen=True, slots=True)
class EngineeredFeatureExtractor:
    """Convert one validated scan into a stable tabular feature mapping."""

    phase_bins: int = 32
    spectral_peaks: int = 5

    def __post_init__(self) -> None:
        if self.phase_bins <= 0:
            raise ValueError("phase_bins must be positive")
        if self.spectral_peaks <= 0:
            raise ValueError("spectral_peaks must be positive")

    def transform(
        self,
        scan_data: Mapping[str, np.ndarray],
    ) -> dict[str, float]:
        modulus = np.asarray(scan_data["modulo_remuestreado"])
        phase = _wrap_phase(scan_data["fase_remuestreada"])
        x_component = np.asarray(scan_data["X_remuestreada"])
        y_component = np.asarray(scan_data["Y_remuestreada"])

        features: dict[str, float] = {}
        features.update(_stats(modulus, "mod"))
        features.update(_stats(phase, "phase"))
        features.update(_circular_stats(phase))
        features.update(_stats(x_component, "X"))
        features.update(_stats(y_component, "Y"))
        features.update(_phase_bins(modulus, phase, self.phase_bins))
        features.update(_spectral(modulus, "mod", self.spectral_peaks))

        scalar_mapping = {
            "P_med": "vector_med_presion",
            "P_cons": "vector_cons_presion",
            "F_med": "vector_med_flujo",
            "F_cons": "vector_cons_flujo",
            "Temp": "vector_temp_Vflujo",
        }
        for feature_name, array_name in scalar_mapping.items():
            value = np.asarray(scan_data[array_name])
            if value.size != 1:
                raise ValueError(f"{array_name} must contain one value")
            features[feature_name] = float(value.reshape(-1)[0])
        return features

    @property
    def feature_names(self) -> tuple[str, ...]:
        sample = {
            "modulo_remuestreado": np.zeros(32),
            "fase_remuestreada": np.zeros(32),
            "X_remuestreada": np.zeros(32),
            "Y_remuestreada": np.zeros(32),
            "vector_med_presion": np.array(1.0),
            "vector_cons_presion": np.array(1.0),
            "vector_med_flujo": np.array(1.0),
            "vector_cons_flujo": np.array(1.0),
            "vector_temp_Vflujo": np.array(1.0),
        }
        return tuple(self.transform(sample))
