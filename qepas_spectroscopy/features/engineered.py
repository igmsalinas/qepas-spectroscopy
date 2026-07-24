"""Hand-engineered feature extraction."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


def _stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_skew": float(skew(arr)),
        f"{prefix}_kurt": float(kurtosis(arr)),
    }


def _phase_bins(mod: np.ndarray, phase: np.ndarray, n_bins: int = 32) -> dict[str, float]:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(phase, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    feats = {}
    for b in range(n_bins):
        mask = idx == b
        feats[f"phase_bin_{b:02d}_mean"] = float(mod[mask].mean()) if mask.any() else 0.0
    return feats


def _spectral(arr: np.ndarray, prefix: str, n_peaks: int = 5) -> dict[str, float]:
    nperseg = min(4096, len(arr))
    freqs, psd = signal.welch(arr, fs=1.0, nperseg=nperseg)
    total = float(psd.sum())
    feats = {
        f"{prefix}_psd_total": total,
        f"{prefix}_psd_max": float(psd.max()),
        f"{prefix}_psd_freq_max": float(freqs[np.argmax(psd)]),
        f"{prefix}_psd_mean": float(psd.mean()),
    }
    peak_idx = np.argsort(psd)[-n_peaks:][::-1]
    for i, p in enumerate(peak_idx):
        feats[f"{prefix}_peak{i}_freq"] = float(freqs[p])
        feats[f"{prefix}_peak{i}_pow"] = float(psd[p])
    return feats


def build_features(scan_data: dict[str, np.ndarray]) -> dict[str, float]:
    mod = scan_data["modulo_remuestreado"]
    phase = scan_data["fase_remuestreada"]
    X = scan_data["X_remuestreada"]
    Y = scan_data["Y_remuestreada"]

    feats = {}
    feats.update(_stats(mod, "mod"))
    feats.update(_stats(phase, "phase"))
    feats.update(_stats(X, "X"))
    feats.update(_stats(Y, "Y"))
    feats.update(_phase_bins(mod, phase, n_bins=32))
    feats.update(_spectral(mod, "mod"))

    feats["P_med"] = float(scan_data["vector_med_presion"].item())
    feats["P_cons"] = float(scan_data["vector_cons_presion"].item())
    feats["F_med"] = float(scan_data["vector_med_flujo"].item())
    feats["F_cons"] = float(scan_data["vector_cons_flujo"].item())
    feats["Temp"] = float(scan_data["vector_temp_Vflujo"].item())
    return feats


FEATURE_COLUMNS: List[str] = list(build_features({
    "modulo_remuestreado": np.zeros(200000),
    "fase_remuestreada": np.zeros(200000),
    "X_remuestreada": np.zeros(200000),
    "Y_remuestreada": np.zeros(200000),
    "vector_med_presion": np.array(1.0),
    "vector_cons_presion": np.array(1.0),
    "vector_med_flujo": np.array(1.0),
    "vector_cons_flujo": np.array(1.0),
    "vector_temp_Vflujo": np.array(1.0),
}).keys())
