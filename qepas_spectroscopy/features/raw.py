"""Raw-signal feature extraction for deep learning."""

from __future__ import annotations

import numpy as np


def _normalize_sample(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + 1e-8)


def downsample(arr: np.ndarray, target: int = 4096) -> np.ndarray:
    """Simple decimation to a fixed length."""
    if len(arr) <= target:
        return arr
    step = len(arr) // target
    return arr[::step][:target]


def build_raw_signal_features(
    scan_data: dict[str, np.ndarray],
    length: int = 4096,
    normalize: bool = False,
) -> dict[str, np.ndarray]:
    """Return downsampled 1-D vectors plus scalar env features.

    Args:
        scan_data: dictionary of loaded numpy arrays for one scan.
        length: target number of time samples after decimation.
        normalize: if True, apply per-sample standardization. The deep-learning
            pipeline will fit a dataset-level normalizer by default, so this is
            usually left False.
    """
    mod = downsample(scan_data["modulo_remuestreado"], length)
    phase = downsample(scan_data["fase_remuestreada"], length)
    X = downsample(scan_data["X_remuestreada"], length)
    Y = downsample(scan_data["Y_remuestreada"], length)

    if normalize:
        mod = _normalize_sample(mod)
        phase = _normalize_sample(phase)
        X = _normalize_sample(X)
        Y = _normalize_sample(Y)

    scalars = np.array([
        float(scan_data["vector_med_presion"].item()),
        float(scan_data["vector_cons_presion"].item()),
        float(scan_data["vector_med_flujo"].item()),
        float(scan_data["vector_cons_flujo"].item()),
        float(scan_data["vector_temp_Vflujo"].item()),
    ], dtype=np.float32)

    return {
        "signals": np.stack([mod, phase, X, Y], axis=-1).astype(np.float32),
        "scalars": scalars,
    }
