"""Named engineered-feature sets for reproducible ablation studies."""

from __future__ import annotations

from collections.abc import Sequence

COMPACT_SPECTROSCOPY_FEATURES = (
    "mod_mean",
    "mod_std",
    "mod_median",
    "mod_p10",
    "mod_p90",
    "mod_skew",
    "mod_kurt",
    "phase_std",
    "phase_median",
    "phase_p10",
    "phase_p90",
    "phase_kurt",
    "phase_resultant_length",
    "phase_sin_mean",
    "phase_cos_mean",
    "phase_bin_05_mean",
    "phase_bin_23_mean",
    "X_std",
    "Y_mean",
    "mod_psd_total",
    "mod_psd_max",
    "mod_peak0_pow",
    "P_med",
    "F_med",
    "Temp",
)


def feature_indices(
    available: Sequence[str],
    selected: Sequence[str],
) -> tuple[int, ...]:
    """Resolve a named feature set and fail loudly when schemas drift."""
    positions = {name: index for index, name in enumerate(available)}
    missing = [name for name in selected if name not in positions]
    if missing:
        raise ValueError(f"Feature set contains unavailable features: {missing}")
    if len(set(selected)) != len(selected):
        raise ValueError("Feature set contains duplicate names")
    return tuple(positions[name] for name in selected)
