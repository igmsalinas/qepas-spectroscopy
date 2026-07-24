"""Streaming data-quality and signal-relationship profiling."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scans import Scan, iter_scans, load_raw_arrays, validate_raw_arrays

ScanSource = Callable[[], Iterable[Scan]]
ArrayLoader = Callable[[str | Path], dict[str, np.ndarray]]


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(left.corr(right))
    return value if np.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """A machine-readable summary plus per-scan observations."""

    summary: dict[str, Any]
    scans: pd.DataFrame

    def save(self, output_dir: str | Path) -> tuple[Path, Path, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "scan_profile.csv"
        json_path = directory / "data_profile.json"
        markdown_path = directory / "data_profile.md"
        self.scans.to_csv(csv_path, index=False)
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(self.summary, stream, indent=2, allow_nan=False)
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return csv_path, json_path, markdown_path

    def to_markdown(self) -> str:
        groups = self.summary["group_counts"]
        group_lines = "\n".join(
            f"| {group} | {count} |" for group, count in groups.items()
        )
        correlations = self.summary["correlations_with_13co2"]
        correlation_lines = "\n".join(
            f"| {name} | {value:.6f} |"
            for name, value in correlations.items()
            if value is not None
        )
        balanced_count = self.summary["balanced_scan_count"]
        per_group = self.summary["balanced_scans_per_group"]
        modulus_error = self.summary["modulus_identity_relative_mae"]
        phase_error = self.summary["phase_xy_circular_mae_radians"]
        median_wraps = self.summary["phase_wrap_jumps"]["median"]
        target_correlation = self.summary["target_correlation"]
        return f"""# QEPAS data profile

Generated from the calibration arrays without modifying them.

## Inventory

- Discovered scans: {self.summary['scan_count']}
- Concentration groups: {self.summary['group_count']}
- Balanced training selection: {balanced_count} scans ({per_group} per group)
- Signal lengths: {self.summary['signal_lengths']}
- Signal dtypes: {self.summary['signal_dtypes']}
- Non-finite values: {self.summary['nonfinite_values']}

| Group | Scans |
|---|---:|
{group_lines}

## Signal integrity

- Relative MAE between stored modulus and `hypot(X, Y)`: {modulus_error:.9g}
- Circular MAE between stored phase and `atan2(X, Y)`: {phase_error:.9g} radians
- Phase-wrap jumps per scan (median): {median_wraps}

## Target structure

The correlation between 13CO2 and 12CO2 labels is
{target_correlation:.12f}. The campaign therefore varies total mixture
concentration but does not independently vary isotope ratio.

## Correlation with 13CO2

These values are descriptive only; group-disjoint validation remains mandatory.

| Per-scan statistic | Pearson r |
|---|---:|
{correlation_lines}
"""


class DatasetProfiler:
    """Profile a campaign one scan at a time to bound memory use."""

    def __init__(
        self,
        scan_source: ScanSource | None = None,
        array_loader: ArrayLoader = load_raw_arrays,
    ) -> None:
        self._scan_source = scan_source or (
            lambda: iter_scans(scans_per_group=None)
        )
        self._array_loader = array_loader

    def profile(self) -> DatasetProfile:
        records: list[dict[str, float | int | str]] = []
        signal_lengths: set[int] = set()
        signal_dtypes: set[str] = set()
        nonfinite_values = 0

        for scan in self._scan_source():
            arrays = self._array_loader(scan.path)
            validate_raw_arrays(arrays)
            modulus = np.asarray(arrays["modulo_remuestreado"])
            phase = np.asarray(arrays["fase_remuestreada"])
            x_component = np.asarray(arrays["X_remuestreada"])
            y_component = np.asarray(arrays["Y_remuestreada"])
            for values in (modulus, phase, x_component, y_component):
                signal_lengths.add(int(values.size))
                signal_dtypes.add(str(values.dtype))
                nonfinite_values += int(
                    np.count_nonzero(~np.isfinite(values))
                )

            denominator = max(float(np.mean(np.abs(modulus))), 1e-12)
            phase_difference = np.angle(
                np.exp(1j * (phase - np.arctan2(x_component, y_component)))
            )
            records.append(
                {
                    "sample_id": scan.sample_id,
                    "time": scan.time,
                    "N": scan.n,
                    "13CO2": scan.label_13co2,
                    "12CO2": scan.label_12co2,
                    "signal_length": int(modulus.size),
                    "mod_mean": float(modulus.mean()),
                    "mod_std": float(modulus.std()),
                    "mod_max": float(modulus.max()),
                    "phase_sin_mean": float(np.sin(phase).mean()),
                    "phase_cos_mean": float(np.cos(phase).mean()),
                    "phase_wrap_jumps": int(
                        np.count_nonzero(np.abs(np.diff(phase)) > np.pi)
                    ),
                    "modulus_identity_relative_mae": float(
                        np.mean(
                            np.abs(modulus - np.hypot(x_component, y_component))
                        )
                        / denominator
                    ),
                    "phase_xy_circular_mae_radians": float(
                        np.mean(np.abs(phase_difference))
                    ),
                    "P_med": float(
                        np.asarray(arrays["vector_med_presion"]).item()
                    ),
                    "P_cons": float(
                        np.asarray(arrays["vector_cons_presion"]).item()
                    ),
                    "F_med": float(
                        np.asarray(arrays["vector_med_flujo"]).item()
                    ),
                    "F_cons": float(
                        np.asarray(arrays["vector_cons_flujo"]).item()
                    ),
                    "Temp": float(
                        np.asarray(arrays["vector_temp_Vflujo"]).item()
                    ),
                }
            )

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            raise ValueError("Cannot profile an empty dataset")
        group_counts = Counter(frame.loc[:, "time"])
        balanced_scans_per_group = min(group_counts.values())
        correlation_columns = (
            "mod_mean",
            "mod_std",
            "mod_max",
            "phase_sin_mean",
            "phase_cos_mean",
            "P_med",
            "P_cons",
            "F_med",
            "F_cons",
            "Temp",
        )
        summary: dict[str, Any] = {
            "scan_count": len(frame),
            "group_count": len(group_counts),
            "group_counts": dict(sorted(group_counts.items())),
            "balanced_scans_per_group": balanced_scans_per_group,
            "balanced_scan_count": balanced_scans_per_group * len(group_counts),
            "signal_lengths": sorted(signal_lengths),
            "signal_dtypes": sorted(signal_dtypes),
            "nonfinite_values": nonfinite_values,
            "modulus_identity_relative_mae": float(
                frame.loc[:, "modulus_identity_relative_mae"].mean()
            ),
            "phase_xy_circular_mae_radians": float(
                frame.loc[:, "phase_xy_circular_mae_radians"].mean()
            ),
            "phase_wrap_jumps": {
                "min": int(frame.loc[:, "phase_wrap_jumps"].min()),
                "median": float(frame.loc[:, "phase_wrap_jumps"].median()),
                "max": int(frame.loc[:, "phase_wrap_jumps"].max()),
            },
            "target_correlation": float(
                frame.loc[:, "13CO2"].corr(frame.loc[:, "12CO2"])
            ),
            "correlations_with_13co2": {
                name: _correlation(
                    frame.loc[:, name], frame.loc[:, "13CO2"]
                )
                for name in correlation_columns
            },
            "scalar_unique_counts": {
                name: int(frame.loc[:, name].nunique())
                for name in ("P_med", "P_cons", "F_med", "F_cons", "Temp")
            },
        }
        return DatasetProfile(summary=summary, scans=frame)
