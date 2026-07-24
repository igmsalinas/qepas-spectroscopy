"""Calibration scan discovery, loading, and array validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from ..core.config import DATA_DIR, LABELS

from ..core.schema import (
    REQUIRED_ARRAY_NAMES,
    SCALAR_ARRAY_NAMES,
    SIGNAL_ARRAY_NAMES,
)

_SCAN_FOLDER_PATTERN = re.compile(r"modulo_fase_X_Y_barrido_N_(\d+)$")


class MissingScanDataError(FileNotFoundError):
    """Raised when a scan does not contain every required array."""


@dataclass(frozen=True, slots=True)
class Scan:
    folder: str
    time: str
    n: int
    path: Path
    label_13co2: float
    label_12co2: float

    @property
    def labels(self) -> np.ndarray:
        return np.array(
            [self.label_13co2, self.label_12co2],
            dtype=np.float32,
        )

    @property
    def sample_id(self) -> str:
        return f"{self.time}/scan-{self.n:03d}"


def extract_time(folder: str) -> str | None:
    match = re.search(r"(\d{2})_(\d{2})_(\d{2})$", folder)
    return (
        f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
        if match
        else None
    )


def _discover_scan_folders(group: Path) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []
    for path in group.iterdir():
        if not path.is_dir():
            continue
        match = _SCAN_FOLDER_PATTERN.fullmatch(path.name)
        if match:
            discovered.append((int(match.group(1)), path))
    return sorted(discovered, key=lambda item: item[0])


def iter_scans(
    base_dir: str | Path = DATA_DIR,
    *,
    labels: Mapping[str, Mapping[str, float]] = LABELS,
    scans_per_group: int | None = 20,
) -> Iterator[Scan]:
    """Yield labeled scans in deterministic group and numeric scan order.

    Set ``scans_per_group=None`` to discover every available scan. The default
    cap deliberately creates a balanced 20-scan dataset across concentration
    groups for the bundled campaign.
    """
    base_path = Path(base_dir)
    if scans_per_group is not None and scans_per_group <= 0:
        raise ValueError("scans_per_group must be positive or None")
    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Calibration data directory does not exist: {base_path}"
        )

    folders = sorted(
        path
        for path in base_path.iterdir()
        if path.is_dir() and path.name.startswith("MEDICIONES")
    )
    for folder in folders:
        time = extract_time(folder.name)
        if time not in labels:
            continue
        scans = _discover_scan_folders(folder)
        if scans_per_group is not None:
            scans = scans[:scans_per_group]
        for number, scan_path in scans:
            yield Scan(
                folder=folder.name,
                time=time,
                n=number,
                path=scan_path,
                label_13co2=float(labels[time]["13CO2"]),
                label_12co2=float(labels[time]["12CO2"]),
            )


def validate_raw_arrays(
    data: Mapping[str, np.ndarray],
    *,
    require_all: bool = True,
) -> None:
    """Validate the numeric and dimensional contract of one scan."""
    if require_all:
        missing = set(REQUIRED_ARRAY_NAMES) - set(data)
        if missing:
            raise ValueError(f"Missing required arrays: {sorted(missing)}")

    signal_lengths: set[int] = set()
    for name in SIGNAL_ARRAY_NAMES:
        if name not in data:
            continue
        values = np.asarray(data[name])
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        signal_lengths.add(values.size)
    if len(signal_lengths) > 1:
        raise ValueError("All signal arrays must have the same length")

    for name in SCALAR_ARRAY_NAMES:
        if name not in data:
            continue
        value = np.asarray(data[name])
        if value.size != 1 or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{name} must be a numeric scalar")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def load_raw_arrays(
    scan_path: str | Path,
    *,
    required: bool = True,
) -> dict[str, np.ndarray]:
    """Load known arrays and fail early on incomplete or invalid scans."""
    directory = Path(scan_path)
    data: dict[str, np.ndarray] = {}
    missing: list[Path] = []
    for name in REQUIRED_ARRAY_NAMES:
        path = directory / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path, allow_pickle=False, mmap_mode="r")
        else:
            missing.append(path)

    if required and missing:
        names = ", ".join(path.name for path in missing)
        raise MissingScanDataError(
            f"Scan {directory} is missing required arrays: {names}"
        )
    validate_raw_arrays(data, require_all=required)
    return data


def load_labels_excel(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Sheet1", header=1)
