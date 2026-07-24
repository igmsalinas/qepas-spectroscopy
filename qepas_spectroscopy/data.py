"""Data loading utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import pandas as pd

from .config import DATA_DIR, LABELS


@dataclass
class Scan:
    folder: str
    time: str
    n: int
    path: str
    label_13co2: float
    label_12co2: float

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.label_13co2, self.label_12co2], dtype=np.float32)


def extract_time(folder: str) -> str | None:
    m = re.search(r"(\d{2})_(\d{2})_(\d{2})$", folder)
    return f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else None


def iter_scans(base_dir: Union[str, os.PathLike] = DATA_DIR) -> Iterator[Scan]:
    base_dir = Path(base_dir)
    folders = [d for d in sorted(os.listdir(base_dir)) if d.startswith("MEDICIONES")]
    for fold in folders:
        t = extract_time(fold)
        if t not in LABELS:
            continue
        for n in range(20):
            scan_path = base_dir / fold / f"modulo_fase_X_Y_barrido_N_{n}"
            if not scan_path.is_dir():
                continue
            yield Scan(
                folder=fold,
                time=t,
                n=n,
                path=str(scan_path),
                label_13co2=LABELS[t]["13CO2"],
                label_12co2=LABELS[t]["12CO2"],
            )


def load_raw_arrays(scan_path: Union[str, os.PathLike]) -> dict[str, np.ndarray]:
    names = [
        "modulo_remuestreado",
        "fase_remuestreada",
        "X_remuestreada",
        "Y_remuestreada",
        "vector_med_presion",
        "vector_cons_presion",
        "vector_med_flujo",
        "vector_cons_flujo",
        "vector_temp_Vflujo",
    ]
    data = {}
    scan_path = Path(scan_path)
    for name in names:
        path = scan_path / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path)
    return data


def load_labels_excel(path: Union[str, os.PathLike]) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Sheet1", header=1)
