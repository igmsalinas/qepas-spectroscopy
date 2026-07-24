"""Project configuration and paths."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "calibration-measures"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FEATURES_DIR = OUTPUT_DIR / "features"
EDA_DIR = OUTPUT_DIR / "eda"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

LABELS = {
    "12:46:00": {"13CO2": 0.0, "12CO2": 0.0},
    "12:55:42": {"13CO2": 0.412372, "12CO2": 36.870962},
    "13:05:15": {"13CO2": 0.824743, "12CO2": 73.741923},
    "13:14:37": {"13CO2": 1.237115, "12CO2": 110.612885},
    "13:24:02": {"13CO2": 1.649487, "12CO2": 147.483847},
    "13:33:30": {"13CO2": 2.061858, "12CO2": 184.354808},
    "13:43:08": {"13CO2": 2.47423, "12CO2": 221.22577},
}

TARGETS = ["13CO2", "12CO2"]

RANDOM_STATE = 42


def ensure_dirs():
    for d in (MODELS_DIR, FEATURES_DIR, EDA_DIR):
        d.mkdir(parents=True, exist_ok=True)
