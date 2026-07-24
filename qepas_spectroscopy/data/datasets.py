"""Model-view assembly pipeline for validated calibration scans."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import TARGETS
from ..core.schema import SCALAR_ARRAY_NAMES
from ..features.engineered import EngineeredFeatureExtractor
from ..features.raw import (
    RawSignalPreprocessingConfig,
    RawSignalPreprocessor,
)
from .scans import (
    Scan,
    iter_scans,
    load_raw_arrays,
    validate_raw_arrays,
)

ScanSource = Callable[[], Iterable[Scan]]
ArrayLoader = Callable[[str | Path], dict[str, np.ndarray]]


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class EngineeredDataset:
    """Validated tabular dataset and its model-facing array views."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.frame.empty:
            raise ValueError("Engineered dataset is empty")
        required = {
            *self.feature_names,
            *TARGETS,
            "sample_id",
            "folder",
            "time",
            "N",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(
                f"Engineered dataset is missing columns: {sorted(missing)}"
            )
        if self.frame.loc[:, "sample_id"].duplicated().any():
            raise ValueError("Engineered sample identifiers must be unique")
        _require_finite("features", self.features)
        _require_finite("targets", self.targets)

    @property
    def features(self) -> np.ndarray:
        return self.frame.loc[:, self.feature_names].to_numpy(dtype=np.float32)

    @property
    def targets(self) -> np.ndarray:
        return self.frame.loc[:, TARGETS].to_numpy(dtype=np.float32)

    @property
    def groups(self) -> np.ndarray:
        return self.frame.loc[:, "time"].to_numpy()

    @property
    def sample_ids(self) -> np.ndarray:
        return self.frame.loc[:, "sample_id"].to_numpy()


@dataclass(frozen=True, slots=True)
class RawSignalDataset:
    """Validated arrays consumed by deep-learning models."""

    signals: np.ndarray
    scalars: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    signal_names: tuple[str, ...]
    preprocessing_id: str
    scalar_names: tuple[str, ...] = SCALAR_ARRAY_NAMES

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError("signals must have shape (samples, time, channels)")
        if self.scalars.ndim != 2:
            raise ValueError("scalars must have shape (samples, features)")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape (samples, targets)")
        if self.groups.ndim != 1 or self.sample_ids.ndim != 1:
            raise ValueError("groups and sample_ids must have shape (samples,)")

        sample_counts = {
            self.signals.shape[0],
            self.scalars.shape[0],
            self.targets.shape[0],
            self.groups.shape[0],
            self.sample_ids.shape[0],
        }
        if len(sample_counts) != 1:
            raise ValueError(
                "All raw dataset arrays must have the same number of samples"
            )
        if self.signals.shape[0] == 0:
            raise ValueError("Raw signal dataset is empty")
        if not self.preprocessing_id:
            raise ValueError("preprocessing_id cannot be empty")
        if self.signals.shape[2] != len(self.signal_names):
            raise ValueError("signal_names must match the signal channel count")
        if self.scalars.shape[1] != len(self.scalar_names):
            raise ValueError("scalar_names must match the scalar feature count")
        if np.unique(self.sample_ids).size != self.sample_ids.size:
            raise ValueError("Raw sample identifiers must be unique")

        _require_finite("signals", self.signals)
        _require_finite("scalars", self.scalars)
        _require_finite("targets", self.targets)


@dataclass(frozen=True, slots=True)
class DatasetViews:
    """Aligned tabular and signal views produced in one I/O pass."""

    engineered: EngineeredDataset
    raw: RawSignalDataset


class DatasetPipeline:
    """Extract and preprocess model views from an injected scan source."""

    def __init__(
        self,
        scan_source: ScanSource = iter_scans,
        array_loader: ArrayLoader = load_raw_arrays,
        feature_extractor: EngineeredFeatureExtractor | None = None,
    ) -> None:
        self._scan_source = scan_source
        self._array_loader = array_loader
        self._feature_extractor = (
            feature_extractor or EngineeredFeatureExtractor()
        )

    def build_engineered(self) -> EngineeredDataset:
        engineered, _ = self._assemble(raw_preprocessor=None)
        if engineered is None:
            raise RuntimeError("Engineered dataset assembly did not run")
        return engineered

    def build_raw(
        self,
        config: RawSignalPreprocessingConfig | None = None,
    ) -> RawSignalDataset:
        raw_config = config or RawSignalPreprocessingConfig()
        _, raw = self._assemble(
            raw_preprocessor=RawSignalPreprocessor(raw_config),
            include_engineered=False,
        )
        if raw is None:
            raise RuntimeError("Raw dataset assembly did not run")
        return raw

    def build_views(
        self,
        config: RawSignalPreprocessingConfig | None = None,
    ) -> DatasetViews:
        raw_config = config or RawSignalPreprocessingConfig()
        engineered, raw = self._assemble(
            raw_preprocessor=RawSignalPreprocessor(raw_config)
        )
        if engineered is None or raw is None:
            raise RuntimeError("Aligned dataset assembly did not run")
        return DatasetViews(engineered=engineered, raw=raw)

    def _assemble(
        self,
        *,
        raw_preprocessor: RawSignalPreprocessor | None,
        include_engineered: bool = True,
    ) -> tuple[EngineeredDataset | None, RawSignalDataset | None]:
        records: list[dict[str, float | int | str]] = []
        signals: list[np.ndarray] = []
        scalars: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        groups: list[str] = []
        sample_ids: list[str] = []
        signal_names: tuple[str, ...] | None = None

        for scan in self._scan_source():
            arrays = self._array_loader(scan.path)
            validate_raw_arrays(arrays)
            if include_engineered:
                features = self._feature_extractor.transform(arrays)
                records.append(
                    {
                        **features,
                        "sample_id": scan.sample_id,
                        "folder": scan.folder,
                        "time": scan.time,
                        "N": scan.n,
                        "13CO2": scan.label_13co2,
                        "12CO2": scan.label_12co2,
                    }
                )
            if raw_preprocessor is not None:
                prepared = raw_preprocessor.transform(arrays)
                signals.append(prepared.signals)
                scalars.append(prepared.scalars)
                targets.append(scan.labels)
                groups.append(scan.time)
                sample_ids.append(scan.sample_id)
                signal_names = prepared.signal_names

        engineered: EngineeredDataset | None = None
        if include_engineered:
            frame = pd.DataFrame.from_records(records)
            engineered = EngineeredDataset(
                frame=frame,
                feature_names=self._feature_extractor.feature_names,
            )

        raw: RawSignalDataset | None = None
        if raw_preprocessor is not None:
            if not signals or signal_names is None:
                raise ValueError("Raw signal dataset is empty")
            raw = RawSignalDataset(
                signals=np.stack(signals),
                scalars=np.stack(scalars),
                targets=np.stack(targets),
                groups=np.asarray(groups),
                sample_ids=np.asarray(sample_ids),
                signal_names=signal_names,
                preprocessing_id=raw_preprocessor.config.fingerprint,
            )
        return engineered, raw
