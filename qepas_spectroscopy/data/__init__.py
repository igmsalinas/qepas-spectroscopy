"""Scan I/O, preprocessing, normalization, profiling, and datasets."""

from .augmentation import (
    AugmentedTrainingBatch,
    SpectralAugmentationConfig,
    augment_training_partition,
)
from .datasets import (
    DatasetPipeline,
    DatasetViews,
    EngineeredDataset,
    RawSignalDataset,
)
from .normalization import (
    NormalizationBundle,
    ScalarNormalizer,
    SignalNormalizer,
    TargetNormalizer,
)
from .profiling import DatasetProfile, DatasetProfiler
from .scans import (
    MissingScanDataError,
    REQUIRED_ARRAY_NAMES,
    SCALAR_ARRAY_NAMES,
    SIGNAL_ARRAY_NAMES,
    Scan,
    extract_time,
    iter_scans,
    load_labels_excel,
    load_raw_arrays,
    validate_raw_arrays,
)

__all__ = [
    "Scan",
    "MissingScanDataError",
    "SIGNAL_ARRAY_NAMES",
    "SCALAR_ARRAY_NAMES",
    "REQUIRED_ARRAY_NAMES",
    "extract_time",
    "iter_scans",
    "load_raw_arrays",
    "validate_raw_arrays",
    "load_labels_excel",
    "EngineeredDataset",
    "RawSignalDataset",
    "DatasetViews",
    "DatasetPipeline",
    "SignalNormalizer",
    "ScalarNormalizer",
    "TargetNormalizer",
    "NormalizationBundle",
    "DatasetProfile",
    "DatasetProfiler",
    "SpectralAugmentationConfig",
    "AugmentedTrainingBatch",
    "augment_training_partition",
]
