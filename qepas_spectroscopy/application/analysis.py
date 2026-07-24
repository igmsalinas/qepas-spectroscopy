"""Application service for calibration data analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import EDA_DIR
from ..data.profiling import DatasetProfile, DatasetProfiler

Reporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    profile: DatasetProfile
    artifacts: tuple[Path, Path, Path]


@dataclass(slots=True)
class DataAnalysisPipeline:
    """Profile the complete campaign and persist reproducible artifacts."""

    profiler: DatasetProfiler = field(default_factory=DatasetProfiler)
    output_dir: Path = EDA_DIR
    reporter: Reporter = print

    def run(self) -> AnalysisRun:
        self.reporter("Profiling calibration scans...")
        profile = self.profiler.profile()
        artifacts = profile.save(self.output_dir)
        self.reporter(
            f"Profiled {profile.summary['scan_count']} scans across "
            f"{profile.summary['group_count']} groups"
        )
        for artifact in artifacts:
            self.reporter(f"Saved {artifact}")
        return AnalysisRun(profile=profile, artifacts=artifacts)
