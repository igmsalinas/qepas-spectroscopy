"""Command-line adapter for the calibration data profiler."""

from __future__ import annotations

from pathlib import Path

import typer

from ..application.analysis import DataAnalysisPipeline
from ..core.config import EDA_DIR

app = typer.Typer()


@app.command()
def profile(
    output_dir: Path = typer.Option(
        EDA_DIR,
        help="Directory for CSV, JSON, and Markdown profile artifacts",
    ),
) -> None:
    """Inspect every discovered calibration scan without training models."""
    DataAnalysisPipeline(output_dir=output_dir, reporter=typer.echo).run()


if __name__ == "__main__":
    app()
