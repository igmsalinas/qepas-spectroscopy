"""Experiment identity and naming utilities."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_LABEL_PATTERN = re.compile(r"[^a-z0-9]+")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def slugify_experiment_label(label: str) -> str:
    """Return a filesystem-safe, non-empty experiment label."""
    slug = _LABEL_PATTERN.sub("-", label.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Experiment label must contain letters or numbers")
    return slug


def validate_experiment_id(experiment_id: str) -> str:
    """Reject path traversal and ambiguous experiment directory names."""
    if not _RUN_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "Experiment ID may contain only letters, numbers, dot, dash, "
            "and underscore"
        )
    return experiment_id


def create_experiment_id(
    label: str = "experiment",
    *,
    now: datetime | None = None,
) -> str:
    """Create a chronologically sortable and collision-resistant run ID."""
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    timestamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{slugify_experiment_label(label)}"
