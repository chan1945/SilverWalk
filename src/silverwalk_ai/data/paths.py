"""Project path definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DataPaths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    external: Path = PROJECT_ROOT / "data" / "external"
    interim: Path = PROJECT_ROOT / "data" / "interim"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    outputs: Path = PROJECT_ROOT / "data" / "outputs"
    nodelink: Path = PROJECT_ROOT / "data" / "NODELINKDATA"
    sigungu: Path = PROJECT_ROOT / "data" / "SIGUNGU"
    artifacts: Path = PROJECT_ROOT / "artifacts"
    predictions: Path = PROJECT_ROOT / "artifacts" / "predictions"
    maps: Path = PROJECT_ROOT / "artifacts" / "maps"
    models: Path = PROJECT_ROOT / "artifacts" / "models"
    reports: Path = PROJECT_ROOT / "artifacts" / "reports"


PATHS = DataPaths()


def ensure_project_dirs() -> None:
    """Create writable output directories used by the app and pipeline."""
    for path in (
        PATHS.raw,
        PATHS.external,
        PATHS.interim,
        PATHS.processed,
        PATHS.outputs,
        PATHS.predictions,
        PATHS.maps,
        PATHS.models,
        PATHS.reports,
    ):
        path.mkdir(parents=True, exist_ok=True)
