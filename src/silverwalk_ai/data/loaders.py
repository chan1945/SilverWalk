"""Small file discovery and loading helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
GEO_EXTENSIONS = {".geojson", ".json", ".shp"}


def list_files(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    if not directory.exists():
        return []

    files = [path for path in directory.rglob("*") if path.is_file()]
    if extensions is None:
        return sorted(files)

    return sorted(path for path in files if path.suffix.lower() in extensions)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table file: {path}")
