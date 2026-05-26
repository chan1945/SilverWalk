"""Runtime helpers shared by Streamlit entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def add_src_to_path() -> None:
    """Make the local src-layout package importable when run by Streamlit."""
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
