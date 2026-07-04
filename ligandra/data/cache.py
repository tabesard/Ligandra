"""On-disk CSV cache for fetched + curated activity tables.

Fetching from ChEMBL is the slowest step of a run and it is deterministic for a
given target/endpoint, so we persist both the **raw** pull and the **curated**
table as CSVs under ``data_cache/``.  A repeat run reads the CSV instead of
hitting the network — and, unlike the in-memory UI cache, this survives an app
restart.  The files double as an export the user can open in Excel / pandas for
further analysis.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = "data_cache"


def _slug(text: object, maxlen: int = 60) -> str:
    """Filesystem-safe slug of an arbitrary key part."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return s[:maxlen] or "none"


def short_hash(*parts: object) -> str:
    """Short stable hash of arbitrary parts (used in curated filenames)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:8]


def raw_cache_path(
    source: str, target: object, endpoint: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> Path:
    """Path of the cached **raw** fetch for a (source, target, endpoint)."""
    name = f"{_slug(source)}__{_slug(target)}__{_slug(endpoint)}.raw.csv"
    return Path(cache_dir) / name


def curated_cache_path(
    source: str,
    target: object,
    endpoint: str,
    curation_sig: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Path:
    """Path of the cached **curated** table (keyed also on curation settings)."""
    name = f"{_slug(source)}__{_slug(target)}__{_slug(endpoint)}__{curation_sig}.curated.csv"
    return Path(cache_dir) / name


def load_dataframe(path: str | Path) -> pd.DataFrame | None:
    """Read a cached CSV, or ``None`` if it is missing/unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # pragma: no cover - corrupt/locked file → treat as miss
        return None


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` (creating parent dirs); return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
