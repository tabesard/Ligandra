"""Data-source contract and registry.

Every source returns activities in a single normalized schema
(:pyattr:`ligandra.core.types.Column.RAW_SCHEMA`) so downstream layers never
care where the data came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ligandra.core.registry import Registry
from ligandra.core.types import Column, Target

#: Registry of data sources. Add one with ``@DATA_SOURCES.register("name")``.
DATA_SOURCES: Registry[DataSource] = Registry("data source")


class DataSource(ABC):
    """Fetches bioactivity data for an arbitrary target/endpoint."""

    @abstractmethod
    def search_targets(self, query: str) -> list[Target]:
        """Return candidate targets matching a free-text ``query``."""

    @abstractmethod
    def fetch_activities(
        self, target_id: str, endpoint: str, **filters: object
    ) -> pd.DataFrame:
        """Return a DataFrame with at least :pyattr:`Column.RAW_SCHEMA`."""

    def select_target(self, query: str, endpoint: str) -> Target | None:
        """Pick the single best target for a free-text ``query`` + ``endpoint``.

        The default returns the top search hit.  Sources whose free-text search
        is noisy (ChEMBL text relevance can rank orthologs, protein families and
        adaptors above the intended human protein) override this to choose the
        target that actually carries the data.  ``endpoint`` is passed so a
        source can prefer a target that has activities for *this* endpoint.
        """
        hits = self.search_targets(query)
        return hits[0] if hits else None

    # -- shared helpers --------------------------------------------------
    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in Column.RAW_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                f"Data source returned columns {list(df.columns)}; "
                f"missing required {missing}."
            )
        return df
