"""On-disk CSV cache: a repeat run reads the CSV instead of re-fetching."""

from __future__ import annotations

import pandas as pd

from ligandra.config.schema import CacheConfig, ExperimentConfig, TargetConfig
from ligandra.core.types import Column, Endpoint
from ligandra.data.base import DATA_SOURCES, DataSource
from ligandra.data.cache import load_dataframe, raw_cache_path, save_dataframe
from ligandra.pipeline.runner import load_and_curate

_ROWS = [
    ("CCO", 800.0),
    ("c1ccccc1", 5000.0),
    ("CC(=O)O", 20000.0),
    ("CC(=O)Nc1ccc(O)cc1", 1200.0),
    ("O=C(O)c1ccccc1", 900.0),
    ("COc1ccc(CCN)cc1", 2500.0),
]


@DATA_SOURCES.register("count_source")
class _CountingSource(DataSource):
    n_fetch = 0

    def search_targets(self, query: str):
        from ligandra.core.types import Target

        return [Target(target_id="T1", name=query, source="count")]

    def fetch_activities(self, target_id: str, endpoint: str, **filters):
        type(self).n_fetch += 1
        return pd.DataFrame(
            {
                Column.MOLECULE_ID: [f"m{i}" for i in range(len(_ROWS))],
                Column.SMILES: [s for s, _ in _ROWS],
                Column.TARGET_ID: [target_id] * len(_ROWS),
                Column.ENDPOINT: [endpoint] * len(_ROWS),
                Column.VALUE: [v for _, v in _ROWS],
                Column.UNITS: ["nM"] * len(_ROWS),
            }
        )


def test_save_and_load_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "sub" / "t.csv"
    save_dataframe(df, path)
    assert path.exists()
    pd.testing.assert_frame_equal(load_dataframe(path), df)
    assert load_dataframe(tmp_path / "missing.csv") is None


def test_second_run_reads_cache_and_skips_fetch(tmp_path):
    _CountingSource.n_fetch = 0
    cfg = ExperimentConfig(
        name="cache-test",
        target=TargetConfig(source="count_source", target_id="T1", endpoint=Endpoint.IC50),
        cache=CacheConfig(enabled=True, dir=str(tmp_path)),
    )

    curated1, _ = load_and_curate(cfg)
    assert _CountingSource.n_fetch == 1  # fetched once
    # raw CSV was written for reuse / analysis
    assert raw_cache_path("count_source", "T1", "IC50", str(tmp_path)).exists()

    curated2, _ = load_and_curate(cfg)
    assert _CountingSource.n_fetch == 1  # NOT re-fetched — served from disk cache
    assert len(curated2) == len(curated1)


def test_refresh_forces_a_refetch(tmp_path):
    _CountingSource.n_fetch = 0
    base = dict(source="count_source", target_id="T1", endpoint=Endpoint.IC50)
    cfg = ExperimentConfig(
        target=TargetConfig(**base), cache=CacheConfig(enabled=True, dir=str(tmp_path))
    )
    load_and_curate(cfg)
    assert _CountingSource.n_fetch == 1

    cfg_refresh = ExperimentConfig(
        target=TargetConfig(**base),
        cache=CacheConfig(enabled=True, dir=str(tmp_path), refresh=True),
    )
    load_and_curate(cfg_refresh)
    assert _CountingSource.n_fetch == 2  # refresh bypassed the cache


def test_disabled_cache_writes_nothing(tmp_path):
    _CountingSource.n_fetch = 0
    cfg = ExperimentConfig(
        target=TargetConfig(source="count_source", target_id="T1", endpoint=Endpoint.IC50),
        cache=CacheConfig(enabled=False, dir=str(tmp_path)),
    )
    load_and_curate(cfg)
    load_and_curate(cfg)
    assert _CountingSource.n_fetch == 2  # no caching → fetched each time
    assert not raw_cache_path("count_source", "T1", "IC50", str(tmp_path)).exists()
