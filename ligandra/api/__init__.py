"""Thin Python API over the pipeline (the library the UI/CLI both call).

Example
-------
>>> from ligandra.api import run, load_config
>>> cfg = load_config("examples/egfr_local.yaml")
>>> result = run(cfg)
>>> result.leaderboard.to_dataframe()
"""

from __future__ import annotations

from pathlib import Path

from ligandra.api.export import export_candidates
from ligandra.config.schema import ExperimentConfig
from ligandra.pipeline import PipelineResult, run_experiment


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)


def run(config: ExperimentConfig | str | Path, do_generate: bool = True) -> PipelineResult:
    """Run an experiment from a config object or a YAML path."""
    if not isinstance(config, ExperimentConfig):
        config = ExperimentConfig.from_yaml(config)
    return run_experiment(config, do_generate=do_generate)


# Registry inventories (drive UI dropdowns / `ligandra list`).
def registries() -> dict[str, list[str]]:
    import ligandra.data  # noqa: F401
    import ligandra.featurize  # noqa: F401
    import ligandra.generate  # noqa: F401
    import ligandra.models  # noqa: F401
    import ligandra.pretrained  # noqa: F401
    import ligandra.score  # noqa: F401
    from ligandra.data import DATA_SOURCES
    from ligandra.dock import DOCKERS
    from ligandra.featurize import FEATURIZERS
    from ligandra.generate import GENERATORS
    from ligandra.models import MODELS
    from ligandra.pretrained import PRETRAINED
    from ligandra.score import SCORERS

    return {
        "data_sources": DATA_SOURCES.available(),
        "featurizers": FEATURIZERS.available(),
        "models": MODELS.available(),
        "generators": GENERATORS.available(),
        "scorers": SCORERS.available(),
        "dockers": DOCKERS.available(),
        "pretrained": PRETRAINED.available(),
    }


__all__ = ["run", "load_config", "export_candidates", "registries", "PipelineResult"]
