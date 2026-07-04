"""Typed experiment configuration (Pydantic v2).

A single :class:`ExperimentConfig` fully describes an end-to-end run: target,
endpoint, curation thresholds, featurizers, models, generator and objectives.
The Streamlit UI writes this object; the CLI runs the identical object headless
(``ligandra run experiment.yaml``).  Nothing about the pipeline is hard-coded
to a target, endpoint, descriptor set or model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ligandra.core.types import Endpoint, SplitStrategy, TaskType


class TargetConfig(BaseModel):
    """Which target to study and where to pull its data from."""

    source: str = "chembl"  # key into DATA_SOURCES
    query: str | None = None  # free-text search (gene name, family, ...)
    target_id: str | None = None  # resolved id, e.g. a ChEMBL target id
    endpoint: Endpoint = Endpoint.IC50

    # For local sources:
    path: str | None = None
    smiles_col: str = "smiles"
    value_col: str = "value"
    id_col: str | None = None


class CurationConfig(BaseModel):
    active_threshold: float = 1000.0  # nM; <= is active
    inactive_threshold: float = 10000.0  # nM; >= is inactive
    default_units: str = "nM"
    standardize: bool = True
    dedupe: bool = True
    drop_invalid: bool = True


class SplitConfig(BaseModel):
    strategy: SplitStrategy = SplitStrategy.SCAFFOLD
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42


class FeaturizerConfig(BaseModel):
    name: str  # key into FEATURIZERS
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str  # key into MODELS
    params: dict[str, Any] = Field(default_factory=dict)
    #: optional pretrained checkpoint (HF hub id or local path) for finetuning
    pretrained: str | None = None
    finetune: bool = False


class ObjectiveConfig(BaseModel):
    scorer: str  # key into SCORERS
    weight: float = 1.0
    minimize: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class GeneratorConfig(BaseModel):
    name: str = "graph_ga"  # key into GENERATORS
    params: dict[str, Any] = Field(default_factory=dict)
    budget: int = 200  # number of scoring calls / candidates to explore
    n_output: int = 50  # size of the ranked candidate set to keep


class CacheConfig(BaseModel):
    """On-disk CSV cache for fetched + curated data (skips re-fetching)."""

    enabled: bool = True
    dir: str = "data_cache"
    refresh: bool = False  # force a re-fetch even if a cached CSV exists


class TrackingConfig(BaseModel):
    backend: str = "local"  # "local" run manifest, or "mlflow"
    experiment_name: str = "ligandra"
    output_dir: str = "runs"


class ExperimentConfig(BaseModel):
    """The full experiment. Serialises to/from YAML round-trip cleanly."""

    name: str = "experiment"
    task: TaskType = TaskType.REGRESSION
    seed: int = 42

    target: TargetConfig = Field(default_factory=TargetConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    featurizers: list[FeaturizerConfig] = Field(
        default_factory=lambda: [FeaturizerConfig(name="ecfp")]
    )
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(name="ridge"),
            ModelConfig(name="random_forest"),
        ]
    )
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    objectives: list[ObjectiveConfig] = Field(
        default_factory=lambda: [
            ObjectiveConfig(scorer="potency", weight=1.0),
            ObjectiveConfig(scorer="qed", weight=0.5),
            ObjectiveConfig(scorer="sa", weight=0.5),
        ]
    )
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # -- serialisation ---------------------------------------------------
    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
        return path

    def to_yaml_str(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data or {})

    @classmethod
    def from_yaml_str(cls, text: str) -> ExperimentConfig:
        return cls.model_validate(yaml.safe_load(text) or {})
