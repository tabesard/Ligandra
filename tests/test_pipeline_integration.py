"""End-to-end integration test on the cached mini-dataset (no network)."""

from __future__ import annotations

from ligandra.config.schema import (
    CacheConfig,
    ExperimentConfig,
    FeaturizerConfig,
    GeneratorConfig,
    ModelConfig,
    ObjectiveConfig,
    SplitConfig,
    TargetConfig,
    TrackingConfig,
)
from ligandra.core.types import Endpoint, SplitStrategy
from ligandra.pipeline import run_experiment


def _fast_config(dataset_path: str, out_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="itest",
        target=TargetConfig(
            source="local", path=dataset_path, smiles_col="smiles",
            value_col="value", id_col="molecule_id", endpoint=Endpoint.IC50,
        ),
        split=SplitConfig(strategy=SplitStrategy.SCAFFOLD, test_size=0.25, val_size=0.0),
        featurizers=[FeaturizerConfig(name="lipinski")],
        models=[ModelConfig(name="ridge")],
        generator=GeneratorConfig(name="graph_ga", params={"population_size": 12}, budget=40, n_output=10),
        objectives=[
            ObjectiveConfig(scorer="potency", weight=1.0, params={"center": 6.0, "scale": 0.5}),
            ObjectiveConfig(scorer="qed", weight=0.5),
            ObjectiveConfig(scorer="novelty", weight=0.25),
        ],
        tracking=TrackingConfig(backend="local", output_dir=out_dir),
        cache=CacheConfig(dir=out_dir),  # keep cache CSVs in the tmp dir
    )


def test_run_experiment_end_to_end(mini_dataset_path, tmp_path):
    cfg = _fast_config(mini_dataset_path, str(tmp_path / "runs"))
    result = run_experiment(cfg)

    # Curation produced data.
    assert len(result.curated) > 10
    assert result.dataset_hash

    # Leaderboard has a real scored model on the scaffold split.
    lb = result.leaderboard.to_dataframe()
    assert not lb.empty
    assert "R2" in lb.columns
    assert result.best_model_name == "ridge"

    # Generation produced novel, valid, unique candidates with score breakdown.
    cand = result.candidates
    assert cand is not None and not cand.empty
    train_smiles = set(result.curated["smiles"])
    assert all(s not in train_smiles for s in cand["smiles"])  # de-duplicated vs training
    for col in ("objective", "potency", "qed"):
        assert col in cand.columns
    assert result.generation_metrics["validity"] == 1.0

    # Reproducible artifacts were written.
    assert result.run_dir is not None
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "candidates.csv").exists()
    assert (result.run_dir / "leaderboard.json").exists()


def test_yaml_roundtrip_runs(tmp_path, mini_dataset_path):
    cfg = _fast_config(mini_dataset_path, str(tmp_path / "runs"))
    path = tmp_path / "exp.yaml"
    cfg.to_yaml(path)
    reloaded = ExperimentConfig.from_yaml(path)
    assert reloaded.target.path == mini_dataset_path
    # A reloaded config runs identically (train-only for speed).
    result = run_experiment(reloaded, do_generate=False)
    assert not result.leaderboard.to_dataframe().empty
