"""Featurizer shapes/caching and model fit/predict/save/load round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from ligandra.core.molecule import MoleculeSet
from ligandra.featurize import FEATURIZERS, build_featurizer
from ligandra.models import Leaderboard, build_model
from ligandra.models.classical import RandomForestModel, RidgeModel
from ligandra.core.types import TaskType


@pytest.mark.parametrize("name,expected_cols", [("lipinski", 4), ("maccs", 167)])
def test_featurizer_shapes(sample_smiles, name, expected_cols):
    ms = MoleculeSet.from_smiles(sample_smiles)
    X = build_featurizer(name).transform(ms)
    assert X.shape == (len(sample_smiles), expected_cols)
    assert np.isfinite(X).all()


def test_ecfp_bits_configurable(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    X = build_featurizer("ecfp", n_bits=256).transform(ms)
    assert X.shape[1] == 256
    assert set(np.unique(X)) <= {0.0, 1.0}


def test_feature_cache_is_used(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    feat = build_featurizer("lipinski")
    feat.transform(ms)
    assert ms[0].has_feature("lipinski")


def _xy(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    X = build_featurizer("ecfp", n_bits=512).transform(ms)
    y = np.arange(len(sample_smiles), dtype=float)
    return X, y


@pytest.mark.parametrize("name", ["linear", "ridge", "lasso", "random_forest", "svr"])
def test_model_fit_predict(sample_smiles, name):
    X, y = _xy(sample_smiles)
    model = build_model(name).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_model_save_load_roundtrip(sample_smiles, tmp_path):
    X, y = _xy(sample_smiles)
    model = RidgeModel().fit(X, y)
    path = tmp_path / "ridge.pkl"
    model.save(path)
    loaded = RidgeModel.load(path)
    assert np.allclose(model.predict(X), loaded.predict(X))


def test_random_forest_uncertainty(sample_smiles):
    X, y = _xy(sample_smiles)
    rf = RandomForestModel().fit(X, y)
    mean, std = rf.predict_with_uncertainty(X)
    assert mean.shape == std.shape == (len(y),)
    assert (std >= 0).all()


def test_leaderboard_ranks_by_primary_metric():
    board = Leaderboard(task=TaskType.REGRESSION)
    from ligandra.models.base import LeaderboardEntry

    board.add(LeaderboardEntry("a", "ecfp", {"R2": 0.2, "RMSE": 1.0}))
    board.add(LeaderboardEntry("b", "ecfp", {"R2": 0.9, "RMSE": 0.3}))
    df = board.to_dataframe()
    assert df.iloc[0]["model"] == "b"  # higher R2 first
    assert board.best().model_name == "b"
