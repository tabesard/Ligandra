"""Scoring functions, Pareto ranking, and the generative baseline-beating check."""

from __future__ import annotations

import numpy as np

from ligandra.core.molecule import MoleculeSet
from ligandra.featurize import build_featurizer
from ligandra.generate import build_generator, generation_metrics
from ligandra.models import build_model
from ligandra.score import build_scorer
from ligandra.score.base import WeightedSumObjective, pareto_front


def test_scorers_return_unit_interval(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    for name in ("qed", "sa", "druglikeness", "novelty"):
        vals = build_scorer(name)(ms)
        assert vals.shape == (len(ms),)
        assert (vals >= 0).all() and (vals <= 1).all()


def test_novelty_reference_makes_known_mols_unnovel(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    scorer = build_scorer("novelty")
    scorer.set_reference(sample_smiles)
    vals = scorer(ms)
    # Identical molecules to the reference should have ~zero novelty.
    assert vals.max() < 1e-6


def test_pareto_front_simple():
    # rows: (obj1, obj2) maximize. (2,2) dominates (1,1); (0,3) and (3,0) are non-dominated.
    mat = np.array([[1, 1], [2, 2], [0, 3], [3, 0]], dtype=float)
    front = set(pareto_front(mat).tolist())
    assert front == {1, 2, 3}


def test_potency_scorer_requires_binding(sample_smiles):
    ms = MoleculeSet.from_smiles(sample_smiles)
    scorer = build_scorer("potency")
    try:
        scorer(ms)
        assert False, "expected RuntimeError when unbound"
    except RuntimeError:
        pass


def test_potency_scorer_works_for_smiles_model_without_featurizer(sample_smiles):
    """A SMILES-consuming model (e.g. the foundation model) is its own
    featurizer, so potency must score it with featurizer=None (regression)."""
    ms = MoleculeSet.from_smiles(sample_smiles)

    class _SmilesModel:
        consumes_smiles = True

        def predict(self, X):
            # X is a list of SMILES strings, not a feature matrix
            assert isinstance(X, list) and all(isinstance(s, str) for s in X)
            return np.full(len(X), 7.0)

    scorer = build_scorer("potency", center=6.0, scale=1.0).bind(_SmilesModel(), None)
    out = scorer(ms)
    assert out.shape == (len(sample_smiles),)
    assert np.all((out >= 0) & (out <= 1)) and np.all(out > 0.5)


def test_generator_metrics_valid_unique(sample_smiles):
    gen = build_generator("graph_ga", seed_smiles=sample_smiles, population_size=10, seed=1)
    batch = gen.sample(15)
    gm = generation_metrics(batch, reference_smiles=set(sample_smiles))
    assert gm.validity == 1.0  # BRICS assembly yields sanitizable molecules
    assert 0.0 <= gm.novelty <= 1.0


def test_generation_beats_random_baseline(sample_smiles):
    y = np.linspace(5.0, 8.0, len(sample_smiles))
    ms = MoleculeSet.from_smiles(sample_smiles)
    feat = build_featurizer("ecfp", n_bits=512)
    model = build_model("random_forest").fit(feat.transform(ms), y)
    potency = build_scorer("potency", center=6.5, scale=0.5).bind(model, feat)
    obj = WeightedSumObjective([(potency, 1.0, False)])

    gen = build_generator("graph_ga", seed_smiles=sample_smiles, population_size=15, seed=3)
    random_pot = potency(gen.sample(30))
    optimized = gen.optimize_for_target(obj, budget=120)[:15]
    opt_pot = potency(optimized)
    # Optimized set has strictly higher mean predicted-potency desirability.
    assert np.mean(opt_pot) > np.mean(random_pot)
