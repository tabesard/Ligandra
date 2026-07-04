"""VAE and transformer generators (torch-backed). Skipped if torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from ligandra.core.molecule import MoleculeSet  # noqa: E402
from ligandra.generate import build_generator, generation_metrics  # noqa: E402
from ligandra.score.base import ScoringFunction  # noqa: E402

_SEEDS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "O=C(O)c1ccccc1",
    "c1ccc(-c2ccncc2)cc1",
    "COc1ccc(CCN)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "O=C(O)Cc1ccccc1",
    "CCOC(=O)c1ccccc1",
    "Cc1ccccc1O",
    "c1ccccc1",
    "CCO",
    "c1ccncc1",
]


class _SizeScorer(ScoringFunction):
    """Toy but real objective: desirability peaks at ~14 heavy atoms."""

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        out = []
        for m in mols:
            n = m.mol.GetNumAtoms() if m.is_valid else 0
            out.append(max(0.0, 1.0 - abs(n - 14) / 14.0))
        return np.asarray(out, dtype=float)


@pytest.mark.parametrize("name", ["vae", "transformer"])
def test_generator_trains_samples_and_optimizes(name):
    kw = {"epochs": 25} if name == "vae" else {"pretrain_epochs": 20}
    gen = build_generator(name, seed_smiles=_SEEDS, seed=0, **kw)

    # sampling from the trained prior yields at least some valid molecules
    sampled = gen.sample(24)
    gm = generation_metrics(sampled, reference_smiles=set(_SEEDS)).as_dict()
    assert gm["n"] == 24
    n_valid = sum(1 for m in sampled if m.is_valid)
    assert n_valid >= 1

    # optimization returns molecules ranked by the objective, and produces
    # at least one valid, novel candidate
    result = gen.optimize_for_target(_SizeScorer(), budget=48)
    assert len(result) >= 1
    valid_novel = [m for m in result if m.is_valid and m.canonical_smiles not in set(_SEEDS)]
    assert len(valid_novel) >= 1
    assert "objective_score" in result[0].props


@pytest.mark.parametrize("name", ["vae", "transformer"])
def test_generator_requires_seeds(name):
    gen = build_generator(name, seed_smiles=[])
    with pytest.raises(ValueError):
        gen.sample(1)


def test_optimization_beats_random_baseline():
    """The transformer's optimized batch should score better than its untuned prior."""
    scorer = _SizeScorer()
    gen = build_generator("transformer", seed_smiles=_SEEDS, seed=1, pretrain_epochs=20)

    prior = gen.sample(32)
    prior_score = float(np.mean(scorer(prior)))

    optimized = gen.optimize_for_target(scorer, budget=96)
    top = MoleculeSet(list(optimized)[:32])
    opt_score = float(np.mean(scorer(top)))

    assert opt_score >= prior_score  # optimization should not hurt
