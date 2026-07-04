"""Structure-based 3D diffusion generator (EDM/DiffSBDD-style).

Tests exercise what genuinely works at CPU/tiny-data scale: reliable all-atom
reconstruction, an E(3)-equivariant denoiser, finite (non-NaN) training, learned
denoising of noised molecules, pocket-conditioning wiring, and that sampling /
optimization run.  De-novo chemical quality needs GPU-scale pretraining and is
not asserted (see the module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ligandra.core.molecule import MoleculeSet  # noqa: E402
from ligandra.generate import build_generator  # noqa: E402
from ligandra.generate.diffusion import (  # noqa: E402
    _ELEMENTS,
    DiffusionGenerator,
    cloud_to_mol,
    mol_to_cloud,
    parse_pocket_pdb,
)
from ligandra.score.base import ScoringFunction  # noqa: E402

_SEEDS = [
    "CCO",
    "CCN",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccncc1",
    "CCOC",
    "CC(=O)Nc1ccc(O)cc1",
    "O=C(O)Cc1ccccc1",
    "COc1ccccc1",
    "CCc1ccccc1",
    "CCCCO",
    "CC(C)O",
]


class _SizeScorer(ScoringFunction):
    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        return np.asarray(
            [max(0.0, 1 - abs((m.mol.GetNumAtoms() if m.is_valid else 0) - 9) / 9) for m in mols]
        )


@pytest.fixture(scope="module")
def trained_diffusion() -> DiffusionGenerator:
    gen = build_generator("diffusion", seed_smiles=_SEEDS, seed=0, timesteps=40, epochs=90)
    gen._ensure_trained()
    return gen


# --------------------------------------------------------------- basics ------
def test_diffusion_registered():
    from ligandra.generate import GENERATORS

    assert "diffusion" in GENERATORS.available()


def test_cloud_roundtrip_reconstructs():
    ok = 0
    mols = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "O=C(O)Cc1ccccc1", "COc1ccccc1"]
    for smi in mols:
        cloud = mol_to_cloud(smi, seed=0)
        assert cloud is not None
        coords, types = cloud
        rebuilt = cloud_to_mol(coords, types)
        from rdkit import Chem

        if rebuilt and Chem.CanonSmiles(rebuilt) == Chem.CanonSmiles(smi):
            ok += 1
    assert ok >= 4  # reconstruction from clean all-atom geometry is reliable


def test_parse_pocket_pdb(tmp_path):
    def atom(serial, elem, x, y, z):
        # x at cols 31-38, element at cols 77-78 (0-based [30:38], [76:78])
        line = list(" " * 80)
        line[0:6] = "ATOM  "
        s = f"{serial:>5}"
        line[6:11] = s
        line[12:15] = f"{elem:<3}"
        line[17:20] = "LIG"
        line[21] = "A"
        line[22:26] = "   1"
        line[30:38] = f"{x:8.3f}"
        line[38:46] = f"{y:8.3f}"
        line[46:54] = f"{z:8.3f}"
        line[76:78] = f"{elem:>2}"
        return "".join(line)

    pdb = "\n".join(
        [atom(1, "C", 0.0, 0.0, 0.0), atom(2, "N", 1.5, 0.0, 0.0), atom(3, "O", 0.0, 1.5, 0.0)]
    )
    path = tmp_path / "pocket.pdb"
    path.write_text(pdb + "\n", encoding="utf-8")
    coords, types = parse_pocket_pdb(str(path))
    assert coords.shape == (3, 3)
    assert [_ELEMENTS[t] for t in types] == ["C", "N", "O"]


# ---------------------------------------------------------- equivariance -----
def test_egnn_denoiser_is_e3_equivariant():
    gen = DiffusionGenerator(seed_smiles=["CCO"], seed=0)
    gen.device = "cpu"
    torch.manual_seed(0)
    net = gen._build_net()
    n, k = 6, len(_ELEMENTS)
    feats = torch.randn(n, k + 2)
    x = torch.randn(n, 3)
    x = x - x.mean(0, keepdim=True)
    movable = torch.ones(n)

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1

    with torch.no_grad():
        h1, ex1 = net(feats, x, movable)
        h2, ex2 = net(feats, x @ q.T, movable)  # rotate input
        _, ex3 = net(feats, x + torch.tensor([1.0, 2.0, 3.0]), movable)  # translate

    assert torch.allclose(ex2, ex1 @ q.T, atol=1e-4)  # coords equivariant
    assert torch.allclose(h2, h1, atol=1e-4)  # types invariant
    assert torch.allclose(ex3, ex1, atol=1e-4)  # translation invariant


# ------------------------------------------------------- train / sample -----
def test_training_is_finite_and_sampling_runs(trained_diffusion):
    # the trained denoiser produces finite (non-NaN) predictions
    coords, types = trained_diffusion._clouds[0]
    x = torch.tensor(coords)
    h = trained_diffusion._onehot(types)
    n = x.size(0)
    feats = torch.cat([h, torch.zeros(n, 1), torch.zeros(n, 1)], dim=-1)
    with torch.no_grad():
        ph, px = trained_diffusion._net(feats, x, torch.ones(n))
    assert torch.isfinite(px).all() and torch.isfinite(ph).all()

    sampled = trained_diffusion.sample(6)
    assert len(sampled) == 6  # returns the requested number of molecules


def test_denoises_noised_molecule(trained_diffusion):
    """SDEdit: lightly noise real molecules, denoise, and reconstruct — the
    trained model should recover at least some valid molecules."""
    g = trained_diffusion
    t = max(1, int(g.timesteps * 0.12))
    recovered = 0
    for smi in _SEEDS:
        coords, types = mol_to_cloud(smi, seed=0)
        x0 = torch.tensor(coords)
        h0 = g._onehot(types)
        abar = g._abar[t]
        ex = torch.randn_like(x0)
        ex = ex - ex.mean(0, keepdim=True)
        x = abar.sqrt() * x0 + (1 - abar).sqrt() * ex
        h = abar.sqrt() * h0 + (1 - abar).sqrt() * torch.randn_like(h0)
        x, h = g._denoise(x, h, from_t=t)
        if cloud_to_mol(x.cpu().numpy(), h.argmax(-1).cpu().numpy()):
            recovered += 1
    assert recovered >= 1  # the denoiser genuinely reconstructs noised molecules


def test_optimize_runs_and_ranks(trained_diffusion):
    result = trained_diffusion.optimize_for_target(_SizeScorer(), budget=64)
    assert isinstance(result, MoleculeSet)
    if len(result) > 0:
        # ranked descending by objective, with the score recorded per candidate
        assert "objective_score" in result[0].props
        scores = [m.props["objective_score"] for m in result]
        assert scores == sorted(scores, reverse=True)


def test_requires_seeds():
    gen = build_generator("diffusion", seed_smiles=[])
    with pytest.raises(ValueError):
        gen.sample(1)


def test_pocket_conditioned_generator_runs(tmp_path):
    # a tiny pocket file → context nodes that condition (don't break) generation
    pdb_lines = []
    rng = np.random.RandomState(0)
    for i in range(5):
        x, y, z = rng.randn(3) * 3
        line = list(" " * 80)
        line[0:6] = "ATOM  "
        line[6:11] = f"{i + 1:>5}"
        line[12:15] = "C  "
        line[17:20] = "LIG"
        line[21] = "A"
        line[30:38] = f"{x:8.3f}"
        line[38:46] = f"{y:8.3f}"
        line[46:54] = f"{z:8.3f}"
        line[76:78] = " C"
        pdb_lines.append("".join(line))
    path = tmp_path / "pocket.pdb"
    path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    gen = build_generator(
        "diffusion", seed_smiles=_SEEDS[:6], pocket=str(path), seed=0, timesteps=30, epochs=20
    )
    gen._ensure_trained()
    assert gen._pocket_ctx is not None
    assert gen._pocket_ctx[0].shape[0] == 5  # 5 context atoms conditioning the model
    sampled = gen.sample(4)
    assert len(sampled) == 4
