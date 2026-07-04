"""Train/val/test splitting.

Scaffold (Bemis–Murcko) splitting groups molecules by their core scaffold and
assigns whole scaffolds to a split, so no scaffold leaks across train/test.
This gives an honest generalization estimate for drug design, where random
splits are optimistic.  A random split is also provided.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


@dataclass(frozen=True)
class Split:
    """Index arrays for the three partitions."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def bemis_murcko_scaffold(smiles: str) -> str:
    """Generic Bemis–Murcko scaffold SMILES ('' if it can't be computed)."""
    if not _HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:  # pragma: no cover
        return ""


def random_split(
    n: int, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return Split(train=np.sort(train), val=np.sort(val), test=np.sort(test))


def scaffold_split(
    smiles: list[str], test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Split:
    """Deterministic scaffold split.

    Scaffolds are sorted largest-group-first (the common deterministic recipe),
    then greedily filled into test, val, train buckets up to their target sizes.
    """
    n = len(smiles)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[bemis_murcko_scaffold(smi)].append(i)

    # Largest scaffolds first for a stable, reproducible assignment.
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    test: list[int] = []
    val: list[int] = []
    train: list[int] = []
    for group in ordered:
        if len(test) + len(group) <= n_test:
            test.extend(group)
        elif len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            train.extend(group)

    return Split(
        train=np.sort(np.asarray(train, dtype=int)),
        val=np.sort(np.asarray(val, dtype=int)),
        test=np.sort(np.asarray(test, dtype=int)),
    )


def make_split(strategy: str, smiles: list[str], **kwargs) -> Split:
    """Dispatch on split strategy name ('random' | 'scaffold')."""
    if strategy == "scaffold":
        return scaffold_split(smiles, **kwargs)
    if strategy == "random":
        return random_split(len(smiles), **kwargs)
    raise ValueError(f"Unknown split strategy: {strategy!r}")
