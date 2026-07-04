"""Graph featurizer for message-passing neural networks.

Produces per-molecule atom/bond feature tensors.  The heavy tensor backend
(PyTorch / PyG) is optional; the class registers regardless so the GNN model can
discover it, and raises a clear error if used without the backend installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ligandra.core.molecule import MoleculeSet
from ligandra.featurize.base import FEATURIZERS, Featurizer


@dataclass
class MolGraph:
    """A minimal molecular graph (framework-agnostic)."""

    atom_features: np.ndarray  # (n_atoms, n_atom_feats)
    edge_index: np.ndarray  # (2, n_edges)
    edge_features: np.ndarray  # (n_edges, n_bond_feats)
    n_atoms: int


_ATOM_FEATS = 6


def _atom_vector(atom) -> list[float]:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]


@FEATURIZERS.register("graph")
class GraphFeaturizer(Featurizer):
    """Build :class:`MolGraph` objects for GNN models."""

    def _featurize_one(self, mol) -> np.ndarray:  # not a dense vector
        raise NotImplementedError("Use `transform` to get graph objects.")

    def transform(self, mols: MoleculeSet) -> list[MolGraph]:  # type: ignore[override]
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("RDKit is required for graph featurization.") from exc
        graphs: list[MolGraph] = []
        for m in mols:
            rdmol = m.mol
            if rdmol is None:
                graphs.append(MolGraph(np.zeros((1, _ATOM_FEATS)), np.zeros((2, 0), int), np.zeros((0, 3)), 1))
                continue
            atoms = np.array([_atom_vector(a) for a in rdmol.GetAtoms()], dtype=np.float32)
            src, dst, ef = [], [], []
            for bond in rdmol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = [bond.GetBondTypeAsDouble(), int(bond.GetIsConjugated()), int(bond.IsInRing())]
                src += [i, j]
                dst += [j, i]
                ef += [bf, bf]
            edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
            edge_features = np.array(ef, dtype=np.float32) if ef else np.zeros((0, 3), dtype=np.float32)
            graphs.append(MolGraph(atoms, edge_index, edge_features, rdmol.GetNumAtoms()))
        return graphs
