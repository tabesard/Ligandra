"""Data-leakage & applicability guards (Section 7.4).

Because a foundation model's pretraining corpus is enormous, the target
molecules may already appear in it.  These helpers let a run:

* deduplicate the fine-tune/test set against a checkpoint's *known* training
  sources by InChIKey and report the overlap, so reported transfer gains are not
  memorization; and
* flag target molecules that fall far from the modelled distribution
  (applicability domain), so scores on them are treated with suspicion.

We never claim to enumerate the full pretraining set (it lives in the weights).
Overlap is computed against whatever InChIKey allow/deny list the caller can
supply for a given checkpoint; the applicability-domain flag is a standard
kNN-distance measure relative to the fine-tune embeddings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def inchikey(smiles: str) -> str | None:
    """Return the InChIKey for ``smiles`` (needs RDKit); ``None`` on failure."""
    try:
        from rdkit import Chem
    except ImportError:  # pragma: no cover - RDKit optional
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - InChI backend may be missing
        return None


@dataclass
class OverlapReport:
    n_total: int
    n_overlap: int
    overlapping_smiles: list[str]

    @property
    def fraction(self) -> float:
        return self.n_overlap / self.n_total if self.n_total else 0.0

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_overlap": self.n_overlap,
            "fraction": round(self.fraction, 4),
        }


def overlap_with_pretraining(
    smiles: Sequence[str],
    known_inchikeys: Iterable[str],
) -> OverlapReport:
    """Report how many of ``smiles`` collide with ``known_inchikeys``.

    ``known_inchikeys`` is a caller-supplied set of InChIKeys known to be in the
    checkpoint's training sources (e.g. a released manifest).  Molecules whose
    InChIKey is present are flagged as potential leakage.
    """
    known = set(known_inchikeys)
    overlapping: list[str] = []
    for s in smiles:
        k = inchikey(s)
        if k is not None and k in known:
            overlapping.append(s)
    return OverlapReport(
        n_total=len(smiles),
        n_overlap=len(overlapping),
        overlapping_smiles=overlapping,
    )


def dedupe_against(
    smiles: Sequence[str],
    known_inchikeys: Iterable[str],
) -> list[str]:
    """Return the subset of ``smiles`` **not** present in ``known_inchikeys``."""
    known = set(known_inchikeys)
    kept: list[str] = []
    for s in smiles:
        k = inchikey(s)
        if k is None or k not in known:
            kept.append(s)
    return kept


def applicability_domain_flags(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 5,
    z: float = 3.0,
) -> np.ndarray:
    """Boolean flags: ``True`` where a query is *outside* the domain.

    Uses the mean distance to the ``k`` nearest fine-tune embeddings.  A query is
    out-of-domain when that distance exceeds ``mean + z * std`` of the training
    set's own kNN distances — a standard, threshold-free-ish AD criterion.
    """
    train = np.asarray(train_embeddings, dtype=float)
    query = np.asarray(query_embeddings, dtype=float)
    if train.ndim != 2 or query.ndim != 2 or train.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=bool)

    k = max(1, min(k, train.shape[0]))

    def _knn_mean_dist(points: np.ndarray, exclude_self: bool) -> np.ndarray:
        # (n_points, n_train) pairwise Euclidean distances
        d = np.linalg.norm(points[:, None, :] - train[None, :, :], axis=2)
        if exclude_self:
            # drop the nearest (itself, distance 0) by taking k+1 then trimming
            idx = np.argsort(d, axis=1)[:, 1 : k + 1]
        else:
            idx = np.argsort(d, axis=1)[:, :k]
        return np.take_along_axis(d, idx, axis=1).mean(axis=1)

    train_dists = _knn_mean_dist(train, exclude_self=True)
    threshold = train_dists.mean() + z * train_dists.std()
    query_dists = _knn_mean_dist(query, exclude_self=False)
    return query_dists > threshold
