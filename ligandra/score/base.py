"""Scoring contract, registry and multi-objective aggregation.

Every scorer maps a :class:`~ligandra.core.molecule.MoleculeSet` to a
normalized desirability in ``[0, 1]`` (1 = best).  Objectives are combined by a
configurable weighted sum, and a Pareto front is available for true
multi-objective ranking.  The aggregated scorer is exactly the objective the
generators (M6) optimize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ligandra.core.molecule import MoleculeSet
from ligandra.core.registry import Registry

#: Registry of scoring functions. Add with ``@SCORERS.register("name")``.
SCORERS: Registry[ScoringFunction] = Registry("scorer")


class ScoringFunction(ABC):
    """Base class: callable returning normalized desirability in [0, 1]."""

    @property
    def name(self) -> str:
        return getattr(self, "registry_name", self.__class__.__name__)

    @abstractmethod
    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        ...


class WeightedSumObjective(ScoringFunction):
    """Aggregate several scorers into one desirability by weighted mean.

    Parameters
    ----------
    terms:
        list of ``(scorer, weight, minimize)``.  ``minimize=True`` flips a
        scorer whose raw desirability should be low (rare; most scorers already
        return higher-is-better).
    """

    def __init__(self, terms: list[tuple[ScoringFunction, float, bool]]) -> None:
        self.terms = terms
        self._total_w = sum(w for _, w, _ in terms) or 1.0

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        if len(mols) == 0:
            return np.zeros(0)
        agg = np.zeros(len(mols), dtype=float)
        for scorer, weight, minimize in self.terms:
            s = np.asarray(scorer(mols), dtype=float)
            if minimize:
                s = 1.0 - s
            agg += weight * s
        return agg / self._total_w

    def breakdown(self, mols: MoleculeSet) -> dict[str, np.ndarray]:
        """Per-scorer desirabilities (for the ranked export table)."""
        return {scorer.name: np.asarray(scorer(mols), dtype=float) for scorer, _, _ in self.terms}


def pareto_front(objective_matrix: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Return indices of the non-dominated rows.

    ``objective_matrix`` is ``(n_samples, n_objectives)`` of desirabilities.
    """
    X = objective_matrix if maximize else -objective_matrix
    n = X.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # j dominates i if j >= i on all and > on at least one
            if np.all(X[j] >= X[i]) and np.any(X[j] > X[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]
