"""Featurizer contract and registry.

A featurizer turns a :class:`~ligandra.core.molecule.MoleculeSet` into a dense
matrix (descriptors / fingerprints), a graph batch (GNNs) or an embedding tensor
(learned).  Vector featurizers cache their output on each molecule so repeated
requests are free and different models can share the same features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ligandra.core.molecule import MoleculeSet
from ligandra.core.registry import Registry

#: Registry of featurizers. Add one with ``@FEATURIZERS.register("name")``.
FEATURIZERS: Registry[Featurizer] = Registry("featurizer")


class Featurizer(ABC):
    """Base class for all featurizers."""

    #: cache key on the Molecule; defaults to the registered name
    @property
    def cache_key(self) -> str:
        return getattr(self, "registry_name", self.__class__.__name__)

    @property
    def feature_names(self) -> list[str] | None:
        """Optional column names (for descriptor interpretability)."""
        return None

    @abstractmethod
    def _featurize_one(self, mol) -> np.ndarray:
        """Featurize a single :class:`~ligandra.core.molecule.Molecule`."""

    def transform(self, mols: MoleculeSet) -> np.ndarray:
        """Featurize a whole set into a 2D array, using the per-molecule cache."""
        rows = []
        key = self.cache_key
        for m in mols:
            cached = m.get_feature(key)
            if cached is None:
                cached = self._featurize_one(m)
                m.set_feature(key, cached)
            rows.append(cached)
        return np.vstack(rows) if rows else np.empty((0, 0))
