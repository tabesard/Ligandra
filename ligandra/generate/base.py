"""Generator contract, registry and validity/novelty metrics.

A generator either samples molecules unconditionally (:meth:`sample`) or
optimizes them against a target-specific objective (:meth:`optimize_for_target`).
Because the objective embeds the target's predictive model, generation is
conditioned on the user's target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ligandra.core.molecule import MoleculeSet
from ligandra.core.registry import Registry
from ligandra.score.base import ScoringFunction

#: Registry of generators. Add with ``@GENERATORS.register("name")``.
GENERATORS: Registry[Generator] = Registry("generator")


class Generator(ABC):
    """Base class for de novo molecular generators."""

    @abstractmethod
    def sample(self, n: int) -> MoleculeSet:
        """Draw ``n`` molecules from the (prior) generative distribution."""

    @abstractmethod
    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """Return molecules optimized to maximize ``objective`` within ``budget``."""


@dataclass
class GenerationMetrics:
    n: int
    validity: float
    uniqueness: float
    novelty: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "validity": self.validity,
            "uniqueness": self.uniqueness,
            "novelty": self.novelty,
        }


def generation_metrics(
    generated: MoleculeSet, reference_smiles: set[str] | None = None
) -> GenerationMetrics:
    """Standard de novo quality metrics on a batch.

    * validity  — fraction parseable by RDKit;
    * uniqueness — fraction of distinct canonical SMILES among valid;
    * novelty   — fraction of valid, unique molecules not in ``reference_smiles``.
    """
    n = len(generated)
    if n == 0:
        return GenerationMetrics(0, 0.0, 0.0, 0.0)
    valid = [m for m in generated if m.is_valid]
    validity = len(valid) / n
    canon = [m.canonical_smiles for m in valid if m.canonical_smiles]
    unique = set(canon)
    uniqueness = (len(unique) / len(canon)) if canon else 0.0
    if reference_smiles is not None and unique:
        novel = [s for s in unique if s not in reference_smiles]
        novelty = len(novel) / len(unique)
    else:
        novelty = 1.0 if unique else 0.0
    return GenerationMetrics(n, validity, uniqueness, novelty)
