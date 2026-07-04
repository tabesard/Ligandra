"""Fragment-recombination genetic-algorithm generator (the working default).

A dependency-light, CPU-only generator that actually optimizes against the
target objective, so the acceptance criterion "predicted potency beats a
random-sampling baseline" is demonstrable without a GPU.

It seeds a fragment library from the training actives (BRICS decomposition),
then evolves a population by streaming new molecules out of a BRICS builder
biased toward the fragments of the current elite.  Every build call is bounded
by a wall-clock budget and a maximum molecule size, so generation can never
hang.  Registered as ``graph_ga``.
"""

from __future__ import annotations

import random
import time

from ligandra.core.molecule import Molecule, MoleculeSet
from ligandra.generate.base import GENERATORS, Generator
from ligandra.score.base import ScoringFunction

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


@GENERATORS.register("graph_ga")
class GeneticGenerator(Generator):
    """Fragment-recombination genetic algorithm (BRICS-based)."""

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        population_size: int = 50,
        max_depth: int = 2,
        max_atoms: int = 50,
        build_time_budget: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.max_depth = max_depth
        self.max_atoms = max_atoms
        self.build_time_budget = build_time_budget
        self._rng = random.Random(seed)
        self.seed_smiles: list[str] = []
        self._fragments: list[str] = []
        if seed_smiles:
            self.set_seeds(seed_smiles)

    # -- fragments -------------------------------------------------------
    def set_seeds(self, smiles: list[str]) -> None:
        self.seed_smiles = list(dict.fromkeys(smiles))
        self._fragments = self._harvest_fragments(self.seed_smiles)

    def _harvest_fragments(self, smiles: list[str]) -> list[str]:
        if not _HAS_RDKIT:
            return []
        frags: set[str] = set()
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                frags.update(BRICS.BRICSDecompose(m))
            except Exception:
                continue
        return list(frags)

    def _build_batch(self, fragment_smiles: list[str], n: int) -> list[str]:
        """Stream up to ``n`` unique molecules from a BRICS builder (bounded)."""
        if not _HAS_RDKIT or len(fragment_smiles) < 2:
            return []
        frag_mols = [Chem.MolFromSmiles(f) for f in fragment_smiles]
        frag_mols = [m for m in frag_mols if m is not None]
        if len(frag_mols) < 2:
            return []
        self._rng.shuffle(frag_mols)
        builder = BRICS.BRICSBuild(frag_mols, scrambleReagents=True, maxDepth=self.max_depth)
        out: list[str] = []
        seen: set[str] = set()
        start = time.time()
        try:
            for product in builder:
                if time.time() - start > self.build_time_budget:
                    break
                try:
                    Chem.SanitizeMol(product)
                except Exception:
                    continue
                if product.GetNumHeavyAtoms() > self.max_atoms:
                    continue
                smi = Chem.MolToSmiles(product)
                if smi in seen:
                    continue
                seen.add(smi)
                out.append(smi)
                if len(out) >= n:
                    break
        except Exception:
            pass
        return out

    # -- Generator API ---------------------------------------------------
    def sample(self, n: int) -> MoleculeSet:
        """Random fragment recombinations (the generative prior / baseline)."""
        if not self.seed_smiles:
            raise ValueError("GeneticGenerator needs seed molecules; call set_seeds().")
        smis = self._build_batch(self._fragments, n)
        return MoleculeSet([Molecule(s) for s in smis])

    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """Evolve the population to maximize ``objective`` within ``budget`` evals."""
        if not self.seed_smiles:
            raise ValueError("GeneticGenerator needs seed molecules; call set_seeds().")

        population = list(self.seed_smiles)[: self.population_size]
        scores = objective(MoleculeSet.from_smiles(population))
        scored: dict[str, float] = {s: float(v) for s, v in zip(population, scores)}
        evals = len(population)

        while evals < budget:
            elite = [
                s
                for s, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[
                    : self.population_size
                ]
            ]
            elite_frags = self._harvest_fragments(elite) or self._fragments
            n_new = min(self.population_size, budget - evals)
            children = [c for c in self._build_batch(elite_frags, n_new * 2) if c not in scored]
            if not children:
                # Diversify from the full seed fragment pool.
                children = [
                    c for c in self._build_batch(self._fragments, n_new * 2) if c not in scored
                ]
            if not children:
                break
            children = children[:n_new]
            child_scores = objective(MoleculeSet.from_smiles(children))
            for s, sc in zip(children, child_scores):
                scored[s] = float(sc)
            evals += len(children)

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return MoleculeSet([Molecule(s, props={"objective_score": sc}) for s, sc in ranked])
