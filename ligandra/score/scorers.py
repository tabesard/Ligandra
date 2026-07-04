"""Concrete scoring functions: potency, QED, SA, drug-likeness, novelty, diversity.

Each returns desirability in ``[0, 1]`` (higher is better).  Potency wraps the
target-specific predictive model (M4/M5), which is what makes generation
conditioned on the user's target.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from ligandra.core.molecule import MoleculeSet
from ligandra.score.base import SCORERS, ScoringFunction

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


# --- Synthetic accessibility (Ertl SA_Score, with a heuristic fallback) ---
def _load_sascorer():
    try:
        from rdkit.Chem import RDConfig

        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer  # type: ignore

        return sascorer
    except Exception:  # pragma: no cover
        return None


_SASCORER = _load_sascorer() if _HAS_RDKIT else None


def synthetic_accessibility(mol) -> float:
    """SA score on the 1 (easy) .. 10 (hard) scale."""
    if _SASCORER is not None:
        try:
            return float(_SASCORER.calculateScore(mol))
        except Exception:  # pragma: no cover
            pass
    # Fallback heuristic: penalise size, rings and stereo complexity.
    n_atoms = mol.GetNumHeavyAtoms()
    n_rings = Descriptors.RingCount(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False))
    raw = 1.0 + 0.02 * n_atoms + 0.4 * n_rings + 0.5 * n_stereo
    return float(min(10.0, raw))


@SCORERS.register("qed")
class QEDScorer(ScoringFunction):
    """Quantitative Estimate of Drug-likeness (already in [0, 1])."""

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        out = np.zeros(len(mols))
        for i, m in enumerate(mols):
            rd = m.mol
            out[i] = QED.qed(rd) if rd is not None else 0.0
        return out


@SCORERS.register("sa")
class SAScorer(ScoringFunction):
    """Synthetic accessibility desirability = (10 - SA) / 9."""

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        out = np.zeros(len(mols))
        for i, m in enumerate(mols):
            rd = m.mol
            if rd is None:
                out[i] = 0.0
            else:
                out[i] = np.clip((10.0 - synthetic_accessibility(rd)) / 9.0, 0.0, 1.0)
        return out


@SCORERS.register("druglikeness")
class DrugLikenessScorer(ScoringFunction):
    """Fraction of Lipinski Ro5 + Veber rules satisfied."""

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        out = np.zeros(len(mols))
        for i, m in enumerate(mols):
            rd = m.mol
            if rd is None:
                continue
            mw = Descriptors.MolWt(rd)
            logp = Crippen.MolLogP(rd)
            hbd = Lipinski.NumHDonors(rd)
            hba = Lipinski.NumHAcceptors(rd)
            tpsa = Descriptors.TPSA(rd)
            rot = Lipinski.NumRotatableBonds(rd)
            rules = [mw <= 500, logp <= 5, hbd <= 5, hba <= 10, tpsa <= 140, rot <= 10]
            out[i] = sum(rules) / len(rules)
        return out


@SCORERS.register("novelty")
class NoveltyScorer(ScoringFunction):
    """1 - max Tanimoto similarity to a reference set (e.g. training actives)."""

    def __init__(self, reference_smiles: list[str] | None = None, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self._ref_fps = self._fingerprints(reference_smiles or [])

    def _fingerprints(self, smiles: list[str]):
        if not _HAS_RDKIT:
            return []
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        fps = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fps.append(gen.GetFingerprint(m))
        return fps

    def set_reference(self, smiles: list[str]) -> None:
        self._ref_fps = self._fingerprints(smiles)

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import BulkTanimotoSimilarity

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        out = np.ones(len(mols))
        if not self._ref_fps:
            return out  # everything is "novel" without a reference
        for i, m in enumerate(mols):
            rd = m.mol
            if rd is None:
                out[i] = 0.0
                continue
            sims = BulkTanimotoSimilarity(gen.GetFingerprint(rd), self._ref_fps)
            out[i] = 1.0 - (max(sims) if sims else 0.0)
        return out


@SCORERS.register("potency")
class PotencyScorer(ScoringFunction):
    """Predicted potency from the target model, squashed into [0, 1].

    Bound at runtime by the pipeline via :meth:`bind`.  ``center``/``scale``
    define the logistic mapping from predicted p-value to desirability.
    """

    def __init__(self, center: float = 6.0, scale: float = 1.0) -> None:
        self.center = center
        self.scale = scale
        self._model = None
        self._featurizer = None

    def bind(self, model, featurizer) -> PotencyScorer:
        self._model = model
        self._featurizer = featurizer
        return self

    def __call__(self, mols: MoleculeSet) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PotencyScorer must be bound to a model via .bind().")
        valid = mols.valid
        if len(valid) == 0:
            return np.zeros(len(mols))
        if getattr(self._model, "consumes_smiles", False):
            # SMILES-consuming models (e.g. the foundation model) are their own
            # featurizer — the tokenizer is the featurizer — so none is required.
            X = [m.canonical_smiles or m.smiles for m in valid]
        else:
            if self._featurizer is None:
                raise RuntimeError(
                    "PotencyScorer needs a featurizer for this model; bind one via "
                    ".bind(model, featurizer)."
                )
            X = self._featurizer.transform(valid)
        preds = np.asarray(self._model.predict(X), dtype=float)
        desir = 1.0 / (1.0 + np.exp(-(preds - self.center) / self.scale))
        # Map back onto the full set (invalids -> 0).
        out = np.zeros(len(mols))
        vi = 0
        for i, m in enumerate(mols):
            if m.is_valid:
                out[i] = desir[vi]
                vi += 1
        return out
