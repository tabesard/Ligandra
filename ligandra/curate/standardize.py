"""Structure standardization via RDKit MolStandardize.

Neutralize charges, strip salts/solvents, pick the parent fragment, and (best
effort) canonicalize the tautomer, then return canonical SMILES.  A failure on
any single molecule returns ``None`` rather than raising, so a bad row can be
dropped by the curator.
"""

from __future__ import annotations

from functools import lru_cache

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


@lru_cache(maxsize=1)
def _tautomer_enumerator():
    return rdMolStandardize.TautomerEnumerator()


def standardize_smiles(smiles: str, canonical_tautomer: bool = True) -> str | None:
    """Return standardized canonical SMILES, or ``None`` if it can't be parsed."""
    if not _HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)  # keep largest organic fragment
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:  # pragma: no cover - malformed edge cases
        return None
    if mol is None:
        return None
    if canonical_tautomer:
        # Tautomer canonicalization is expensive and can fail/time out on large
        # molecules; a failure here shouldn't discard an otherwise-clean ligand,
        # so fall back to the uncharged parent rather than returning None.
        try:
            canonical = _tautomer_enumerator().Canonicalize(mol)
            if canonical is not None:
                mol = canonical
        except Exception:  # pragma: no cover - tautomer edge cases
            pass
    return Chem.MolToSmiles(mol)
