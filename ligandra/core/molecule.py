"""Canonical internal molecule objects.

The whole pipeline speaks in :class:`Molecule` / :class:`MoleculeSet`.  A
``Molecule`` lazily parses its RDKit mol, canonicalizes SMILES, can compute
SELFIES / a 3D conformer on demand, and carries a per-feature cache so any
featurizer or model can request what it needs without recomputation.

RDKit is an optional import: the objects construct fine without it (carrying the
raw SMILES string), and only operations that genuinely need chemistry raise a
clear error if RDKit is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

try:  # RDKit is optional at import time
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover - exercised only in bare envs
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    _HAS_RDKIT = False


def has_rdkit() -> bool:
    return _HAS_RDKIT


def _require_rdkit() -> None:
    if not _HAS_RDKIT:
        raise ImportError(
            "RDKit is required for this operation. Install it with "
            "`pip install rdkit`."
        )


class Molecule:
    """A single molecule with lazy chemistry and a feature cache.

    Parameters
    ----------
    smiles:
        Input SMILES.  Stored verbatim; :pyattr:`canonical_smiles` and
        :pyattr:`mol` are computed lazily.
    mol_id:
        Optional external identifier (e.g. a ChEMBL id).
    props:
        Arbitrary per-molecule metadata (target, endpoint value, label, ...).
    """

    __slots__ = ("smiles", "mol_id", "props", "_mol", "_canonical", "_features")

    def __init__(
        self,
        smiles: str,
        mol_id: str | None = None,
        props: dict[str, Any] | None = None,
    ) -> None:
        self.smiles = smiles
        self.mol_id = mol_id
        self.props: dict[str, Any] = dict(props or {})
        self._mol: Any = None
        self._canonical: str | None = None
        self._features: dict[str, np.ndarray] = {}

    # -- chemistry -------------------------------------------------------
    @property
    def mol(self) -> Any:
        """Lazily parsed RDKit mol (``None`` if the SMILES is invalid)."""
        if self._mol is None and _HAS_RDKIT:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol

    @property
    def is_valid(self) -> bool:
        if not _HAS_RDKIT:
            # Best-effort: non-empty string is "valid" without a chem toolkit.
            return bool(self.smiles and self.smiles.strip())
        # An empty SMILES parses to a 0-atom mol (not None); treat it as invalid.
        return self.mol is not None and self.mol.GetNumAtoms() > 0

    @property
    def canonical_smiles(self) -> str | None:
        """RDKit-canonical SMILES, or ``None`` for an invalid molecule."""
        if self._canonical is None and _HAS_RDKIT and self.mol is not None:
            self._canonical = Chem.MolToSmiles(self.mol)
        return self._canonical if _HAS_RDKIT else self.smiles

    def to_selfies(self) -> str:
        """SELFIES encoding (requires the optional ``selfies`` package)."""
        try:
            import selfies as sf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `selfies` to use SELFIES encodings.") from exc
        return sf.encoder(self.canonical_smiles or self.smiles)

    def embed_3d(self, seed: int = 42) -> Any:
        """Generate and return a 3D conformer (ETKDG). Cached on the mol."""
        _require_rdkit()
        mol = self.mol
        if mol is None:
            raise ValueError(f"Cannot embed invalid SMILES: {self.smiles!r}")
        if mol.GetNumConformers() == 0:
            molh = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            AllChem.EmbedMolecule(molh, params)
            try:
                AllChem.MMFFOptimizeMolecule(molh)
            except Exception:  # pragma: no cover - forcefield may be missing
                pass
            self._mol = molh
        return self._mol

    # -- feature cache ---------------------------------------------------
    def get_feature(self, key: str) -> np.ndarray | None:
        return self._features.get(key)

    def set_feature(self, key: str, value: np.ndarray) -> None:
        self._features[key] = value

    def has_feature(self, key: str) -> bool:
        return key in self._features

    # -- dunder ----------------------------------------------------------
    def __repr__(self) -> str:
        ident = self.mol_id or "?"
        return f"Molecule(id={ident!r}, smiles={self.smiles!r})"


class MoleculeSet:
    """An ordered collection of :class:`Molecule` with tabular helpers."""

    def __init__(self, molecules: Sequence[Molecule] | None = None) -> None:
        self._mols: list[Molecule] = list(molecules or [])

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str],
        ids: Iterable[str] | None = None,
        props: Iterable[dict] | None = None,
    ) -> MoleculeSet:
        smiles = list(smiles)
        ids = list(ids) if ids is not None else [None] * len(smiles)
        props = list(props) if props is not None else [None] * len(smiles)
        return cls(
            [Molecule(s, mol_id=i, props=p) for s, i, p in zip(smiles, ids, props)]
        )

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        smiles_col: str = "smiles",
        id_col: str | None = "molecule_id",
    ) -> MoleculeSet:
        mols = []
        for _, row in df.iterrows():
            mol_id = str(row[id_col]) if id_col and id_col in df.columns else None
            props = {k: row[k] for k in df.columns if k not in {smiles_col}}
            mols.append(Molecule(str(row[smiles_col]), mol_id=mol_id, props=props))
        return cls(mols)

    # -- access ----------------------------------------------------------
    @property
    def smiles(self) -> list[str]:
        return [m.smiles for m in self._mols]

    @property
    def valid(self) -> MoleculeSet:
        return MoleculeSet([m for m in self._mols if m.is_valid])

    def prop(self, key: str) -> list[Any]:
        return [m.props.get(key) for m in self._mols]

    def targets_array(self, key: str) -> np.ndarray:
        """Return property ``key`` as a float array (for model ``y``)."""
        return np.asarray([m.props.get(key) for m in self._mols], dtype=float)

    def to_dataframe(self):
        import pandas as pd

        rows = []
        for m in self._mols:
            row = {"molecule_id": m.mol_id, "smiles": m.smiles}
            row.update(m.props)
            rows.append(row)
        return pd.DataFrame(rows)

    # -- dunder ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._mols)

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self._mols)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MoleculeSet(self._mols[idx])
        return self._mols[idx]

    def __add__(self, other: MoleculeSet) -> MoleculeSet:
        return MoleculeSet(self._mols + list(other))

    def __repr__(self) -> str:
        return f"MoleculeSet(n={len(self)})"
