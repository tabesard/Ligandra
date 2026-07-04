"""Descriptor featurizers: Lipinski (ported), RDKit 2D, Mordred, PaDEL.

The prototype's ``lipinski`` and PaDEL-via-shell path are preserved here as two
interchangeable plugins alongside the full RDKit 2D descriptor set.
"""

from __future__ import annotations

import numpy as np

from ligandra.featurize.base import FEATURIZERS, Featurizer

try:
    from rdkit.Chem import Descriptors, Lipinski

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


@FEATURIZERS.register("lipinski")
class LipinskiFeaturizer(Featurizer):
    """MW, LogP, #H-donors, #H-acceptors — the prototype's descriptor set."""

    @property
    def feature_names(self):
        return ["MW", "LogP", "NumHDonors", "NumHAcceptors"]

    def _featurize_one(self, mol) -> np.ndarray:
        if not _HAS_RDKIT:
            raise ImportError("RDKit is required for Lipinski descriptors.")
        rdmol = mol.mol
        if rdmol is None:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [
                Descriptors.MolWt(rdmol),
                Descriptors.MolLogP(rdmol),
                Lipinski.NumHDonors(rdmol),
                Lipinski.NumHAcceptors(rdmol),
            ],
            dtype=np.float32,
        )


@FEATURIZERS.register("rdkit_descriptors")
class RDKitDescriptorFeaturizer(Featurizer):
    """The full RDKit 2D descriptor block (~200 descriptors)."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._funcs = []
        if _HAS_RDKIT:
            self._names = [name for name, _ in Descriptors.descList]
            self._funcs = [fn for _, fn in Descriptors.descList]

    @property
    def feature_names(self):
        return self._names

    def _featurize_one(self, mol) -> np.ndarray:
        if not _HAS_RDKIT:
            raise ImportError("RDKit is required for RDKit descriptors.")
        rdmol = mol.mol
        if rdmol is None:
            return np.zeros(len(self._funcs), dtype=np.float32)
        vals = np.empty(len(self._funcs), dtype=np.float32)
        for i, fn in enumerate(self._funcs):
            try:
                vals[i] = fn(rdmol)
            except Exception:  # pragma: no cover - a few descriptors can fail
                vals[i] = np.nan
        return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


@FEATURIZERS.register("mordred")
class MordredFeaturizer(Featurizer):
    """Mordred 2D descriptors (optional dependency)."""

    def __init__(self, ignore_3d: bool = True) -> None:
        try:
            from mordred import Calculator, descriptors

            self._calc = Calculator(descriptors, ignore_3D=ignore_3d)
        except ImportError:  # pragma: no cover
            self._calc = None

    def _featurize_one(self, mol) -> np.ndarray:
        if self._calc is None:
            raise ImportError("Install `mordred` to use the mordred featurizer.")
        rdmol = mol.mol
        res = self._calc(rdmol)
        return np.array(list(res.fill_missing(0.0).values()), dtype=np.float32)


@FEATURIZERS.register("padel")
class PaDELFeaturizer(Featurizer):
    """PaDEL PubChem fingerprints via the bundled Java jar (optional plugin).

    Kept for parity with the prototype's ``padel.sh``.  Batches all molecules in
    one Java invocation; requires Java and the PaDEL-Descriptor jar on disk.
    """

    def __init__(self, padel_script: str = "padel.sh", output_csv: str = "descriptors_output.csv") -> None:
        self.padel_script = padel_script
        self.output_csv = output_csv

    def _featurize_one(self, mol) -> np.ndarray:  # pragma: no cover - needs Java
        raise NotImplementedError(
            "PaDEL is a batch featurizer; call `transform(MoleculeSet)` instead."
        )

    def transform(self, mols) -> np.ndarray:  # pragma: no cover - needs Java
        import subprocess

        import pandas as pd

        with open("molecule.smi", "w") as fh:
            for m in mols:
                fh.write(f"{m.canonical_smiles or m.smiles}\n")
        subprocess.run(["bash", self.padel_script], check=True)
        df = pd.read_csv(self.output_csv)
        num = df.select_dtypes(include=[np.number])
        return num.to_numpy(dtype=np.float32)
