"""Fingerprint featurizers: ECFP/Morgan, MACCS, RDKit path FP."""

from __future__ import annotations

import numpy as np

from ligandra.featurize.base import FEATURIZERS, Featurizer

try:
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


def _require():
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for fingerprint featurizers.")


@FEATURIZERS.register("ecfp")
class ECFPFeaturizer(Featurizer):
    """Extended-connectivity (Morgan) fingerprint as a bit vector."""

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self._gen = None
        if _HAS_RDKIT:
            self._gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=n_bits
            )

    def _featurize_one(self, mol) -> np.ndarray:
        _require()
        rdmol = mol.mol
        if rdmol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = self._gen.GetFingerprint(rdmol)
        arr = np.zeros(self.n_bits, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        return arr


@FEATURIZERS.register("maccs")
class MACCSFeaturizer(Featurizer):
    """166-bit MACCS structural keys."""

    def _featurize_one(self, mol) -> np.ndarray:
        _require()
        rdmol = mol.mol
        if rdmol is None:
            return np.zeros(167, dtype=np.float32)
        fp = MACCSkeys.GenMACCSKeys(rdmol)
        arr = np.zeros(167, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        return arr


@FEATURIZERS.register("rdkit_fp")
class RDKitFPFeaturizer(Featurizer):
    """RDKit topological (path) fingerprint."""

    def __init__(self, n_bits: int = 2048) -> None:
        self.n_bits = n_bits
        self._gen = None
        if _HAS_RDKIT:
            self._gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)

    def _featurize_one(self, mol) -> np.ndarray:
        _require()
        rdmol = mol.mol
        if rdmol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = self._gen.GetFingerprint(rdmol)
        arr = np.zeros(self.n_bits, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        return arr
