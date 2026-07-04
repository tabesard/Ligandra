"""Unit handling and pXC50 conversion — generalized beyond IC50.

The prototype assumed nM and IC50.  Real bioactivity tables mix nM/uM/pM/M and
several endpoints.  These helpers convert any concentration to molar and compute
``pXC50 = -log10(value_in_molar)`` for any concentration endpoint.
"""

from __future__ import annotations

import math

import numpy as np

#: Multiplier to convert a value in the given (molar) unit into molar (mol/L).
_TO_MOLAR: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "um": 1e-6,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

#: Multiplier to convert a mass-concentration unit into grams per litre.  These
#: need a molecular weight to reach molar (real ChEMBL tables mix them in).
_TO_G_PER_L: dict[str, float] = {
    "g.L-1": 1.0, "g/L": 1.0,
    "mg.mL-1": 1.0, "mg/mL": 1.0,  # mg/mL == g/L
    "mg.L-1": 1e-3, "mg/L": 1e-3,
    "ug.mL-1": 1e-3, "ug/mL": 1e-3, "µg.mL-1": 1e-3, "µg/mL": 1e-3,
    "ug.L-1": 1e-6, "ug/L": 1e-6,
    "ng.mL-1": 1e-6, "ng/mL": 1e-6,
    "ng.L-1": 1e-9,
    "pg.mL-1": 1e-9,
}


def to_molar(value: float, units: str, mol_weight: float | None = None) -> float:
    """Convert a concentration ``value`` in ``units`` to molar (mol/L).

    Molar units (``nM``/``uM``/...) convert directly.  Mass-concentration units
    (``ug.mL-1``/``mg/mL``/...) require ``mol_weight`` (g/mol) to convert.

    Raises
    ------
    KeyError
        If ``units`` is not a recognised concentration unit, or a mass unit is
        given without a molecular weight.
    """
    key = units.strip() if isinstance(units, str) else units
    if key in _TO_MOLAR:
        return float(value) * _TO_MOLAR[key]
    if key in _TO_G_PER_L:
        if mol_weight is None or not (float(mol_weight) > 0):
            raise KeyError(
                f"Mass-concentration unit {units!r} needs a molecular weight to convert."
            )
        return float(value) * _TO_G_PER_L[key] / float(mol_weight)
    raise KeyError(f"Unrecognised concentration unit: {units!r}")


def pxc50(value: float, units: str = "nM", mol_weight: float | None = None) -> float:
    """``-log10`` of a concentration expressed in molar.

    ``value`` must be > 0.  Mirrors the classic pIC50 transform but works for
    Ki/Kd/EC50 as well, and for mass-concentration units given ``mol_weight``.
    """
    molar = to_molar(value, units, mol_weight=mol_weight)
    if molar <= 0:
        raise ValueError(f"Non-positive concentration cannot be log-transformed: {value} {units}")
    return -math.log10(molar)


def pxc50_array(values, units: str = "nM") -> np.ndarray:
    """Vectorised :func:`pxc50` returning NaN for non-positive/invalid inputs."""
    out = np.full(len(values), np.nan, dtype=float)
    mult = _TO_MOLAR.get(units.strip() if isinstance(units, str) else units)
    if mult is None:
        raise KeyError(f"Unrecognised concentration unit: {units!r}")
    for i, v in enumerate(values):
        try:
            molar = float(v) * mult
            if molar > 0:
                out[i] = -math.log10(molar)
        except (TypeError, ValueError):
            continue
    return out
