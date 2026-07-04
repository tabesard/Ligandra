"""End-to-end curation of a normalized activities table.

Reproduces the prototype's science (drop NaNs, dedupe SMILES, label
active/inactive/intermediate, concentration -> p-value) but generalized:
- any concentration endpoint (IC50/EC50/Ki/Kd), correct per-row unit handling;
- RDKit structure standardization;
- duplicate SMILES aggregated by median (not just first-wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ligandra.config.schema import CurationConfig
from ligandra.core.types import ActivityLabel, Column, Endpoint
from ligandra.curate.standardize import standardize_smiles
from ligandra.curate.units import to_molar


@dataclass
class CurationReport:
    """Bookkeeping of what curation did (logged to the run manifest)."""

    n_input: int = 0
    n_missing_dropped: int = 0
    n_invalid_dropped: int = 0
    n_duplicates_merged: int = 0
    n_output: int = 0
    label_counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_missing_dropped": self.n_missing_dropped,
            "n_invalid_dropped": self.n_invalid_dropped,
            "n_duplicates_merged": self.n_duplicates_merged,
            "n_output": self.n_output,
            "label_counts": self.label_counts,
        }


def label_activity(value_nm: float, active: float, inactive: float) -> str:
    """Threshold labeling on a nM scale (configurable cutoffs)."""
    if value_nm is None or (isinstance(value_nm, float) and np.isnan(value_nm)):
        return ActivityLabel.UNDEFINED.value
    if value_nm <= active:
        return ActivityLabel.ACTIVE.value
    if value_nm >= inactive:
        return ActivityLabel.INACTIVE.value
    return ActivityLabel.INTERMEDIATE.value


def curate(
    df: pd.DataFrame,
    endpoint: Endpoint | str,
    config: CurationConfig | None = None,
) -> tuple[pd.DataFrame, CurationReport]:
    """Curate a normalized activities frame.

    Returns the curated DataFrame plus a :class:`CurationReport`.  Output columns:
    ``molecule_id, smiles, value, units, endpoint, <p_name>, label`` where
    ``<p_name>`` is ``pIC50``/``pKi``/... for concentration endpoints.
    """
    config = config or CurationConfig()
    endpoint = Endpoint(endpoint) if not isinstance(endpoint, Endpoint) else endpoint
    report = CurationReport(n_input=len(df))
    df = df.copy()

    # 1) Drop rows with missing SMILES or value.
    before = len(df)
    df = df[df[Column.SMILES].notna() & df[Column.VALUE].notna()]
    df = df[df[Column.SMILES].astype(str).str.strip() != ""]
    report.n_missing_dropped = before - len(df)

    # 2) Numeric coercion of the value.
    df[Column.VALUE] = pd.to_numeric(df[Column.VALUE], errors="coerce")
    df = df[df[Column.VALUE].notna()]

    # 3) Units: fill blanks with the configured default.
    if Column.UNITS not in df.columns:
        df[Column.UNITS] = config.default_units
    df[Column.UNITS] = df[Column.UNITS].fillna(config.default_units).replace("", config.default_units)

    # 4) Structure standardization.
    if config.standardize:
        std = df[Column.SMILES].astype(str).map(lambda s: standardize_smiles(s))
        before = len(df)
        df = df.assign(**{Column.SMILES: std})
        df = df[df[Column.SMILES].notna()]
        report.n_invalid_dropped = before - len(df)

    if df.empty:
        report.n_output = 0
        return _empty_curated(endpoint), report

    # 5) Concentration -> p-value (nM canonicalized for labeling too).
    #    Uses each molecule's weight so mass-concentration units (ug.mL-1 etc.)
    #    convert too; any row whose unit can't be converted becomes NaN and is
    #    dropped rather than crashing the run.
    p_name = endpoint.p_name
    if endpoint.is_concentration:
        mws = df[Column.SMILES].map(_mol_weight)
        molar = np.array(
            [
                _safe_to_molar(v, u, mw)
                for v, u, mw in zip(df[Column.VALUE], df[Column.UNITS], mws)
            ],
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            df[p_name] = np.where(molar > 0, -np.log10(molar), np.nan)
        df["_value_nm"] = molar * 1e9
        df = df[df[p_name].notna()]
    else:
        # Percent-style endpoint: use the value directly.
        df[p_name] = df[Column.VALUE].astype(float)
        df["_value_nm"] = np.nan

    # 6) Deduplicate on canonical SMILES, aggregating the p-value by median.
    before = len(df)
    agg = {
        p_name: "median",
        "_value_nm": "median",
        Column.MOLECULE_ID: "first",
        Column.UNITS: "first",
    }
    grouped = df.groupby(Column.SMILES, as_index=False).agg(agg)
    report.n_duplicates_merged = before - len(grouped)
    df = grouped

    # 7) Labeling (only meaningful for concentration endpoints).
    if endpoint.is_concentration:
        df["label"] = [
            label_activity(v, config.active_threshold, config.inactive_threshold)
            for v in df["_value_nm"]
        ]
    else:
        df["label"] = ActivityLabel.UNDEFINED.value

    df[Column.ENDPOINT] = endpoint.value
    df = df.rename(columns={"_value_nm": Column.VALUE})
    out_cols = [Column.MOLECULE_ID, Column.SMILES, Column.VALUE, Column.UNITS, Column.ENDPOINT, p_name, "label"]
    df = df[out_cols].reset_index(drop=True)

    report.n_output = len(df)
    report.label_counts = df["label"].value_counts().to_dict()
    return df, report


def _mol_weight(smiles: str) -> float:
    """Molecular weight (g/mol) from a SMILES, or NaN if it can't be parsed."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:  # pragma: no cover - RDKit optional
        return float("nan")
    mol = Chem.MolFromSmiles(str(smiles))
    return float(Descriptors.MolWt(mol)) if mol is not None else float("nan")


def _safe_to_molar(value, units, mol_weight) -> float:
    """Convert to molar; return NaN (drop the row) for unconvertible units."""
    try:
        return to_molar(float(value), units, mol_weight=mol_weight)
    except (KeyError, ValueError, TypeError):
        return float("nan")


def _empty_curated(endpoint: Endpoint) -> pd.DataFrame:
    cols = [Column.MOLECULE_ID, Column.SMILES, Column.VALUE, Column.UNITS, Column.ENDPOINT, endpoint.p_name, "label"]
    return pd.DataFrame(columns=cols)
