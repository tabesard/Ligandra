"""Export a ranked candidate set to CSV or SDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_candidates(df: pd.DataFrame, path: str | Path, fmt: str | None = None) -> Path:
    """Write candidates to ``path``. Format inferred from extension if ``fmt`` is None."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
        return path
    if fmt in {"sdf", "sd"}:
        _write_sdf(df, path)
        return path
    raise ValueError(f"Unsupported export format: {fmt!r} (use csv or sdf).")


def _write_sdf(df: pd.DataFrame, path: Path) -> None:
    from rdkit import Chem

    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    writer = Chem.SDWriter(str(path))
    try:
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(str(row[smiles_col]))
            if mol is None:
                continue
            for col, val in row.items():
                if col == smiles_col:
                    continue
                mol.SetProp(str(col), str(val))
            writer.write(mol)
    finally:
        writer.close()
