"""Local file data source: user-provided CSV or SDF.

Lets a scientist bring their own bioactivity table without any network access —
essential for offline / reproducible runs and for the integration test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ligandra.core.types import Column, Target
from ligandra.data.base import DATA_SOURCES, DataSource


@DATA_SOURCES.register("local")
class LocalSource(DataSource):
    """Read activities from a CSV/SDF and normalize the schema.

    Parameters
    ----------
    path:
        CSV or SDF file.
    smiles_col, value_col, id_col:
        Column names in the CSV to map onto the normalized schema.
    endpoint, units, target_id:
        Constant metadata applied to every row (a local file is usually one
        target/endpoint already).
    """

    def __init__(
        self,
        path: str,
        smiles_col: str = "smiles",
        value_col: str = "value",
        id_col: str | None = None,
        endpoint: str = "IC50",
        units: str = "nM",
        target_id: str = "LOCAL",
    ) -> None:
        self.path = Path(path)
        self.smiles_col = smiles_col
        self.value_col = value_col
        self.id_col = id_col
        self.endpoint = endpoint
        self.units = units
        self.target_id = target_id

    def search_targets(self, query: str) -> list[Target]:
        return [Target(target_id=self.target_id, name=self.target_id, source="local")]

    def _read(self) -> pd.DataFrame:
        if self.path.suffix.lower() in {".sdf", ".sd"}:
            return self._read_sdf()
        return pd.read_csv(self.path)

    def _read_sdf(self) -> pd.DataFrame:
        from rdkit import Chem

        rows = []
        supplier = Chem.SDMolSupplier(str(self.path))
        for mol in supplier:
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            row = {self.smiles_col: Chem.MolToSmiles(mol)}
            row.update(props)
            rows.append(row)
        return pd.DataFrame(rows)

    def fetch_activities(
        self, target_id: str | None = None, endpoint: str | None = None, **filters: object
    ) -> pd.DataFrame:
        raw = self._read()
        if self.smiles_col not in raw.columns:
            raise ValueError(
                f"SMILES column {self.smiles_col!r} not found in {self.path.name}; "
                f"columns are {list(raw.columns)}."
            )
        out = pd.DataFrame(
            {
                Column.SMILES: raw[self.smiles_col].astype(str),
                Column.VALUE: raw[self.value_col] if self.value_col in raw else pd.NA,
            }
        )
        out[Column.MOLECULE_ID] = (
            raw[self.id_col].astype(str)
            if self.id_col and self.id_col in raw.columns
            else [f"MOL{i}" for i in range(len(out))]
        )
        out[Column.TARGET_ID] = target_id or self.target_id
        out[Column.ENDPOINT] = endpoint or self.endpoint
        out[Column.UNITS] = self.units
        return self._validate_schema(out)
