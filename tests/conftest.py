"""Shared pytest fixtures and warning filters."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="session")
def mini_dataset_path() -> str:
    return str(EXAMPLES / "mini_dataset.csv")


@pytest.fixture(scope="session")
def sample_smiles() -> list[str]:
    return [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Nc1ccc(O)cc1",
        "O=C(O)c1ccccc1",
        "c1ccc(-c2ccncc2)cc1",
        "CN1CCC[C@H]1c1cccnc1",
        "COc1ccc(CCN)cc1",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "O=C(O)Cc1ccccc1",
    ]


@pytest.fixture(scope="session")
def sample_activities():
    """(smiles, IC50 nM) rows for curation tests."""
    return [
        ("CCO", 1000.0, "nM"),
        ("c1ccccc1", 5000.0, "nM"),
        ("CC(=O)O", 20000.0, "nM"),
        ("OCC", 1.0, "uM"),  # duplicate of CCO in different units
    ]
