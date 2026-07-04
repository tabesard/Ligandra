"""Curation math: unit conversion, pXC50, labeling, scaffold split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ligandra.core.types import Column, Endpoint
from ligandra.curate import curate, make_split
from ligandra.curate.curator import label_activity
from ligandra.curate.splits import bemis_murcko_scaffold, scaffold_split
from ligandra.curate.units import pxc50, pxc50_array, to_molar


def test_to_molar_units():
    assert to_molar(1000, "nM") == pytest.approx(1e-6)
    assert to_molar(1, "uM") == pytest.approx(1e-6)
    assert to_molar(1, "M") == pytest.approx(1.0)
    assert to_molar(1000, "pM") == pytest.approx(1e-9)


def test_pxc50_matches_classic_pic50():
    # 1000 nM IC50 -> pIC50 = 6.0
    assert pxc50(1000, "nM") == pytest.approx(6.0)
    assert pxc50(1, "uM") == pytest.approx(6.0)
    # unit-independence: same molar concentration -> same pXC50
    assert pxc50(1000, "nM") == pytest.approx(pxc50(0.001, "mM"))


def test_pxc50_endpoint_generalizes():
    # The transform is endpoint-agnostic (Ki uses the same math).
    assert Endpoint.KI.p_name == "pKi"
    assert Endpoint.KI.is_concentration


def test_pxc50_array_handles_bad_values():
    out = pxc50_array([1000, 0, -5, None], "nM")
    assert out[0] == pytest.approx(6.0)
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])


def test_pxc50_rejects_unknown_unit():
    with pytest.raises(KeyError):
        pxc50(1.0, "furlongs")


def test_to_molar_mass_units_need_molecular_weight():
    # aspirin MW ~180.16 g/mol; 1 ug/mL -> ~5.55 uM
    assert to_molar(1.0, "ug.mL-1", mol_weight=180.16) == pytest.approx(5.55e-6, rel=1e-2)
    assert to_molar(1.0, "mg/mL", mol_weight=180.16) == pytest.approx(5.55e-3, rel=1e-2)
    # a mass unit without a molecular weight is not convertible
    with pytest.raises(KeyError):
        to_molar(1.0, "ug.mL-1")


def test_curate_converts_mass_units_and_drops_unconvertible():
    df = pd.DataFrame(
        {
            Column.MOLECULE_ID: ["a", "b", "c"],
            Column.SMILES: ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"],
            Column.TARGET_ID: ["T"] * 3,
            Column.ENDPOINT: ["IC50"] * 3,
            Column.VALUE: [1000.0, 2.0, 5.0],
            Column.UNITS: ["nM", "ug.mL-1", "some.weird.unit"],
        }
    )
    curated, _ = curate(df, Endpoint.IC50)
    kept = set(curated[Column.SMILES])
    # nM + mass-unit rows convert; the unrecognised unit row is dropped (no crash)
    assert "some.weird.unit" not in set(curated[Column.UNITS])
    assert len(curated) == 2
    assert kept == {"CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"}


def test_label_activity_thresholds():
    assert label_activity(500, 1000, 10000) == "active"
    assert label_activity(50000, 1000, 10000) == "inactive"
    assert label_activity(5000, 1000, 10000) == "intermediate"
    assert label_activity(float("nan"), 1000, 10000) == "undefined"


def test_curate_dedupes_mixed_units_and_drops_invalid(sample_activities):
    df = pd.DataFrame(
        {
            Column.MOLECULE_ID: ["A", "B", "C", "A2"],
            Column.SMILES: [s for s, _, _ in sample_activities],
            Column.TARGET_ID: ["T"] * 4,
            Column.ENDPOINT: ["IC50"] * 4,
            Column.VALUE: [v for _, v, _ in sample_activities],
            Column.UNITS: [u for _, _, u in sample_activities],
        }
    )
    curated, report = curate(df, Endpoint.IC50)
    # CCO and OCC (1000 nM and 1 uM) collapse to a single molecule.
    assert report.n_duplicates_merged == 1
    assert len(curated) == 3
    ethanol = curated[curated[Column.SMILES] == "CCO"]
    assert float(ethanol["pIC50"].iloc[0]) == pytest.approx(6.0)
    assert ethanol["label"].iloc[0] == "active"


def test_curate_empty_after_missing():
    df = pd.DataFrame(
        {
            Column.MOLECULE_ID: ["A"],
            Column.SMILES: [None],
            Column.TARGET_ID: ["T"],
            Column.ENDPOINT: ["IC50"],
            Column.VALUE: [None],
            Column.UNITS: ["nM"],
        }
    )
    curated, report = curate(df, Endpoint.IC50)
    assert curated.empty
    assert report.n_output == 0


def test_scaffold_split_no_leakage(sample_smiles):
    # Repeat molecules so scaffolds have multiple members.
    smiles = sample_smiles * 3
    split = scaffold_split(smiles, test_size=0.3, val_size=0.0, seed=1)
    train_scaffolds = {bemis_murcko_scaffold(smiles[i]) for i in split.train}
    test_scaffolds = {bemis_murcko_scaffold(smiles[i]) for i in split.test}
    # No scaffold may appear in both partitions.
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_scaffold_split_is_deterministic(sample_smiles):
    a = scaffold_split(sample_smiles, seed=7)
    b = scaffold_split(sample_smiles, seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


def test_make_split_dispatch(sample_smiles):
    for strat in ("random", "scaffold"):
        sp = make_split(strat, sample_smiles, test_size=0.25, val_size=0.0)
        total = len(sp.train) + len(sp.val) + len(sp.test)
        assert total == len(sample_smiles)
