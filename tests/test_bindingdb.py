"""BindingDB source — REST parsing tested offline via a mocked HTTP layer."""

from __future__ import annotations

import ligandra.data.bindingdb as bdb
from ligandra.core.types import Column
from ligandra.data import DATA_SOURCES

_UNIPROT_RESPONSE = {
    "results": [
        {
            "primaryAccession": "P00533",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
            },
            "organism": {"scientificName": "Homo sapiens"},
            "genes": [{"geneName": {"value": "EGFR"}}],
        }
    ]
}

_BDB_RESPONSE = {
    "getLigandsByUniprotsResponse": {
        "affinities": [
            {
                "query": "P00533",
                "monomerid": 1,
                "smile": "CCO",
                "affinity_type": "IC50",
                "affinity": ">500",
            },
            {
                "query": "P00533",
                "monomerid": 2,
                "smile": "c1ccccc1",
                "affinity_type": "Ki",
                "affinity": "12.5",
            },
            {
                "query": "P00533",
                "monomerid": 3,
                "smile": "CC(=O)O",
                "affinity_type": "IC50",
                "affinity": "340",
            },
            {
                "query": "P00533",
                "monomerid": 4,
                "smile": "",
                "affinity_type": "IC50",
                "affinity": "10",
            },
        ]
    }
}


def test_bindingdb_registered():
    assert "bindingdb" in DATA_SOURCES.available()


def test_search_targets_parses_uniprot(monkeypatch):
    monkeypatch.setattr(bdb, "_fetch_json", lambda url, timeout=30.0: _UNIPROT_RESPONSE)
    src = DATA_SOURCES.create("bindingdb")
    targets = src.search_targets("EGFR")
    assert len(targets) == 1
    t = targets[0]
    assert t.target_id == "P00533"
    assert "Epidermal growth factor" in t.name
    assert t.organism == "Homo sapiens"
    assert t.extra["gene"] == "EGFR"


_UNIPROT_MULTISPECIES = {
    "results": [
        {  # honeybee EGFR -- UniProt relevance ranks it first, but it has ~no data
            "primaryAccession": "P0CY46",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
            },
            "organism": {"scientificName": "Apis mellifera"},
            "genes": [{"geneName": {"value": "Egfr"}}],
        },
        {  # human EGFR -- the accession a drug-design user actually means
            "primaryAccession": "P00533",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
            },
            "organism": {"scientificName": "Homo sapiens"},
            "genes": [{"geneName": {"value": "EGFR"}}],
        },
    ]
}


def test_select_target_prefers_human_over_ortholog(monkeypatch):
    # UniProt returns a non-human ortholog first; select_target must still land
    # on the human accession that carries the BindingDB ligands (else the source
    # fetches an empty target and "the link looks broken").
    monkeypatch.setattr(bdb, "_fetch_json", lambda url, timeout=30.0: _UNIPROT_MULTISPECIES)
    src = DATA_SOURCES.create("bindingdb")
    t = src.select_target("EGFR", "IC50")
    assert t.target_id == "P00533"
    assert t.organism == "Homo sapiens"


def test_fetch_activities_filters_endpoint_and_parses_relations(monkeypatch):
    monkeypatch.setattr(bdb, "_fetch_json", lambda url, timeout=30.0: _BDB_RESPONSE)
    src = DATA_SOURCES.create("bindingdb")
    df = src.fetch_activities("P00533", "IC50")

    # only IC50 rows with a parseable value and a non-empty SMILES survive
    assert list(df[Column.ENDPOINT].unique()) == ["IC50"]
    assert len(df) == 2  # >500 and 340; the empty-SMILES IC50 row is dropped
    assert set(df[Column.SMILES]) == {"CCO", "CC(=O)O"}
    assert set(df[Column.UNITS]) == {"nM"}

    row = df[df[Column.SMILES] == "CCO"].iloc[0]
    assert row[Column.VALUE] == 500.0
    assert row[Column.RELATION] == ">"
    assert row[Column.MOLECULE_ID] == "BDB:1"


def test_fetch_activities_parses_bindingdb_typo_wrapper_key(monkeypatch):
    # BindingDB's live API nests the payload under "getLindsByUniprotsResponse"
    # (a typo in their own service).  Parsing must not depend on the correctly
    # spelled key, otherwise every real pull silently returns zero rows.
    typo = {"getLindsByUniprotsResponse": _BDB_RESPONSE["getLigandsByUniprotsResponse"]}
    monkeypatch.setattr(bdb, "_fetch_json", lambda url, timeout=30.0: typo)
    src = DATA_SOURCES.create("bindingdb")
    df = src.fetch_activities("P00533", "IC50")
    assert len(df) == 2  # the two IC50 rows are recovered, not dropped
    assert set(df[Column.SMILES]) == {"CCO", "CC(=O)O"}


def test_fetch_activities_empty_keeps_schema(monkeypatch):
    monkeypatch.setattr(
        bdb,
        "_fetch_json",
        lambda url, timeout=30.0: {"getLigandsByUniprotsResponse": {"affinities": []}},
    )
    src = DATA_SOURCES.create("bindingdb")
    df = src.fetch_activities("P99999", "EC50")
    assert df.empty
    for col in Column.RAW_SCHEMA:
        assert col in df.columns


def test_parse_affinity_variants():
    assert bdb._parse_affinity("12.5") == (12.5, "=")
    assert bdb._parse_affinity(">1000") == (1000.0, ">")
    assert bdb._parse_affinity("<0.5") == (0.5, "<")
    assert bdb._parse_affinity(None) == (None, None)
    assert bdb._parse_affinity("n/a") == (None, None)
