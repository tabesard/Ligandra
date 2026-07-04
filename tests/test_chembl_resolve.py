"""ChEMBL target-id resolution (offline; the UniProt/text paths are mocked).

Guards the regression where a UniProt accession (e.g. ``p00533``) was passed
straight to ChEMBL's ``target_chembl_id`` filter and silently returned nothing.
"""

from __future__ import annotations

import ligandra.data.chembl as chembl
from ligandra.data import DATA_SOURCES


class _FakeQuery(list):
    def only(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return _FakeQuery(result) if isinstance(key, slice) else result


class _FakeTarget:
    def __init__(self, acc_hits, text_hits):
        self._acc_hits = acc_hits
        self._text_hits = text_hits

    def filter(self, **kwargs):
        return _FakeQuery(self._acc_hits)

    def search(self, query):
        return _FakeQuery(self._text_hits)


class _FakeClient:
    def __init__(self, acc_hits=None, text_hits=None):
        self.target = _FakeTarget(acc_hits or [], text_hits or [])


def test_chembl_id_fast_path_needs_no_client(monkeypatch):
    # A ChEMBL id resolves without ever touching the network/client.
    def _boom():
        raise AssertionError("client should not be called for a ChEMBL id")

    monkeypatch.setattr(chembl, "_get_client", _boom)
    src = DATA_SOURCES.create("chembl")
    assert src.resolve_target_id("CHEMBL203") == "CHEMBL203"
    assert src.resolve_target_id("chembl203") == "CHEMBL203"  # case-normalized
    assert src.resolve_target_id("  CHEMBL2074  ") == "CHEMBL2074"


def test_uniprot_accession_resolves_preferring_single_protein(monkeypatch):
    hits = [
        {"target_chembl_id": "CHEMBL2111431", "target_type": "PROTEIN FAMILY"},
        {"target_chembl_id": "CHEMBL203", "target_type": "SINGLE PROTEIN"},
    ]
    monkeypatch.setattr(chembl, "_get_client", lambda: _FakeClient(acc_hits=hits))
    src = DATA_SOURCES.create("chembl")
    assert src.resolve_target_id("P00533") == "CHEMBL203"  # single protein preferred
    assert src.resolve_target_id("p00533") == "CHEMBL203"  # accession upper-cased


def test_empty_target_id_is_not_searched(monkeypatch):
    # an empty/blank id must not trigger a network search (returns as-is)
    def _boom():
        raise AssertionError("client should not be called for an empty id")

    monkeypatch.setattr(chembl, "_get_client", _boom)
    src = DATA_SOURCES.create("chembl")
    assert src.resolve_target_id("") == ""
    assert src.resolve_target_id("   ") == ""
    assert src.search_targets("") == []
    assert src.search_targets("   ") == []


def test_empty_target_id_falls_back_to_query_search():
    # the pipeline treats target_id="" (from an empty UI box) as "not provided"
    import pandas as pd

    from ligandra.config.schema import CacheConfig, ExperimentConfig, TargetConfig
    from ligandra.core.types import Column, Endpoint, Target
    from ligandra.data.base import DATA_SOURCES, DataSource
    from ligandra.pipeline.runner import _fetch_raw

    @DATA_SOURCES.register("blank_id_source")
    class _Src(DataSource):
        searched: list = []

        def search_targets(self, query):
            type(self).searched.append(query)
            return [Target(target_id="RESOLVED", name=query, source="t")]

        def fetch_activities(self, target_id, endpoint, **f):
            return pd.DataFrame(
                {
                    Column.MOLECULE_ID: ["m0"],
                    Column.SMILES: ["CCO"],
                    Column.TARGET_ID: [target_id],
                    Column.ENDPOINT: [endpoint],
                    Column.VALUE: [100.0],
                    Column.UNITS: ["nM"],
                }
            )

    cfg = ExperimentConfig(
        target=TargetConfig(
            source="blank_id_source", target_id="", query="EGFR", endpoint=Endpoint.IC50
        ),
        cache=CacheConfig(enabled=False),
    )
    raw = _fetch_raw(cfg)
    assert _Src.searched == ["EGFR"]  # searched by name, not by the empty id
    assert raw[Column.TARGET_ID].iloc[0] == "RESOLVED"


class _FakeCountQuery:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _FakeActivity:
    def __init__(self, counts):
        self._counts = counts

    def filter(self, target_chembl_id=None, standard_type=None, **kw):
        return _FakeCountQuery(self._counts.get(target_chembl_id, 0))


class _FakeSearchTarget:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query):
        return _FakeQuery(self._hits)


class _FakeSelectClient:
    def __init__(self, hits, counts):
        self.target = _FakeSearchTarget(hits)
        self.activity = _FakeActivity(counts or {})


def _hit(tid, name, *, score, organism="Homo sapiens", ttype="SINGLE PROTEIN"):
    return {
        "target_chembl_id": tid,
        "pref_name": name,
        "organism": organism,
        "target_type": ttype,
        "score": score,
    }


def test_select_target_numeric_token_beats_high_score_adaptor(monkeypatch):
    # "FGFR 1": text relevance ranks the FRS2 adaptor (few ligands) and non-human
    # orthologs first; the numeric token "1" + human single-protein filter must
    # still land on human FGFR1, which carries the data.
    hits = [
        _hit("CHEMBL_FRS2", "Fibroblast growth factor receptor substrate 2", score=17.0),
        _hit("CHEMBL_MOUSE", "Fibroblast growth factor receptor 1", score=14.0,
             organism="Mus musculus"),
        _hit("CHEMBL_FAMILY", "Fibroblast growth factor receptor", score=16.0,
             ttype="PROTEIN FAMILY"),
        _hit("CHEMBL_FGFR1", "Fibroblast growth factor receptor 1", score=11.0),
    ]
    counts = {"CHEMBL_FRS2": 3, "CHEMBL_MOUSE": 15, "CHEMBL_FGFR1": 9000}
    monkeypatch.setattr(
        chembl, "_get_client", lambda: _FakeSelectClient(hits, counts)
    )
    src = DATA_SOURCES.create("chembl")
    assert src.select_target("FGFR 1", "IC50").target_id == "CHEMBL_FGFR1"


def test_select_target_skips_zero_data_target(monkeypatch):
    # When the query matches no target name, the highest-relevance human single
    # protein with *no* activities for this endpoint must lose to one that has data.
    hits = [
        _hit("CHEMBL_EMPTY", "Some kinase", score=20.0),
        _hit("CHEMBL_DATA", "Another kinase", score=10.0),
    ]
    counts = {"CHEMBL_EMPTY": 0, "CHEMBL_DATA": 500}
    monkeypatch.setattr(
        chembl, "_get_client", lambda: _FakeSelectClient(hits, counts)
    )
    src = DATA_SOURCES.create("chembl")
    assert src.select_target("KIT", "IC50").target_id == "CHEMBL_DATA"


def test_select_target_none_when_no_hits(monkeypatch):
    monkeypatch.setattr(chembl, "_get_client", lambda: _FakeSelectClient([], {}))
    src = DATA_SOURCES.create("chembl")
    assert src.select_target("nothing-here", "IC50") is None


def test_resolve_target_id_prefers_human_single_protein(monkeypatch):
    # A gene-name in the "exact target id" box hits ChEMBL text search, which
    # ranks non-human orthologs first ("FGFR1" -> mouse before human).  The
    # human single protein (with ~600x more activities) must win, otherwise the
    # id box silently pulls a sparse ortholog ("only a few ligands").
    text = [
        {"target_chembl_id": "CHEMBL_MOUSE", "target_type": "SINGLE PROTEIN",
         "organism": "Mus musculus"},
        {"target_chembl_id": "CHEMBL_RAT", "target_type": "SINGLE PROTEIN",
         "organism": "Rattus norvegicus"},
        {"target_chembl_id": "CHEMBL_HUMAN", "target_type": "SINGLE PROTEIN",
         "organism": "Homo sapiens"},
    ]
    monkeypatch.setattr(
        chembl, "_get_client", lambda: _FakeClient(acc_hits=[], text_hits=text)
    )
    src = DATA_SOURCES.create("chembl")
    assert src.resolve_target_id("FGFR1") == "CHEMBL_HUMAN"


def test_falls_back_to_text_search_then_original(monkeypatch):
    # no accession hit, but a text-search hit
    text = [{"target_chembl_id": "CHEMBL2074", "target_type": "SINGLE PROTEIN"}]
    monkeypatch.setattr(
        chembl, "_get_client", lambda: _FakeClient(acc_hits=[], text_hits=text)
    )
    src = DATA_SOURCES.create("chembl")
    assert src.resolve_target_id("Maltase-glucoamylase") == "CHEMBL2074"

    # nothing resolves -> return the original string (caller gets a clear error)
    monkeypatch.setattr(chembl, "_get_client", lambda: _FakeClient())
    assert src.resolve_target_id("not-a-target") == "not-a-target"
