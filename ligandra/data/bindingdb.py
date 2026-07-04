"""BindingDB data source (real REST implementation).

BindingDB is organized by protein (UniProt).  This source therefore:

* ``search_targets(query)`` resolves a free-text query (gene name / protein /
  organism, or a raw UniProt accession) to UniProt accessions via the UniProt
  REST API — those accessions are the BindingDB target ids; and
* ``fetch_activities(uniprot, endpoint, cutoff=...)`` pulls ligands + affinities
  from BindingDB's ``getLigandsByUniprots`` service and maps them into the
  normalized schema, filtered to the requested endpoint (IC50/Ki/Kd/EC50).

HTTP goes through :func:`_fetch_json` (stdlib ``urllib``; no extra dependency),
which the tests monkeypatch so the *parsing* is exercised offline.  Network is
only needed to hit the live services.

Refs:
  BindingDB RESTful API — https://www.bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp
  UniProt REST search   — https://www.uniprot.org/help/api_queries
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from ligandra.core.types import Column, Target
from ligandra.data.base import DATA_SOURCES, DataSource

_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_BINDINGDB_LIGANDS = "https://bindingdb.org/rest/getLigandsByUniprots"
_USER_AGENT = "ligandra/0.1 (https://github.com/; data-source=bindingdb)"

#: BindingDB reports affinities in nM; values may carry a relation prefix.
_RELATION_RE = re.compile(r"^\s*([<>~=]{1,2})?\s*([0-9.eE+-]+)")

#: Split a name/gene into lowercase alpha/digit tokens ("FGFR1" -> {"fgfr", "1"}).
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")


def _tokens(text) -> set[str]:
    """Lowercase alphanumeric tokens, splitting letter/digit runs.

    ``"FGFR1"`` and ``"FGFR 1"`` both tokenize to ``{"fgfr", "1"}`` so a gene
    symbol matches the query whether or not the user typed a separator.
    """
    return set(_TOKEN_RE.findall(str(text).lower()))


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    """GET ``url`` and parse JSON. Isolated so tests can monkeypatch it."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
            payload = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
        raise RuntimeError(f"BindingDB/UniProt request failed: {exc}") from exc
    return json.loads(payload)


def _as_list(value) -> list:
    """BindingDB/UniProt return a bare object when there is a single record."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _find_affinities(data: dict) -> list:
    """Locate the ``affinities`` list regardless of the response wrapper key.

    BindingDB's live JSON nests the payload under ``getLindsByUniprotsResponse``
    — a typo in their own API ("Linds", not "Ligands") — so keying on the
    correctly-spelled ``getLigandsByUniprotsResponse`` (as older/mock payloads
    use) always missed it and returned nothing.  Search the top level and any
    nested object so both spellings, and a bare ``affinities`` list, work.
    """
    if not isinstance(data, dict):
        return []
    if "affinities" in data:
        return _as_list(data["affinities"])
    for value in data.values():
        if isinstance(value, dict) and "affinities" in value:
            return _as_list(value["affinities"])
    return []


def _parse_affinity(raw) -> tuple[float | None, str | None]:
    """Split a BindingDB affinity string like ``'>1000'`` into (value, relation)."""
    if raw is None:
        return None, None
    m = _RELATION_RE.match(str(raw))
    if not m:
        return None, None
    relation = m.group(1) or "="
    try:
        return float(m.group(2)), relation
    except (TypeError, ValueError):
        return None, relation


@DATA_SOURCES.register("bindingdb")
class BindingDBSource(DataSource):
    """Query BindingDB (via UniProt) for targets and their bioactivities.

    Parameters
    ----------
    cutoff:
        Affinity cutoff in nM passed to BindingDB (keep only data at least this
        potent).  Default 100000 nM ≈ effectively "no cutoff" for curation.
    reviewed_only:
        Restrict UniProt search to reviewed (Swiss-Prot) entries.
    max_records:
        Optional cap on the number of activity rows returned.
    """

    def __init__(
        self,
        cutoff: int = 100000,
        reviewed_only: bool = True,
        max_records: int | None = None,
    ) -> None:
        self.cutoff = cutoff
        self.reviewed_only = reviewed_only
        self.max_records = max_records

    # -- targets ---------------------------------------------------------
    def search_targets(self, query: str) -> list[Target]:
        q = query.strip()
        if self.reviewed_only:
            q = f"({q}) AND reviewed:true"
        params = {
            "query": q,
            "fields": "accession,protein_name,organism_name,gene_names",
            "format": "json",
            "size": "25",
        }
        url = f"{_UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}"
        data = _fetch_json(url)
        return self._parse_targets(data)

    def select_target(self, query: str, endpoint: str = "IC50") -> Target | None:
        """Pick the human protein a drug-design user means by a gene/protein name.

        UniProt relevance frequently ranks a non-human ortholog first (``EGFR``
        -> honeybee ``P0CY46``; ``FGFR1`` -> newt ``Q91285``), and those
        orthologs carry little or no BindingDB data -- so the base class's "first
        hit" resolves to an effectively empty accession and BindingDB looks
        broken.  Rank hits by (gene symbol matches the query, human organism,
        token overlap) so the query lands on the human accession that actually
        holds the ligands (``EGFR`` -> ``P00533``, ``FGFR1`` -> ``P11362``).
        """
        hits = self.search_targets(query)
        if not hits:
            return None
        q = _tokens(query)

        def _rank(t: Target) -> tuple[int, int, int]:
            gene = (t.extra or {}).get("gene") or ""
            gene_match = 1 if q and _tokens(gene) == q else 0
            human = 1 if (t.organism or "") == "Homo sapiens" else 0
            overlap = len(q & (_tokens(gene) | _tokens(t.name)))
            return (gene_match, human, overlap)

        # max keeps the first hit on ties, preserving UniProt's relevance order.
        return max(hits, key=_rank)

    @staticmethod
    def _parse_targets(data: dict) -> list[Target]:
        targets: list[Target] = []
        for r in _as_list(data.get("results")):
            acc = r.get("primaryAccession")
            if not acc:
                continue
            name = (
                r.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value")
            )
            genes = r.get("genes") or []
            gene = genes[0].get("geneName", {}).get("value") if genes else None
            targets.append(
                Target(
                    target_id=acc,
                    name=name or gene or acc,
                    organism=(r.get("organism") or {}).get("scientificName"),
                    target_type="SINGLE PROTEIN",
                    source="bindingdb",
                    extra={"gene": gene},
                )
            )
        return targets

    # -- activities ------------------------------------------------------
    def fetch_activities(
        self, target_id: str, endpoint: str = "IC50", **filters: object
    ) -> pd.DataFrame:
        cutoff = int(filters.get("cutoff", self.cutoff))  # type: ignore[arg-type]
        params = {
            "uniprot": target_id,
            "cutoff": str(cutoff),
            "response": "application/json",
        }
        url = f"{_BINDINGDB_LIGANDS}?{urllib.parse.urlencode(params)}"
        data = _fetch_json(url)
        df = self._parse_activities(data, target_id, endpoint)
        return self._validate_schema(df)

    def _parse_activities(self, data: dict, target_id: str, endpoint: str) -> pd.DataFrame:
        records = _find_affinities(data)
        rows = []
        for i, rec in enumerate(records):
            if self.max_records is not None and len(rows) >= self.max_records:
                break
            atype = str(rec.get("affinity_type", "")).strip()
            if atype.lower() != endpoint.lower():
                continue
            smiles = rec.get("smile") or rec.get("smiles")
            if not smiles:
                continue
            value, relation = _parse_affinity(rec.get("affinity"))
            if value is None:
                continue
            monomer = rec.get("monomerid")
            rows.append(
                {
                    Column.MOLECULE_ID: f"BDB:{monomer}" if monomer else f"BDB:{i}",
                    Column.SMILES: smiles,
                    Column.TARGET_ID: target_id,
                    Column.ENDPOINT: atype,
                    Column.VALUE: value,
                    Column.UNITS: "nM",
                    Column.RELATION: relation,
                    Column.ASSAY_METADATA: json.dumps(
                        {"source": "bindingdb", "monomerid": monomer}
                    ),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=list(Column.RAW_SCHEMA) + [Column.RELATION, Column.ASSAY_METADATA]
            )
        return pd.DataFrame(rows)
