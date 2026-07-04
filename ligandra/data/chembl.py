"""ChEMBL data source — generalized from the Siglec prototype.

The prototype hard-coded ``search("SIGLEC")`` and ``standard_type="IC50"``.
Here both the target query and the endpoint are parameters, so the source works
for *any* target/endpoint pair.  Results are mapped into the normalized schema.

``target_id`` is resolved leniently: a ChEMBL id (``CHEMBL203``) is used directly
(case-insensitively), while a UniProt accession (``P00533``) or anything else is
resolved to a ChEMBL target id via the target API — so ``p00533`` or a gene name
"just works" instead of silently returning no activities.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from ligandra.core.types import Column, Target
from ligandra.data.base import DATA_SOURCES, DataSource

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, splitting letter/digit runs.

    ``"FGFR 1"`` and ``"FGFR1"`` both tokenize to ``{"fgfr", "1"}`` so a query
    matches a target name whether or not the user typed a space.
    """
    return set(_TOKEN_RE.findall(str(text).lower()))


def _name_match(query: str, name: str | None) -> int:
    """Overlap between the query and a target name, weighting numeric tokens.

    A numeric token (the "1" in "FGFR 1", the "7" in "siglec 7") is the strongest
    signal for *which* paralog is wanted, so it counts double; short/common
    alpha tokens are ignored to avoid rewarding words like "receptor".
    """
    q = _tokens(query)
    n = _tokens(name)
    alpha = {t for t in q if t.isalpha() and len(t) >= 3}
    nums = {t for t in q if t.isdigit()}
    return len(alpha & n) + 2 * len(nums & n)


def _get_client():
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The ChEMBL source needs `chembl_webresource_client`. Install it "
            "with `pip install chembl_webresource_client`, or use the 'local' "
            "source with a CSV/SDF."
        ) from exc
    return new_client


def _retry(fn, *, tries: int = 3, delay: float = 1.5):
    """Call ``fn`` with a few retries on transient network errors.

    ChEMBL occasionally drops the connection mid-transfer; a short backoff and
    retry turns a flaky ``ConnectionReset`` into a successful pull instead of a
    dead run.
    """
    import time

    try:
        from requests.exceptions import (
            ChunkedEncodingError,
            Timeout,
        )
        from requests.exceptions import (
            ConnectionError as ReqConnectionError,
        )

        transient: tuple[type[Exception], ...] = (
            ChunkedEncodingError,
            ReqConnectionError,
            Timeout,
            ConnectionError,
        )
    except ImportError:  # pragma: no cover - requests always present with chembl client
        transient = (ConnectionError,)

    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except transient as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(delay * (attempt + 1))
    raise last  # type: ignore[misc]


@DATA_SOURCES.register("chembl")
class ChEMBLSource(DataSource):
    """Query ChEMBL for targets and their bioactivities."""

    def __init__(self, max_records: int | None = None) -> None:
        self.max_records = max_records

    def search_targets(self, query: str) -> list[Target]:
        q = (query or "").strip()
        if not q:
            return []
        client = _get_client()
        hits = _retry(lambda: list(client.target.search(q)[:25]))
        targets: list[Target] = []
        for h in hits:
            targets.append(
                Target(
                    target_id=h.get("target_chembl_id", ""),
                    name=h.get("pref_name") or h.get("target_chembl_id", ""),
                    organism=h.get("organism"),
                    target_type=h.get("target_type"),
                    source="chembl",
                    extra={"score": h.get("score")},
                )
            )
        return targets

    def select_target(self, query: str, endpoint: str = "IC50") -> Target | None:
        """Choose the best-populated human target for a free-text ``query``.

        ChEMBL's text search ranks by string relevance, which routinely puts a
        wrong-species ortholog (mouse/rat FGFR1), a protein family/complex, a
        protein-protein interaction, or an adaptor (FRS2 for "FGFR 1") above the
        human single protein the user actually wants — so blindly taking the
        first hit yields a target with a handful of ligands.

        Instead, restrict to human ``SINGLE PROTEIN`` hits and rank them by
        ``(name match, has data, ChEMBL relevance, activity count)`` where the
        activity count is a cheap server-side ``total_count`` for this endpoint.
        The name match (numeric tokens weighted, e.g. the "1" in "FGFR 1")
        disambiguates paralogs; the relevance/count tie-breakers pick the
        canonical, well-studied target when no number is given.
        """
        hits = self.search_targets(query)
        if not hits:
            return None
        single = [h for h in hits if h.target_type == "SINGLE PROTEIN"]
        human = [h for h in single if (h.organism or "") == "Homo sapiens"]
        pool = (human or single or hits)[:12]
        if len(pool) == 1:
            return pool[0]

        client = _get_client()

        def _count(tid: str) -> int:
            if not tid:
                return 0
            try:
                return _retry(
                    lambda: len(
                        client.activity.filter(
                            target_chembl_id=tid, standard_type=endpoint
                        )
                    )
                )
            except Exception:  # noqa: BLE001 - a flaky count must not sink resolution
                return 0

        def _rank(h: Target) -> tuple[int, int, float, int]:
            count = _count(h.target_id)
            score = float((h.extra or {}).get("score") or 0.0)
            name_match = _name_match(query, h.name)
            return (name_match, 1 if count > 0 else 0, score, count)

        return max(pool, key=_rank)

    def resolve_target_id(self, target_id: str) -> str:
        """Map a ChEMBL id / UniProt accession / free text to a ChEMBL target id.

        A ``CHEMBL...`` id is normalized to upper case and returned as-is.
        Anything else is looked up as a UniProt accession first, then as a
        free-text target search, preferring a *human* ``SINGLE PROTEIN``.  If
        nothing resolves, the original string is returned so the caller still
        gets a clear "no activities" error rather than a crash.
        """
        tid = str(target_id).strip()
        if not tid:
            return tid  # empty -> caller surfaces a clear "no activities" error
        if _CHEMBL_ID_RE.match(tid):
            return tid.upper()

        client = _get_client()

        def _pick(hits: list[dict]) -> str | None:
            """Best target id from raw hits: human ``SINGLE PROTEIN`` first.

            Text relevance ranks non-human orthologs ahead of the human protein
            (``FGFR1`` -> mouse ``CHEMBL3960`` before human ``CHEMBL3650``, which
            holds ~600x more activities), so preferring a human single protein
            keeps the id box on the well-populated target instead of a sparse
            ortholog that looks like "only a few ligands".
            """
            hits = [h for h in hits if h.get("target_chembl_id")]
            if not hits:
                return None
            single = [h for h in hits if h.get("target_type") == "SINGLE PROTEIN"]
            pool = single or hits
            human = [h for h in pool if (h.get("organism") or "") == "Homo sapiens"]
            return (human or pool)[0]["target_chembl_id"]

        # 1) treat it as a UniProt accession
        by_acc = _retry(
            lambda: list(
                client.target.filter(target_components__accession=tid.upper()).only(
                    ["target_chembl_id", "target_type", "organism"]
                )[:10]
            )
        )
        resolved = _pick(by_acc)
        if resolved:
            return resolved
        # 2) fall back to a free-text target search
        by_text = _retry(
            lambda: list(
                client.target.search(tid).only(
                    ["target_chembl_id", "target_type", "organism"]
                )[:10]
            )
        )
        return _pick(by_text) or tid

    def fetch_activities(
        self, target_id: str, endpoint: str = "IC50", **filters: object
    ) -> pd.DataFrame:
        client = _get_client()
        target_id = self.resolve_target_id(target_id)

        def _collect() -> list[dict]:
            query = client.activity.filter(target_chembl_id=target_id).filter(
                standard_type=endpoint
            )
            out = []
            for i, act in enumerate(query):
                if self.max_records is not None and i >= self.max_records:
                    break
                smiles = act.get("canonical_smiles")
                if not smiles:
                    continue
                out.append(
                    {
                        Column.MOLECULE_ID: act.get("molecule_chembl_id"),
                        Column.SMILES: smiles,
                        Column.TARGET_ID: target_id,
                        Column.ENDPOINT: act.get("standard_type", endpoint),
                        Column.VALUE: act.get("standard_value"),
                        Column.UNITS: act.get("standard_units"),
                        Column.RELATION: act.get("standard_relation"),
                        Column.ASSAY_METADATA: json.dumps(
                            {
                                "assay_chembl_id": act.get("assay_chembl_id"),
                                "assay_type": act.get("assay_type"),
                                "assay_description": act.get("assay_description"),
                            }
                        ),
                    }
                )
            return out

        rows = _retry(_collect)
        df = pd.DataFrame(rows)
        if df.empty:
            # Keep the schema even for empty results so callers can validate.
            df = pd.DataFrame(columns=list(Column.RAW_SCHEMA) + [Column.RELATION, Column.ASSAY_METADATA])
        return self._validate_schema(df)
