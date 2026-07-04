"""Page 1 — Target, Data & Curation."""

from __future__ import annotations

import streamlit as st

from ligandra.core.types import Column
from ligandra.pipeline.runner import load_and_curate
from ligandra.ui.common import get_config, sidebar_disclaimer
from ligandra.ui.jobs import hash_key, render_job

st.title("① Data & Curation")
sidebar_disclaimer()
cfg = get_config()
st.caption(f"Source: **{cfg.target.source}** · Endpoint: **{cfg.target.endpoint.value}**")

if st.button("Fetch & curate data", type="primary"):
    st.session_state["_curate_req"] = True

if st.session_state.get("_curate_req"):
    # Runs in the background (cached on target+curation) so the tab stays live.
    key = hash_key(
        cfg.target.model_dump(mode="json"), cfg.curation.model_dump(mode="json")
    )
    status, payload = render_job(
        "curate", key, load_and_curate, cfg.model_copy(deep=True),
        running_msg="Fetching & curating…",
    )
    if status in ("done", "cached"):
        st.session_state["curated"], st.session_state["curation_report"] = payload
        st.session_state["_curate_req"] = False
    elif status == "error":  # surface a meaningful error, never a dead page
        st.error(f"Data step failed: {payload}")
        st.session_state["_curate_req"] = False

if "curated" in st.session_state:
    curated = st.session_state["curated"]
    report = st.session_state["curation_report"]
    st.success(f"Curated **{len(curated)}** unique molecules.")
    c1, c2 = st.columns(2)
    c1.json(report.as_dict())
    if "label" in curated:
        c2.bar_chart(curated["label"].value_counts())
    st.dataframe(curated, width="stretch")
    st.caption(f"p-value column: `{cfg.target.endpoint.p_name}` · SMILES column: `{Column.SMILES}`")

    # Download the curated table, and the raw pull if it was cached to disk.
    from ligandra.data.cache import load_dataframe, raw_cache_path

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Curated CSV",
        curated.to_csv(index=False),
        file_name="curated.csv",
        mime="text/csv",
    )
    target_key = cfg.target.target_id or cfg.target.query or cfg.target.path or "none"
    raw_df = load_dataframe(
        raw_cache_path(cfg.target.source, target_key, cfg.target.endpoint.value, cfg.cache.dir)
    )
    if raw_df is not None:
        # Show which target the query actually resolved to, and how many raw rows
        # it returned — so a query landing on the wrong/sparse target is visible
        # rather than silently producing "a few ligands".
        if Column.TARGET_ID in raw_df.columns:
            resolved = ", ".join(str(t) for t in raw_df[Column.TARGET_ID].dropna().unique()[:3])
            st.caption(
                f"🎯 Query **{target_key}** → target **{resolved}** · "
                f"{len(raw_df)} raw activities fetched → {len(curated)} curated."
            )
        d2.download_button(
            "⬇️ Raw fetched CSV",
            raw_df.to_csv(index=False),
            file_name="raw_activities.csv",
            mime="text/csv",
        )
    if cfg.cache.enabled:
        st.caption(
            f"💾 Saved under `{cfg.cache.dir}/` — the next run reads these instead "
            "of re-fetching (tick refresh in the config to force a re-pull)."
        )
else:
    st.info("Configure the experiment on the Home page, then fetch data here.")
