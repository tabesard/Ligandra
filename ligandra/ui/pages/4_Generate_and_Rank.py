"""Page 4 — Generate de novo molecules & multi-objective rank (M6/M7)."""

from __future__ import annotations

import streamlit as st

from ligandra.core.types import Column
from ligandra.pipeline.runner import generate_and_rank
from ligandra.ui.common import get_config, molecule_grid, require, sidebar_disclaimer
from ligandra.ui.jobs import hash_key, render_job

st.title("④ Generate & Rank")
sidebar_disclaimer()
cfg = get_config()

if not require("curated", "Run the Data & Curation page first."):
    st.stop()
if "best" not in st.session_state:
    st.warning("No trained model yet — the potency objective will be skipped. "
               "Train a model on page ② for target-conditioned generation.")

st.caption(
    f"Generator: **{cfg.generator.name}** · budget {cfg.generator.budget} · "
    f"objectives: {', '.join(o.scorer for o in cfg.objectives)}"
)

if st.button("Generate candidates", type="primary"):
    st.session_state["_gen_req"] = True

if st.session_state.get("_gen_req"):
    curated = st.session_state["curated"]
    best = st.session_state.get("best", {})
    ds_sig = hash_key(sorted(curated[Column.SMILES].astype(str).tolist()))
    key = hash_key(
        "generate", ds_sig, cfg.seed, cfg.generator.model_dump(mode="json"),
        [o.model_dump(mode="json") for o in cfg.objectives],
        [m.model_dump(mode="json") for m in cfg.models],
        best.get("model_name"), best.get("featurizer_name"),
    )
    status, payload = render_job(
        "generate", key, generate_and_rank, cfg.model_copy(deep=True), curated, best,
        running_msg="Optimizing molecules against the target objective…",
    )
    if status in ("done", "cached"):
        st.session_state["candidates"], st.session_state["gen_metrics"] = payload
        st.session_state["_gen_req"] = False
    elif status == "error":
        st.error(f"Generation failed: {payload}")
        st.session_state["_gen_req"] = False

if "candidates" in st.session_state:
    gm = st.session_state["gen_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Validity", f"{gm.get('validity', 0):.0%}")
    c2.metric("Uniqueness", f"{gm.get('uniqueness', 0):.0%}")
    c3.metric("Novelty", f"{gm.get('novelty', 0):.0%}")
    cand = st.session_state["candidates"]
    if cand is None or cand.empty:
        st.info("No novel candidates survived de-duplication — try a larger budget.")
    else:
        st.subheader(f"Ranked candidates ({len(cand)})")

        # 2D structures of the top candidates, captioned with predicted potency.
        p_name = cfg.target.endpoint.p_name
        pred_col = f"pred_{p_name}"
        n_show = st.slider("Structures to show", 4, min(48, len(cand)), min(12, len(cand)))
        top = cand.head(n_show).reset_index(drop=True)
        legends = []
        for i in range(len(top)):
            parts = [f"#{i + 1}"]
            if pred_col in top.columns:
                parts.append(f"{p_name} {top[pred_col][i]:.2f}")
            elif "potency" in top.columns:
                parts.append(f"potency {top['potency'][i]:.2f}")
            if "qed" in top.columns:
                parts.append(f"QED {top['qed'][i]:.2f}")
            legends.append("  ·  ".join(parts))
        molecule_grid(top[Column.SMILES].tolist(), legends)

        with st.expander("Full ranked table"):
            st.dataframe(cand, width="stretch")
