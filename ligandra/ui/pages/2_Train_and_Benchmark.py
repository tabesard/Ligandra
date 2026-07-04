"""Page 2 — Featurize, Train & Benchmark (scaffold-split leaderboard)."""

from __future__ import annotations

import streamlit as st

from ligandra.core.types import Column
from ligandra.pipeline.runner import train_and_benchmark
from ligandra.ui.common import get_config, metric_bar_chart, require, sidebar_disclaimer
from ligandra.ui.jobs import hash_key, render_job

st.title("② Train & Benchmark")
sidebar_disclaimer()
cfg = get_config()

if not require("curated", "Run the Data & Curation page first."):
    st.stop()

st.caption(
    f"Featurizers: {', '.join(f.name for f in cfg.featurizers)} · "
    f"Models: {', '.join(m.name for m in cfg.models)} · Split: {cfg.split.strategy.value}"
)

if st.button("Train & benchmark", type="primary"):
    st.session_state["_train_req"] = True

if st.session_state.get("_train_req"):
    curated = st.session_state["curated"]
    ds_sig = hash_key(sorted(curated[Column.SMILES].astype(str).tolist()))
    key = hash_key(
        "train", ds_sig, cfg.task.value, cfg.split.model_dump(mode="json"),
        [f.model_dump(mode="json") for f in cfg.featurizers],
        [m.model_dump(mode="json") for m in cfg.models],
    )
    status, payload = render_job(
        "train", key, train_and_benchmark, cfg.model_copy(deep=True), curated,
        running_msg="Training models on the held-out split…",
    )
    if status in ("done", "cached"):
        board, best = payload
        st.session_state["leaderboard"] = board
        st.session_state["best"] = best
        st.session_state["_train_req"] = False
    elif status == "error":
        st.error(f"Training failed: {payload}")
        st.session_state["_train_req"] = False

if "leaderboard" in st.session_state:
    board = st.session_state["leaderboard"]
    df = board.to_dataframe()
    st.subheader(f"Leaderboard (ranked by {board.primary_metric})")
    st.dataframe(df, width="stretch")
    best = st.session_state.get("best", {})
    if best:
        st.success(f"Best: **{best.get('model_name')}** on **{best.get('featurizer_name')}** features.")
    metric_bar_chart(df, ["R2", "RMSE", "Pearson", "ROC_AUC"], index_col="model")
