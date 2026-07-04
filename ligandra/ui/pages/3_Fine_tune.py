"""Page 3 — Transfer learning: fine-tune a pretrained foundation model (M5)."""

from __future__ import annotations

import streamlit as st

from ligandra.config.schema import ModelConfig
from ligandra.core.types import Column
from ligandra.pipeline.runner import train_and_benchmark
from ligandra.ui.common import get_config, require, sidebar_disclaimer
from ligandra.ui.jobs import hash_key, render_job

st.title("③ Fine-tune a foundation model")
sidebar_disclaimer()
cfg = get_config()

st.markdown(
    "Load a pretrained molecular transformer (ChemBERTa / MoLFormer / any newer "
    "HF checkpoint) and fine-tune a head on **this target's** data. Needs "
    "`torch` + `transformers`; runs best on GPU."
)

if not require("curated", "Run the Data & Curation page first."):
    st.stop()

ckpt = st.text_input("Pretrained checkpoint (HF hub id or local path)", "DeepChem/ChemBERTa-77M-MTR")
freeze = st.checkbox("Frozen encoder + trainable head (fast, small data)", value=True)
epochs = st.number_input("Epochs", value=20, step=5)

if st.button("Fine-tune", type="primary"):
    st.session_state["_ft_req"] = True

if st.session_state.get("_ft_req"):
    curated = st.session_state["curated"]
    ft_cfg = cfg.model_copy(deep=True)
    ft_cfg.models = [
        ModelConfig(
            name="foundation",
            params={"freeze_encoder": bool(freeze), "epochs": int(epochs)},
            pretrained=ckpt,
            finetune=True,
        )
    ]
    ds_sig = hash_key(sorted(curated[Column.SMILES].astype(str).tolist()))
    key = hash_key("finetune", ds_sig, ckpt, bool(freeze), int(epochs), cfg.task.value)
    status, payload = render_job(
        "finetune", key, train_and_benchmark, ft_cfg, curated,
        running_msg="Fine-tuning… (this can take a while)",
    )
    if status in ("done", "cached"):
        board, best = payload
        st.session_state["leaderboard_ft"] = board
        st.session_state["best"] = best  # reuse as the potency objective
        st.session_state["_ft_req"] = False
        st.success("Fine-tuned checkpoint trained and selected as the potency model.")
        st.dataframe(board.to_dataframe(), width="stretch")
    elif status == "error":
        st.error(f"{payload}")
        st.session_state["_ft_req"] = False
