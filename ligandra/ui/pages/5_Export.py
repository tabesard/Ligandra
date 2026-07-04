"""Page 5 — Export the ranked candidate set (CSV/SDF)."""

from __future__ import annotations

import streamlit as st

from ligandra.core.types import Column
from ligandra.ui.common import get_config, molecule_grid, require, sidebar_disclaimer

st.title("⑤ Export")
sidebar_disclaimer()
cfg = get_config()

if not require("candidates", "Generate candidates on page ④ first."):
    st.stop()

cand = st.session_state["candidates"]
if cand is None or cand.empty:
    st.info("No candidates to export.")
    st.stop()

# Show the top candidates as 2D structures with predicted potency.
p_name = cfg.target.endpoint.p_name
pred_col = f"pred_{p_name}"
top = cand.head(12).reset_index(drop=True)
legends = [
    "  ·  ".join(
        [f"#{i + 1}"]
        + ([f"{p_name} {top[pred_col][i]:.2f}"] if pred_col in top.columns else [])
        + ([f"QED {top['qed'][i]:.2f}"] if "qed" in top.columns else [])
    )
    for i in range(len(top))
]
molecule_grid(top[Column.SMILES].tolist(), legends)
with st.expander("Full ranked table"):
    st.dataframe(cand, width="stretch")
st.download_button(
    "⬇️ Download candidates.csv",
    cand.to_csv(index=False),
    file_name="candidates.csv",
    mime="text/csv",
)

st.caption(
    "SDF export is available headless via the API/CLI: "
    "`ligandra run experiment.yaml --export candidates.sdf`."
)
