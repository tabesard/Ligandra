"""Shared Streamlit helpers. Keeps every page a thin shell over the library.

All heavy logic lives in :mod:`ligandra.pipeline` / :mod:`ligandra.api`; these
helpers only manage session state and rendering.
"""

from __future__ import annotations

import streamlit as st

from ligandra.api import registries
from ligandra.config.schema import ExperimentConfig, TargetConfig

DISCLAIMER = (
    "⚠️ **Research / early-discovery tool.** Predictions and generated molecules "
    "are *hypotheses* that require experimental validation. Scores are not ground "
    "truth — always inspect the reported uncertainty and novelty."
)


def get_config() -> ExperimentConfig:
    """The single shared experiment config (what the UI reads/writes).

    Defaults the *UI's* first session to the bundled local demo dataset
    (rather than the library's general ``chembl`` default) so a first-time
    visitor gets a fast, network-free result immediately. Switch "Data
    source" to chembl/bindingdb any time to query a real target.
    """
    if "config" not in st.session_state:
        cfg = ExperimentConfig()
        cfg.target = TargetConfig(
            source="local",
            path="examples/mini_dataset.csv",
            smiles_col="smiles",
            value_col="value",
            id_col="molecule_id",
        )
        st.session_state["config"] = cfg
    return st.session_state["config"]


def set_config(cfg: ExperimentConfig) -> None:
    st.session_state["config"] = cfg


def get_registries() -> dict[str, list[str]]:
    if "registries" not in st.session_state:
        st.session_state["registries"] = registries()
    return st.session_state["registries"]


def require(key: str, message: str) -> bool:
    """Guard a page on a prerequisite result; render an info box if missing."""
    if key not in st.session_state:
        st.info(message)
        return False
    return True


def sidebar_disclaimer() -> None:
    inject_style()
    st.sidebar.caption(DISCLAIMER)
    # Cached results make identical fetches/trainings instant; this forces a
    # fresh recompute on the next run.
    from ligandra.ui.jobs import clear_cache

    if st.sidebar.button("↻ Clear cached results"):
        clear_cache()
        st.sidebar.success("Cache cleared — the next run recomputes.")


def metric_bar_chart(df, metric_cols, index_col="model"):
    """Port of the prototype's per-metric bar charts."""
    import matplotlib.pyplot as plt

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        ax.bar(df[index_col].astype(str), df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)


def inject_style() -> None:
    """A little CSS polish (fonts, spacing, metric cards) — visual only, no cost."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        [data-testid="stMetric"] {
            background: rgba(46, 139, 122, 0.06);
            border: 1px solid rgba(46, 139, 122, 0.20);
            border-radius: 12px; padding: 12px 14px;
        }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
        .stButton>button { border-radius: 10px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def molecule_grid(
    smiles: list[str],
    legends: list[str] | None = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (260, 200),
) -> None:
    """Render 2D depictions of molecules in a grid (RDKit), captioned by ``legends``.

    Invalid SMILES are skipped.  Used to show generated candidates as structures
    with their predicted potency, not just a table of strings.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:  # pragma: no cover - RDKit is a core dep
        st.info("Install RDKit to see 2D structures.")
        return

    mols, kept_legends = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        mols.append(mol)
        kept_legends.append(legends[i] if legends and i < len(legends) else "")
    if not mols:
        st.info("No valid structures to display.")
        return

    img = Draw.MolsToGridImage(
        mols,
        legends=kept_legends,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        returnPNG=False,
    )
    st.image(img, width="stretch")
