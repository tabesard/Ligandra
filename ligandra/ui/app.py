"""Ligandra Streamlit app — Home / experiment configuration.

Run with:  ``streamlit run ligandra/ui/app.py``

The workflow pages live in ``pages/`` and Streamlit lists them automatically.
Every page is a thin shell: it edits the shared :class:`ExperimentConfig` or
calls :mod:`ligandra.pipeline`. Because dropdowns are filled from the plugin
registries, a newly registered model/generator appears here with no edits.
"""

from __future__ import annotations

import streamlit as st

from ligandra.config.schema import (
    FeaturizerConfig,
    GeneratorConfig,
    ModelConfig,
    ObjectiveConfig,
)
from ligandra.core.types import Endpoint, SplitStrategy
from ligandra.ui.common import (
    DISCLAIMER,
    get_config,
    get_registries,
    inject_style,
    set_config,
)

st.set_page_config(page_title="Ligandra", page_icon="🧪", layout="wide")
inject_style()
st.title("🧪 Ligandra — target-agnostic drug design")
st.warning(DISCLAIMER)

cfg = get_config()
reg = get_registries()

st.header("1 · Experiment configuration")
col1, col2 = st.columns(2)
with col1:
    cfg.name = st.text_input("Experiment name", cfg.name)
    cfg.seed = st.number_input("Random seed", value=cfg.seed, step=1)
    cfg.target.source = st.selectbox(
        "Data source", reg["data_sources"], index=reg["data_sources"].index(cfg.target.source)
        if cfg.target.source in reg["data_sources"] else 0,
    )
    cfg.target.endpoint = Endpoint(
        st.selectbox("Endpoint", [e.value for e in Endpoint],
                     index=[e.value for e in Endpoint].index(cfg.target.endpoint.value))
    )
with col2:
    if cfg.target.source == "local":
        cfg.target.path = st.text_input("Local CSV/SDF path", cfg.target.path or "examples/mini_dataset.csv")
        cfg.target.smiles_col = st.text_input("SMILES column", cfg.target.smiles_col)
        cfg.target.value_col = st.text_input("Value column", cfg.target.value_col)
    else:
        # store blank inputs as None (not "") so an empty box means "not provided"
        cfg.target.query = st.text_input(
            "Target search query (e.g. EGFR, SIGLEC)", cfg.target.query or ""
        ).strip() or None
        cfg.target.target_id = st.text_input(
            "…or exact target id (ChEMBL id / UniProt accession)", cfg.target.target_id or ""
        ).strip() or None

st.subheader("Split & curation")
c1, c2, c3 = st.columns(3)
cfg.split.strategy = SplitStrategy(
    c1.selectbox("Split strategy", [s.value for s in SplitStrategy],
                 index=[s.value for s in SplitStrategy].index(cfg.split.strategy.value))
)
cfg.split.test_size = c2.slider("Test fraction", 0.1, 0.5, cfg.split.test_size)
cfg.curation.active_threshold = c3.number_input("Active ≤ (nM)", value=cfg.curation.active_threshold)

st.subheader("Featurizers & models")
c1, c2 = st.columns(2)
feat_names = c1.multiselect("Featurizers", reg["featurizers"], default=[f.name for f in cfg.featurizers])
cfg.featurizers = [FeaturizerConfig(name=n) for n in feat_names] or [FeaturizerConfig(name="ecfp")]
model_names = c2.multiselect("Models", reg["models"], default=[m.name for m in cfg.models])
cfg.models = [ModelConfig(name=n) for n in model_names] or [ModelConfig(name="ridge")]

st.subheader("Generator & objectives")
c1, c2 = st.columns(2)
cfg.generator = GeneratorConfig(
    name=c1.selectbox("Generator", reg["generators"],
                      index=reg["generators"].index(cfg.generator.name) if cfg.generator.name in reg["generators"] else 0),
    budget=int(c1.number_input("Generation budget", value=cfg.generator.budget, step=50)),
    n_output=int(c1.number_input("Candidates to keep", value=cfg.generator.n_output, step=5)),
)
obj_names = c2.multiselect("Objectives (scorers)", reg["scorers"],
                           default=[o.scorer for o in cfg.objectives])
cfg.objectives = [ObjectiveConfig(scorer=n) for n in obj_names] or [ObjectiveConfig(scorer="potency")]

set_config(cfg)

st.header("2 · Config preview")
st.code(cfg.to_yaml_str(), language="yaml")
st.download_button("⬇️ Download experiment.yaml", cfg.to_yaml_str(), file_name="experiment.yaml")
st.info("Proceed through the workflow pages in the sidebar → Data, Curate, Train, Generate, Export.")
