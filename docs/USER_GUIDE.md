# Ligandra — User Guide

Ligandra is a **target-agnostic, config-driven computer-aided drug-design (CADD)
platform**. Point it at any protein target and it collects & curates bioactivity
data, trains and benchmarks QSAR/QSPR models, fine-tunes pretrained molecular
foundation models, generates novel candidate molecules optimised for that target,
scores them, and exports a ranked candidate set — reproducibly.

> ⚠️ **Research / early-discovery tool.** Predictions and generated molecules are
> *hypotheses* requiring experimental validation. Treat every score as uncertain.

---

## 1. Installation

Requires **Python ≥ 3.11** and RDKit (installed with the core dependencies).

```bash
# core: classical models + graph_ga generator (CPU-only, no GPU needed)
pip install -e .

# optional extras
pip install -e ".[chembl]"        # live ChEMBL data source
pip install -e ".[deep]"          # torch + transformers: foundation model,
                                  #   learned featurizer, VAE / transformer /
                                  #   SMILES-LM / diffusion generators
pip install -e ".[boost]"         # XGBoost / LightGBM
pip install -e ".[ui]"            # Streamlit UI + matplotlib
pip install -e ".[dev]"           # pytest, ruff, mypy
```

Everything runs CPU-only; GPU is optional and only accelerates the torch paths.

---

## 2. The workflow

Ligandra follows one pipeline:

```
data → curate → featurize → predict → generate → score → rank → export
```

Each stage is a plugin behind a small interface, chosen at runtime by a single
typed config. There are three ways to drive it — UI, CLI, and Python API — and
they all execute the **same** config.

### 2a. Streamlit UI (guided)

```bash
streamlit run ligandra/ui/app.py
```

Walk the sidebar pages in order:

1. **Home** — pick the target (source, query/id, endpoint), split, curation
   thresholds, featurizers, models, generator and objectives. Everything writes a
   shared `ExperimentConfig`; download it as `experiment.yaml`.
2. **① Data & Curation** — fetch + curate. Results are cached to `data_cache/` as
   CSVs, so a repeat run reads the file instead of re-hitting the network.
   Download the raw and curated tables.
3. **② Train & Benchmark** — train every model on the scaffold-split leaderboard.
4. **③ Fine-tune** — load a pretrained transformer (ChemBERTa/MoLFormer) and
   fine-tune a head on this target (needs `torch`+`transformers`).
5. **④ Generate & Rank** — de-novo generation optimised against the objective;
   candidates shown as **2D structures with predicted potency**.
6. **⑤ Export** — download the ranked candidates (CSV; SDF via CLI).

Long steps run in a background thread with a live timer, so the tab stays
responsive; identical runs are served instantly from cache. The sidebar has a
**↻ Clear cached results** button to force a recompute.

### 2b. CLI (headless, reproducible)

```bash
ligandra list                                   # every available plugin
ligandra init -o experiment.yaml                # write a starter config
ligandra run experiment.yaml --export out.sdf   # run end-to-end
ligandra run experiment.yaml --no-generate      # train/benchmark only
```

### 2c. Python API

```python
from ligandra.api import run, load_config

result = run(load_config("examples/local_experiment.yaml"))
print(result.leaderboard.to_dataframe())   # scaffold-split benchmark
print(result.candidates.head())            # ranked de-novo molecules
```

---

## 3. Choosing a target

Ligandra is target-agnostic. For the ChEMBL source you can give:

- a **ChEMBL id** (`CHEMBL203`),
- a **UniProt accession** (`P00533` — resolved automatically), or
- a **gene / protein name** (`EGFR`, `Maltase-glucoamylase`).

```yaml
target:
  source: chembl
  query: EGFR          # or a name / family
  target_id: null      # ...or an exact id (leave null to search by name)
  endpoint: IC50       # or Ki, Kd, EC50, %inhibition
```

An empty `target_id` (e.g. from a blank UI box) is treated as "not provided" and
the `query` is used. Transient ChEMBL network drops are retried automatically.

---

## 4. Config reference (essentials)

```yaml
name: my_experiment
task: regression            # or classification
seed: 42

target: { source, query, target_id, endpoint }
curation:
  active_threshold: 1000.0   # nM, ≤ is active
  inactive_threshold: 10000.0
  standardize: true          # RDKit neutralise/strip-salts/canonical-tautomer
split:
  strategy: scaffold         # scaffold (honest) or random
  test_size: 0.2
featurizers: [{ name: ecfp }]           # ecfp/maccs/rdkit_descriptors/lipinski/…
models: [{ name: ridge }, { name: random_forest }]
generator: { name: graph_ga, budget: 200, n_output: 50 }
objectives:                             # combined into a weighted desirability
  - { scorer: potency, weight: 1.0 }
  - { scorer: qed, weight: 0.5 }
  - { scorer: sa, weight: 0.5 }
cache: { enabled: true, dir: data_cache, refresh: false }
tracking: { backend: local, output_dir: runs }
```

Run `ligandra list` for the live names in every registry.

---

## 5. Outputs

Each run writes to `runs/<name>_<timestamp>/`:

- `manifest.json` — full config, dataset hash, params, metrics, artifact paths;
- `curated.csv`, `leaderboard.json`, `candidates.csv`.

Fetched/curated tables are also cached under `data_cache/` for reuse and analysis.

---

## 6. Interpreting results honestly

- The **leaderboard** reports metrics on a held-out **scaffold split** — a
  deliberately hard, honest generalisation estimate. Negative R² on small or
  chemically diverse targets is common and *expected*; it means the model does
  not generalise, not that the tool is broken.
- **Generated molecules** are de-duplicated against training actives and reported
  with validity / uniqueness / novelty plus per-candidate ADMET-style scores.
  They are starting hypotheses, not validated hits.
- The GPU-scale generators (`diffusion`, foundation fine-tuning) produce
  lower-quality output on small CPU training than a pretrained checkpoint would.

See the [Developer Guide](DEVELOPER_GUIDE.md) to add your own model, featurizer,
generator, scorer or data source.
