# 🧪 Ligandra

A **target-agnostic, config-driven computer-aided drug-design (CADD) platform**.
Point it at *any* protein target, and it collects & curates bioactivity data,
trains and benchmarks predictive (QSAR/QSPR) models, fine-tunes pretrained
molecular foundation models, **generates novel candidate molecules optimized for
that target**, scores them (potency + drug-likeness + synthesizability +
novelty), and exports a ranked candidate set — all reproducibly tracked.

Ligandra is the rebuild of a Streamlit prototype into a modular,
extensible library. Every layer is a **registry of plugins behind a small ABC**,
so adding a transformer or diffusion model is *one new class*, never a pipeline
rewrite.

> ⚠️ **Research / early-discovery tool.** Predictions and generated molecules are
> *hypotheses* that require experimental validation. Scores are not ground truth —
> always inspect the reported uncertainty and novelty.

📖 **Docs:** [User Guide](docs/USER_GUIDE.md) · [Developer / Architecture Guide](docs/DEVELOPER_GUIDE.md)

---

## Architecture

```
                 config (Pydantic + YAML)  ← the UI writes it, the CLI runs it
                         │
   data ──▶ curate ──▶ featurize ──▶ predict ──▶ generate ──▶ score ──▶ rank ──▶ report
    │         │            │            │            │           │
 DATA_SOURCES │        FEATURIZERS    MODELS      GENERATORS   SCORERS      (all string-keyed
 (chembl,   (standardize, (ecfp, maccs, (ridge, rf,  (graph_ga,  (potency,     registries of
  local,     dedupe,       lipinski,     svr, gnn,     smiles_lm,  qed, sa,     plugins behind
  bindingdb) unit conv,    rdkit_desc,   foundation)  vae, …)     novelty,      one ABC each)
             pXC50,        graph,                                 druglikeness)
             scaffold      learned)
             split)
```

Each layer depends only on the **interface** of the previous one. The Streamlit
UI and the CLI are both thin shells over `ligandra.pipeline`.

| Layer | ABC | Registry | Built-ins |
|-------|-----|----------|-----------|
| Data | `DataSource` | `DATA_SOURCES` | `chembl`, `local`, `bindingdb` |
| Featurize | `Featurizer` | `FEATURIZERS` | `ecfp`, `maccs`, `rdkit_fp`, `lipinski`, `rdkit_descriptors`, `mordred`, `graph`, `learned`, `padel` |
| Predict | `PredictiveModel` | `MODELS` | `linear`, `ridge`, `lasso`, `random_forest`, `svr`, `xgboost`, `lightgbm`, `gnn`, `foundation` |
| Generate | `Generator` | `GENERATORS` | `graph_ga`, `smiles_lm`, `vae`, `transformer`, `diffusion`‡ |
| Score | `ScoringFunction` | `SCORERS` | `potency`, `qed`, `sa`, `druglikeness`, `novelty` |
| Dock | `DockingEngine` | `DOCKERS` | `vina`† |
| Pretrained | `PretrainedCheckpoint` | `PRETRAINED` | `reference-char`, `chemberta`, `molformer`, `smiles-lm`, `diffsbdd`* |

`*` = registered extension point wired end-to-end; heavy/GPU weights left as a
one-class addition (`diffsbdd` is the pretrained-checkpoint hook for a released
DiffSBDD model).
`†` = fully implemented; a live dock needs the external AutoDock Vina binary plus
a ligand-prep tool (Meeko or Open Babel).
`‡` = the `diffusion` generator is a real, trainable E(3)-equivariant 3D
diffusion model (see below); drug-scale de-novo quality needs GPU pretraining.
`bindingdb` is a live REST source (needs network); `chembl`/`mordred`/`padel`
need their optional deps; `vae`/`transformer`/`smiles_lm`/`foundation` need
`torch` (`pip install -e ".[deep]"`).

---

## Install

```bash
# core (classical models + graph_ga generator; CPU-only, no GPU needed)
pip install -e .

# optional extras
pip install -e ".[chembl]"       # live ChEMBL data source
pip install -e ".[deep]"         # foundation-model transfer learning + SMILES-LM generator
pip install -e ".[boost,ui,dev]" # XGBoost/LightGBM, Streamlit UI, test/lint tooling
```

Requires Python ≥ 3.11 and RDKit (installed via the core dependencies).

---

## Quickstart

### CLI (headless, reproducible)

```bash
# 1. See every available plugin
ligandra list

# 2. Write a starter config, or use the bundled offline example
ligandra init -o experiment.yaml

# 3. Run the whole pipeline on the cached mini-dataset (no network needed)
ligandra run examples/local_experiment.yaml --export candidates.sdf
```

This curates the data, trains a scaffold-split leaderboard, generates
target-optimized molecules, and writes a ranked `candidates.sdf` plus a run
manifest under `runs/`.

### Python API

```python
from ligandra.api import run, load_config

result = run(load_config("examples/local_experiment.yaml"))
print(result.leaderboard.to_dataframe())     # scaffold-split benchmark
print(result.candidates.head())              # ranked de-novo molecules
```

### Streamlit UI

```bash
streamlit run ligandra/ui/app.py
```

Walk the workflow in the sidebar: **Home (config) → Data & Curate → Train &
Benchmark → Fine-tune → Generate & Rank → Export.** Dropdowns are filled from the
registries, so any plugin you add shows up automatically.

### Point it at a real target (ChEMBL)

```yaml
# examples/chembl_experiment.yaml
name: egfr_demo
target:
  source: chembl
  query: EGFR          # any gene name / family / ChEMBL id
  endpoint: IC50        # or Ki, Kd, EC50, %inhibition
```

```bash
pip install -e ".[chembl]"
ligandra run examples/chembl_experiment.yaml
```

No code changes needed to switch targets or endpoints — that coupling is exactly
what this rebuild removes.

---

## Add your own model in ~20 lines

Subclass the ABC, decorate with `@MODELS.register("name")`, and it appears in the
CLI, the API, and the UI dropdown with **zero** orchestration edits:

```python
from ligandra.models.base import MODELS, PredictiveModel
from ligandra.core.types import TaskType
import numpy as np, pickle
from pathlib import Path

@MODELS.register("knn")
class KNNModel(PredictiveModel):
    task = TaskType.REGRESSION
    def __init__(self, k: int = 5):
        from sklearn.neighbors import KNeighborsRegressor
        self._m = KNeighborsRegressor(n_neighbors=k)
    def fit(self, X, y):        self._m.fit(X, y); return self
    def predict(self, X):       return self._m.predict(X)
    def save(self, path):       Path(path).write_bytes(pickle.dumps(self._m))
    @classmethod
    def load(cls, path):
        obj = cls(); obj._m = pickle.loads(Path(path).read_bytes()); return obj
```

Generators (`@GENERATORS.register`), featurizers, scorers, data sources and
docking engines extend the same way. A GNN, transformer, or structure-based
diffusion generator is just another registered subclass.

---

## Transfer learning & the representation-consistency contract (M5 / §7)

The `foundation` model loads a pretrained SMILES transformer (ChemBERTa /
MoLFormer / any newer HF checkpoint — the id is a config string) and either
trains a head on a **frozen encoder** or **fine-tunes** it with discriminative
learning rates. Its encoder is also exposed as the `learned` featurizer, so its
embeddings can feed the classical models too. The fine-tuned checkpoint is saved
and reused as the generative potency objective.

```yaml
models:
  - name: foundation
    params: { checkpoint: chemberta, freeze_encoder: false, epochs: 20 }
featurizers:
  - name: learned
    params: { checkpoint: chemberta }   # same checkpoint ⇒ same preprocess_hash
```

**Why this is correct, not just "loss goes down."** A foundation model's
large-scale chemistry lives in its *released weights* — MoLFormer-XL was
pretrained on ~1.1B PubChem/ZINC SMILES, ChemBERTa on tens of millions. So
Ligandra **never re-downloads a corpus**: picking a checkpoint fetches only the
small target set (10²–10⁴ compounds) for fine-tuning/steering. Pretraining and
fine-tuning are separate stages and are never concatenated (§7.1, enforced by a
test that mocks the data source).

The `ligandra.pretrained` package makes the core invariant a *bound contract*
rather than a convention (§7.2):

- Every checkpoint carries a **`PreprocessingSpec`** — tokenizer id, exact
  canonicalizer (a dotted import path), RDKit version, max length, special
  tokens, vocab — and a stable **`preprocess_hash`**.
- `PretrainedCheckpoint.load()` returns `(model, spec)`; the `learned` featurizer
  encodes molecules **only** through that spec's `preprocess()`, never a generic
  tokenizer.
- `assert_consistent_preprocessing(...)` refuses to mix pipelines: the fine-tune
  set, later inference, and any generator reusing the model must share one
  `preprocess_hash`, else it raises `PreprocessingMismatchError`.
- **Leakage & applicability guards** (§7.4): `overlap_with_pretraining` /
  `dedupe_against` (by InChIKey) and `applicability_domain_flags` for target
  molecules far from the modelled distribution.

The transfer *path* depends on the checkpoint `family`: `predictive_transformer`
(encoder + head), `generative_lm` (fine-tune / RL-steer a prior), and
`structure_diffusion` (condition on a 3D pocket at inference — ties to docking).

The built-in **`reference-char`** checkpoint is fully self-contained (a
deterministic SMILES tokenizer + feature-hashing encoder), so the entire
contract — and a working `learned` featurizer — runs with **no `torch` /
`transformers`**. Point at `chemberta` / `molformer` for real embeddings.

### Add your own checkpoint

```python
from ligandra.pretrained.base import PRETRAINED, PretrainedCheckpoint
from ligandra.pretrained.checkpoints import HFPretrainedCheckpoint

@PRETRAINED.register("my-new-encoder")
class MyEncoder(HFPretrainedCheckpoint):
    family = "predictive_transformer"
    def __init__(self, max_length: int = 256):
        super().__init__("org/newer-chem-model", max_length=max_length)
```

It now appears wherever `learned` / `foundation` accept a `checkpoint=` — no
orchestration edits.

---

## Generative engines (M6)

All generators share one `Generator` interface (`sample` / `optimize_for_target`)
and are interchangeable via config:

- **`graph_ga`** — Bemis–Murcko/graph genetic algorithm. Zero extra deps, the
  default; runs anywhere.
- **`smiles_lm`** — char-level GRU SMILES policy + REINFORCE (REINVENT-style).
- **`vae`** — char-level SMILES VAE with **continuous latent-space optimization**
  (encode actives → evolve latents by the objective).
- **`transformer`** — GPT-style causal Transformer LM, pretrained on actives then
  REINFORCE-steered.
- **`diffusion`** — a genuine **E(3)-equivariant 3D diffusion** model
  (EDM/DiffSBDD-style): an EGNN denoiser runs a DDPM over all-atom point clouds
  (coordinates + atom types), molecules are reconstructed by RDKit bond
  perception, and a **protein pocket** conditions generation as *fixed context
  nodes*:

  ```yaml
  generator:
    name: diffusion
    params: { pocket: receptor_pocket.pdb, timesteps: 100, epochs: 100 }
  ```

  The denoiser is verifiably equivariant and trains without instability; note
  that drug-scale *de-novo* validity needs GPU-scale pretraining on protein–
  ligand complexes (CrossDocked/PDBbind) — on small CPU training it runs, trains,
  samples, and reconstructs, but produces lower-quality molecules than a
  pretrained checkpoint would. `vae`/`transformer`/`diffusion`/`smiles_lm` need
  `torch` (`pip install -e ".[deep]"`).

---

## Reproducibility & tracking (M10)

Every run seeds all RNGs, hashes the dataset, and writes a manifest
(`runs/<name>_<timestamp>/manifest.json`) with the full config, params, metrics
and artifact paths (`curated.csv`, `leaderboard.json`, `candidates.csv`). Set
`tracking.backend: mlflow` to log the same information to MLflow.

---

## Tests

```bash
pip install -e ".[dev,ui]"
pytest            # curation math, registry wiring, featurizer/model round-trips,
                  # generator beats-random, and an end-to-end integration run
```

---

## Guardrails & non-goals

Ligandra is for **early discovery**. It enforces validity/uniqueness/novelty
checks and de-duplicates generated molecules against the training set so
"generated" candidates are not memorized actives. Out of scope for v1 (interfaces
left ready): wet-lab integration, retrosynthesis planning, FEP, and multi-target
polypharmacology.
