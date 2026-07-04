# Ligandra — Developer / Architecture Guide

Ligandra's design goal is **extensibility without touching orchestration**:
adding a transformer or diffusion model is one new class + one decorator, and it
appears in the CLI, the API, and the UI automatically.

---

## 1. Layered architecture

```
config (Pydantic + YAML)  ← the UI writes it, the CLI/API run it
   │
 data ─▶ curate ─▶ featurize ─▶ predict ─▶ generate ─▶ score ─▶ rank ─▶ report
```

Each layer depends only on the **interface** of the previous one. The Streamlit
UI and CLI are thin shells over `ligandra.pipeline` / `ligandra.api`.

| Layer | ABC | Registry | Module |
|-------|-----|----------|--------|
| Data | `DataSource` | `DATA_SOURCES` | `ligandra/data` |
| Curate | — | — | `ligandra/curate` |
| Featurize | `Featurizer` | `FEATURIZERS` | `ligandra/featurize` |
| Predict | `PredictiveModel` | `MODELS` | `ligandra/models` |
| Generate | `Generator` | `GENERATORS` | `ligandra/generate` |
| Score | `ScoringFunction` | `SCORERS` | `ligandra/score` |
| Dock | `DockingEngine` | `DOCKERS` | `ligandra/dock` |
| Pretrained | `PretrainedCheckpoint` | `PRETRAINED` | `ligandra/pretrained` |

---

## 2. The registry pattern

Every layer is a string-keyed `Registry[T]`. A plugin registers itself with a
decorator; the sub-package `__init__` imports the concrete modules so the
decorators run on import.

```python
from ligandra.models.base import MODELS, PredictiveModel
from ligandra.core.types import TaskType

@MODELS.register("knn")
class KNNModel(PredictiveModel):
    task = TaskType.REGRESSION
    def __init__(self, k: int = 5):
        from sklearn.neighbors import KNeighborsRegressor
        self._m = KNeighborsRegressor(n_neighbors=k)
    def fit(self, X, y):  self._m.fit(X, y); return self
    def predict(self, X): return self._m.predict(X)
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

`ligandra list` and the UI dropdowns now include `knn` — no orchestration edits.
The same pattern applies to `@FEATURIZERS.register`, `@GENERATORS.register`,
`@SCORERS.register`, `@DATA_SOURCES.register`, `@DOCKERS.register`,
`@PRETRAINED.register`.

### The canonical molecule object

Everything speaks `ligandra.core.molecule.Molecule` / `MoleculeSet`: SMILES +
lazy RDKit mol + optional SELFIES/3D + a per-feature cache, so any featurizer or
model requests what it needs without recomputation.

---

## 3. The interfaces

```python
class DataSource(ABC):
    def search_targets(self, query) -> list[Target]: ...
    def fetch_activities(self, target_id, endpoint, **f) -> pd.DataFrame: ...

class Featurizer(ABC):
    def transform(self, mols: MoleculeSet) -> "Features": ...

class PredictiveModel(ABC):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_with_uncertainty(self, X): ...   # default: (pred, zeros)
    def finetune(self, pretrained, X, y): ...     # transfer-learning hook
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...

class Generator(ABC):
    def sample(self, n) -> MoleculeSet: ...
    def optimize_for_target(self, objective: ScoringFunction, budget) -> MoleculeSet: ...

class ScoringFunction(ABC):
    def __call__(self, mols: MoleculeSet) -> np.ndarray: ...   # normalised [0,1]
```

A model that consumes SMILES directly (e.g. the `foundation` transformer) sets
`consumes_smiles = True`; a GNN sets `consumes_graphs = True`. The pipeline gives
each model the representation it declares, so a SMILES model needs no featurizer.

---

## 4. Transfer learning & the representation-consistency contract

`ligandra/pretrained` is what makes fine-tuning *correct* rather than
loss-goes-down-but-learns-nothing.

- A pretrained checkpoint's chemistry lives in its **released weights**; Ligandra
  fetches only the small target set for fine-tuning — never a corpus (a test with
  a mocked data source enforces this).
- Each checkpoint carries a **`PreprocessingSpec`** (tokenizer id, exact
  canonicaliser, RDKit version, max length, vocab) and a stable
  **`preprocess_hash`**. Downstream molecules are encoded **only** through that
  spec, and `assert_consistent_preprocessing(...)` refuses to mix pipelines — the
  fine-tune set, later inference, and any generator reusing the model must share
  one hash.
- The built-in `reference-char` checkpoint is fully self-contained (a
  deterministic SMILES tokenizer + feature-hashing encoder), so the whole
  contract runs and is tested with **no `torch`/`transformers`**.
- `ligandra/pretrained/leakage.py` provides InChIKey overlap/dedup and a
  kNN applicability-domain flag.

To add a real checkpoint:

```python
from ligandra.pretrained.base import PRETRAINED
from ligandra.pretrained.checkpoints import HFPretrainedCheckpoint

@PRETRAINED.register("my-encoder")
class MyEncoder(HFPretrainedCheckpoint):
    family = "predictive_transformer"     # or generative_lm / structure_diffusion
    def __init__(self, max_length: int = 256):
        super().__init__("org/newer-chem-model", max_length=max_length)
```

It now works everywhere `learned` / `foundation` accept a `checkpoint=`.

---

## 5. Reproducibility

Global seeding, dataset content hashing, a per-run manifest, and an on-disk CSV
cache keyed on (source, target, endpoint, curation settings). Set
`tracking.backend: mlflow` to log to MLflow instead of the local manifest.

---

## 6. Testing & quality

```bash
pip install -e ".[dev,ui]"
pytest            # ~90 tests: curation math, registry wiring, featurizer/model
                  # round-trips, the §7 representation-consistency contract, the
                  # diffusion engine, background-job UI, and an end-to-end run
ruff check ligandra
```

Torch-dependent tests use `pytest.importorskip("torch")` so the suite stays
portable. Conventions: type hints, Ruff formatting, docstrings on public APIs,
and a test for every new plugin's `fit/predict/save/load` (or equivalent)
round-trip.
