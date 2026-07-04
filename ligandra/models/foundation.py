"""Transfer learning from pretrained molecular foundation models (M5).

Loads a pretrained SMILES transformer (ChemBERTa / MoLFormer / any newer HF
checkpoint — the source string is configurable) and trains a regression or
classification head for the user's endpoint.  Two transfer modes:

* ``freeze_encoder=True``  — frozen encoder + trainable head (fast, small data);
* ``freeze_encoder=False`` — full/partial fine-tuning with a lower encoder LR
  (discriminative learning rates).

The fine-tuned checkpoint is saved so it can be registered in the model registry
and reused by the generative objectives (M6).  Requires ``torch`` +
``transformers``; a clear error is raised if they are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ligandra.core.types import TaskType
from ligandra.models.base import MODELS, PredictiveModel


def _require_torch():
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The foundation model needs `torch` and `transformers`. Install "
            "them with `pip install torch transformers`."
        ) from exc


@MODELS.register("foundation")
class FoundationModel(PredictiveModel):
    """Pretrained transformer encoder + a small MLP head."""

    task = TaskType.REGRESSION

    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        checkpoint: str | None = None,
        freeze_encoder: bool = True,
        epochs: int = 20,
        lr: float = 1e-3,
        encoder_lr: float = 1e-5,
        batch_size: int = 32,
        max_length: int = 256,
        device: str | None = None,
        task: TaskType = TaskType.REGRESSION,
    ) -> None:
        # Resolve the preprocessing contract (Section 7.2). ``checkpoint`` keys
        # into the PRETRAINED registry; otherwise ``model_name`` is wrapped as a
        # raw Hugging Face checkpoint. Either way we bind a PreprocessingSpec and
        # its preprocess_hash, so the fine-tune set, later inference and any
        # generator reusing this model are provably encoded the same way.
        from ligandra.pretrained import PRETRAINED
        from ligandra.pretrained.checkpoints import HFPretrainedCheckpoint

        if checkpoint is not None:
            self._ckpt = PRETRAINED.create(checkpoint)
            self.model_name = getattr(self._ckpt, "model_id", checkpoint)
        else:
            self._ckpt = HFPretrainedCheckpoint(model_name, max_length=max_length)
            self.model_name = model_name
        self.checkpoint = checkpoint
        self._spec = self._ckpt.spec()
        self.preprocess_hash: str = self._spec.hash()

        self.freeze_encoder = freeze_encoder
        self.epochs = epochs
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.batch_size = batch_size
        # a chosen checkpoint knows its own padding length; else use the arg
        self.max_length = self._spec.max_length or max_length
        self.device = device
        self.task = task
        self._encoder = None
        self._tokenizer = None
        self._head = None

    @property
    def spec(self):
        """The bound :class:`PreprocessingSpec` for this model's checkpoint."""
        return self._spec

    # NOTE: for this model, ``X`` is the list of SMILES strings, not a feature
    # matrix — the tokenizer is the featurizer. The pipeline passes SMILES when
    # the model declares ``consumes_smiles = True``.
    consumes_smiles = True

    def _build(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
        if self.freeze_encoder:
            for p in self._encoder.parameters():
                p.requires_grad = False
        hidden = self._encoder.config.hidden_size
        self._head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
        ).to(self.device)

    def _canonicalize(self, smiles_batch):
        """Standardize each SMILES through the checkpoint's own canonicalizer."""
        out = []
        for s in smiles_batch:
            c = self._spec.canonicalize(s)
            out.append(c if c is not None else s)
        return out

    def _encode(self, smiles_batch):
        # Honor the preprocessing contract: same standardization the checkpoint's
        # spec pins, then the checkpoint's own tokenizer.
        enc = self._tokenizer(
            self._canonicalize(smiles_batch),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self._encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

    def fit(self, X, y) -> PredictiveModel:
        _require_torch()
        import torch

        if self._encoder is None:
            self._build()
        smiles = list(X)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1, 1).to(self.device)

        enc_params = [p for p in self._encoder.parameters() if p.requires_grad]
        groups = [{"params": self._head.parameters(), "lr": self.lr}]
        if enc_params:
            groups.append({"params": enc_params, "lr": self.encoder_lr})
        opt = torch.optim.AdamW(groups)
        loss_fn = (
            torch.nn.MSELoss() if self.task == TaskType.REGRESSION else torch.nn.BCEWithLogitsLoss()
        )

        self._encoder.train() if not self.freeze_encoder else self._encoder.eval()
        self._head.train()
        n = len(smiles)
        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                batch = [smiles[i] for i in idx]
                target = y_t[idx]
                opt.zero_grad()
                ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
                with ctx:
                    feats = self._encode(batch)
                if self.freeze_encoder:
                    feats = feats.detach()
                pred = self._head(feats)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
        return self

    def finetune(self, pretrained: str, X, y) -> PredictiveModel:
        """Load ``pretrained`` (hub id or local path) then fine-tune on (X, y)."""
        from ligandra.pretrained.checkpoints import HFPretrainedCheckpoint

        self.model_name = pretrained
        # Rebind the contract to the new checkpoint so preprocess_hash tracks it.
        self._ckpt = HFPretrainedCheckpoint(pretrained, max_length=self.max_length)
        self._spec = self._ckpt.spec()
        self.preprocess_hash = self._spec.hash()
        self.freeze_encoder = False
        return self.fit(X, y)

    def predict(self, X) -> np.ndarray:
        import torch

        self._encoder.eval()
        self._head.eval()
        smiles = list(X)
        preds = []
        with torch.no_grad():
            for s in range(0, len(smiles), self.batch_size):
                batch = smiles[s : s + self.batch_size]
                logits = self._head(self._encode(batch))
                if self.task == TaskType.CLASSIFICATION:
                    logits = torch.sigmoid(logits)
                preds.append(logits.cpu().numpy().ravel())
        return np.concatenate(preds) if preds else np.empty(0)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._encoder.save_pretrained(path / "encoder")
        self._tokenizer.save_pretrained(path / "encoder")
        torch.save(
            {"head": self._head.state_dict(), "config": self._config()},
            path / "head.pt",
        )

    def _config(self) -> dict:
        return {
            "model_name": self.model_name,
            "freeze_encoder": self.freeze_encoder,
            "task": self.task.value,
            "max_length": self.max_length,
        }

    @classmethod
    def load(cls, path: str | Path) -> PredictiveModel:
        _require_torch()
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        blob = torch.load(path / "head.pt", map_location="cpu")
        cfg = blob["config"]
        obj = cls(model_name=str(path / "encoder"), task=TaskType(cfg["task"]))
        obj.freeze_encoder = cfg["freeze_encoder"]
        obj.max_length = cfg["max_length"]
        obj.device = "cuda" if torch.cuda.is_available() else "cpu"
        obj._tokenizer = AutoTokenizer.from_pretrained(path / "encoder")
        obj._encoder = AutoModel.from_pretrained(path / "encoder").to(obj.device)
        hidden = obj._encoder.config.hidden_size
        obj._head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
        ).to(obj.device)
        obj._head.load_state_dict(blob["head"])
        return obj
