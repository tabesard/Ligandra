"""Concrete pretrained checkpoints (the ``PRETRAINED`` registry contents).

Two flavours:

``reference-char``
    A **self-contained** checkpoint that needs no ``torch``/``transformers``.
    Its "model" is a deterministic feature-hashing encoder over the spec's token
    ids.  It exists so the whole preprocessing-consistency contract is runnable,
    testable, and demonstrable on a bare interpreter — and so CPU-only users have
    a working ``LearnedFeaturizer`` out of the box.

``chemberta`` / ``molformer`` / ``smiles-lm`` / ``diffsbdd``
    Thin wrappers around real released checkpoints.  ``spec()`` (and therefore
    the ``preprocess_hash``) is available offline; ``load()`` needs the heavy
    optional deps and raises a clear error if they are missing.  The model id is
    configurable so newer checkpoints drop in without code changes.

None of these fetch a molecule corpus — the pretraining data lives in the
released weights (Section 7.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ligandra.pretrained.base import PRETRAINED, PretrainedCheckpoint
from ligandra.pretrained.spec import (
    DEFAULT_SMILES_VOCAB,
    DEFAULT_SPECIAL_TOKENS,
    EncodedMol,
    PreprocessingSpec,
)


def _rdkit_version() -> str:
    try:
        import rdkit

        return rdkit.__version__
    except Exception:  # pragma: no cover - RDKit optional
        return ""


# --------------------------------------------------------------------------
# Deterministic, dependency-free encoder for the reference checkpoint.
# --------------------------------------------------------------------------
class HashingEncoder:
    """A reproducible token-id → embedding map (feature hashing).

    Not a learned model — a *stand-in* whose only jobs are to (a) be byte-for-byte
    deterministic given identical ``input_ids`` and (b) require nothing beyond
    NumPy, so the representation-consistency contract can be exercised anywhere.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, encoded: EncodedMol) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tid, m in zip(encoded.input_ids, encoded.attention_mask):
            if not m:
                continue  # skip padding
            bucket = (tid * 2654435761) % self.dim  # Knuth multiplicative hash
            sign = 1.0 if (tid % 2 == 0) else -1.0
            v[bucket] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v


class _TransferredHead:
    """A tiny frozen-encoder + linear-head transfer result (reference family)."""

    def __init__(self, encoder: HashingEncoder, spec: PreprocessingSpec) -> None:
        self.encoder = encoder
        self.spec = spec
        self.preprocess_hash = spec.hash()
        self._head: Any = None
        self.n_train = 0

    def _embed_smiles(self, smiles_list) -> np.ndarray:
        return np.vstack([self.encoder.embed(self.spec.preprocess(s)) for s in smiles_list])

    def fit(self, smiles_list, y) -> _TransferredHead:
        from sklearn.linear_model import Ridge

        X = self._embed_smiles(smiles_list)
        self._head = Ridge().fit(X, np.asarray(y, dtype=float))
        self.n_train = len(smiles_list)
        return self

    def predict(self, smiles_list) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("Call fit() before predict().")
        return self._head.predict(self._embed_smiles(smiles_list))


@PRETRAINED.register("reference-char")
class ReferenceCharCheckpoint(PretrainedCheckpoint):
    """Self-contained reference checkpoint (no heavy deps)."""

    name = "reference-char"
    family = "predictive_transformer"

    def __init__(self, dim: int = 64, max_length: int = 128) -> None:
        self.dim = dim
        self.max_length = max_length

    def spec(self) -> PreprocessingSpec:
        return PreprocessingSpec(
            tokenizer_id="char:smiles-v1",
            representation="smiles",
            canonicalizer="ligandra.curate.standardize.standardize_smiles",
            rdkit_version=_rdkit_version(),
            max_length=self.max_length,
            special_tokens=dict(DEFAULT_SPECIAL_TOKENS),
            backend="char",
            vocab=DEFAULT_SMILES_VOCAB,
            normalization="canonical",
        )

    def load(self) -> tuple[HashingEncoder, PreprocessingSpec]:
        # No download, no corpus — the "weights" are the deterministic hash.
        return HashingEncoder(self.dim), self.spec()

    def transfer(self, target_set, mode: str = "frozen_head", **kwargs) -> _TransferredHead:
        if mode not in {"frozen_head", "finetune"}:
            raise ValueError(f"{self.name} supports 'frozen_head'/'finetune', not {mode!r}.")
        encoder, spec = self.load()
        head = _TransferredHead(encoder, spec)
        y_key = kwargs.get("y_key", "y")
        smiles = [m.canonical_smiles or m.smiles for m in target_set]
        y = kwargs.get("y")
        if y is None:
            y = [m.props.get(y_key) for m in target_set]
        return head.fit(smiles, y)


# --------------------------------------------------------------------------
# Real released checkpoints (heavy deps needed only for load()).
# --------------------------------------------------------------------------
class HFPretrainedCheckpoint(PretrainedCheckpoint):
    """Generic Hugging Face encoder checkpoint.

    ``spec()`` / ``preprocess_hash`` are available offline; ``load()`` needs
    ``torch`` + ``transformers``.
    """

    family = "predictive_transformer"

    def __init__(
        self,
        model_id: str,
        representation: str = "smiles",
        max_length: int = 256,
        canonicalizer: str = "ligandra.curate.standardize.standardize_smiles",
    ) -> None:
        self.model_id = model_id
        self.name = model_id
        self.representation = representation
        self.max_length = max_length
        self.canonicalizer = canonicalizer

    def spec(self) -> PreprocessingSpec:
        return PreprocessingSpec(
            tokenizer_id=self.model_id,
            representation=self.representation,
            canonicalizer=self.canonicalizer,
            rdkit_version=_rdkit_version(),
            max_length=self.max_length,
            special_tokens=dict(DEFAULT_SPECIAL_TOKENS),
            backend="hf",
            normalization="canonical",
        )

    def load(self):  # pragma: no cover - needs torch/transformers
        try:
            import torch  # noqa: F401
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                f"Loading {self.model_id!r} needs `torch` + `transformers`. "
                "Install them with `pip install torch transformers`."
            ) from exc
        spec = self.spec()
        model = AutoModel.from_pretrained(self.model_id).eval()
        return _HFEncoder(model, spec), spec


class _HFEncoder:  # pragma: no cover - needs torch/transformers
    """Wraps an HF encoder to expose ``embed(EncodedMol) -> np.ndarray``."""

    def __init__(self, model, spec: PreprocessingSpec) -> None:
        self.model = model
        self.spec = spec
        self.preprocess_hash = spec.hash()

    def embed(self, encoded: EncodedMol) -> np.ndarray:
        import torch

        ids = torch.tensor([list(encoded.input_ids)])
        mask = torch.tensor([list(encoded.attention_mask)])
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return pooled[0].cpu().numpy().astype(np.float32)


@PRETRAINED.register("chemberta")
class ChemBERTaCheckpoint(HFPretrainedCheckpoint):
    """ChemBERTa-2 (masked-LM pretrained on ~77M PubChem SMILES)."""

    name = "chemberta"
    family = "predictive_transformer"

    def __init__(self, max_length: int = 256) -> None:
        super().__init__("DeepChem/ChemBERTa-77M-MTR", max_length=max_length)
        self.name = "chemberta"


@PRETRAINED.register("molformer")
class MoLFormerCheckpoint(HFPretrainedCheckpoint):
    """MoLFormer-XL (pretrained on ~1.1B PubChem+ZINC SMILES)."""

    name = "molformer"
    family = "predictive_transformer"

    def __init__(self, max_length: int = 202) -> None:
        super().__init__("ibm/MoLFormer-XL-both-10pct", max_length=max_length)
        self.name = "molformer"


@PRETRAINED.register("smiles-lm")
class SmilesLMCheckpoint(HFPretrainedCheckpoint):
    """Generative SMILES language model (REINVENT-style prior).

    Same preprocessing contract; ``family='generative_lm'`` so the transfer path
    is fine-tune / RL-steer on the target's actives rather than an attached head.
    """

    name = "smiles-lm"
    family = "generative_lm"

    def __init__(self, max_length: int = 128) -> None:
        super().__init__("entropy/gpt2_zinc_87m", max_length=max_length)
        self.name = "smiles-lm"


@PRETRAINED.register("diffsbdd")
class DiffSBDDCheckpoint(PretrainedCheckpoint):
    """Structure-based diffusion checkpoint (pocket-conditioned generation).

    Pretrained on protein–ligand complexes (CrossDocked/PDBbind, ~10^5 complexes)
    — *not* a SMILES corpus.  Transfer is by conditioning on a 3D pocket at
    inference (Section 7.3); requires a PDB/predicted structure (ties to M8).
    Wired here as a registry entry; ``load``/``transfer`` are clear stubs.
    """

    name = "diffsbdd"
    family = "structure_diffusion"

    def __init__(self, model_id: str = "DiffSBDD/crossdocked") -> None:
        self.model_id = model_id
        self.name = "diffsbdd"

    def spec(self) -> PreprocessingSpec:
        # 3D-graph representation; SMILES tokenization does not apply, but a spec
        # still pins the canonicalization used for any 2D bookkeeping.
        return PreprocessingSpec(
            tokenizer_id="diffsbdd:3d-graph",
            representation="smiles",
            canonicalizer="ligandra.curate.standardize.standardize_smiles",
            rdkit_version=_rdkit_version(),
            max_length=0,
            backend="char",
            normalization="3d-complex",
        )

    def load(self):  # pragma: no cover - stub extension point
        raise NotImplementedError(
            "DiffSBDD weights + inference are not bundled. Install a structure-"
            "based diffusion backend and implement load() here."
        )

    def transfer(self, target_set, mode: str = "condition", **kwargs):  # pragma: no cover
        if mode != "condition":
            raise ValueError("structure_diffusion transfers by mode='condition'.")
        raise NotImplementedError(
            "Pocket-conditioned generation needs a receptor structure (M8) and a "
            "diffusion backend; wiring is in place, weights are not bundled."
        )
