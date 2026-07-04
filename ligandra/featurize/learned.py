"""Learned embeddings from a pretrained molecular foundation model (M3 + §7).

This is the bridge that lets a foundation model's encoder feed *classical* models
(M4): its embeddings are just another featurizer.  Per the preprocessing-
consistency contract (Section 7.2), this featurizer encodes molecules **only**
through its checkpoint's :class:`~ligandra.pretrained.spec.PreprocessingSpec` —
never a generic tokenizer — and carries the checkpoint's ``preprocess_hash`` so a
mismatched pipeline can be rejected.

By default it uses the self-contained ``reference-char`` checkpoint, so it runs
with no heavy dependencies.  Point it at ``chemberta`` / ``molformer`` (or any
newer registered checkpoint, or a raw Hugging Face id via ``model_name``) to use
real learned embeddings.
"""

from __future__ import annotations

import numpy as np

from ligandra.core.molecule import Molecule
from ligandra.featurize.base import FEATURIZERS, Featurizer
from ligandra.pretrained import PRETRAINED
from ligandra.pretrained.checkpoints import HFPretrainedCheckpoint


@FEATURIZERS.register("learned")
class LearnedFeaturizer(Featurizer):
    """Pooled embeddings from a pretrained checkpoint, via its own preprocessing.

    Parameters
    ----------
    checkpoint:
        Key into the ``PRETRAINED`` registry (e.g. ``"reference-char"``,
        ``"chemberta"``, ``"molformer"``).  Default ``"reference-char"`` needs no
        heavy deps.
    model_name:
        Escape hatch: a raw Hugging Face id, wrapped as an HF checkpoint on the
        fly.  Mutually exclusive with a non-default ``checkpoint``.
    max_length:
        Optional override of the checkpoint's default padding length.
    """

    def __init__(
        self,
        checkpoint: str = "reference-char",
        model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        if model_name is not None:
            kw = {"max_length": max_length} if max_length is not None else {}
            self._ckpt = HFPretrainedCheckpoint(model_name, **kw)
            self.checkpoint_name = model_name
        else:
            kw = {"max_length": max_length} if max_length is not None else {}
            self._ckpt = PRETRAINED.create(checkpoint, **kw)
            self.checkpoint_name = checkpoint

        self._spec = self._ckpt.spec()
        #: the bound contract hash — asserted equal across fine-tune/inference/gen
        self.preprocess_hash: str = self._spec.hash()
        self.batch_size = batch_size
        self._device = device
        self._encoder = None

    @property
    def spec(self):
        """The bound :class:`PreprocessingSpec` (read-only)."""
        return self._spec

    @property
    def cache_key(self) -> str:
        # include the hash so features from a different pipeline never collide
        return f"learned::{self.checkpoint_name}::{self.preprocess_hash}"

    def encode(self, smiles: str):
        """Encode one SMILES through the checkpoint's own preprocessing."""
        return self._spec.preprocess(smiles)

    def _lazy_load(self) -> None:
        if self._encoder is None:
            self._encoder, spec = self._ckpt.load()
            # defensive: the loaded model must agree with the bound spec
            if spec.hash() != self.preprocess_hash:  # pragma: no cover
                raise RuntimeError(
                    "Checkpoint load() returned a spec whose preprocess_hash "
                    "differs from the featurizer's — refusing to featurize."
                )

    def _featurize_one(self, mol: Molecule) -> np.ndarray:
        self._lazy_load()
        encoded = self._spec.preprocess(mol.canonical_smiles or mol.smiles)
        return np.asarray(self._encoder.embed(encoded), dtype=np.float32)

    # ``transform`` (with per-molecule caching) is inherited from Featurizer.
