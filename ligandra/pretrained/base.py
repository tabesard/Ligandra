"""Pretrained-checkpoint contract and registry (Section 7.5).

A :class:`PretrainedCheckpoint` names a released molecular foundation model and
carries its :class:`~ligandra.pretrained.spec.PreprocessingSpec`.  ``load()``
returns ``(model, spec)``; the transfer path a checkpoint supports depends on its
``family`` (Section 7.3):

* ``predictive_transformer`` (ChemBERTa/MoLFormer): encoder + head; transfer by
  frozen-head or end-to-end fine-tune.  Its encoder is also exposed as a
  :class:`~ligandra.featurize.learned.LearnedFeaturizer`.
* ``generative_lm`` (SMILES/SELFIES LM): fine-tune / RL-steer on the target's
  actives to bias generation.
* ``structure_diffusion`` (DiffSBDD/TargetDiff/Pocket2Mol): transfer primarily by
  *conditioning at inference* on the target's 3D pocket.

Crucially, the millions of pretraining molecules live in the released weights —
``transfer`` only ever consumes the small target set it is handed (Section 7.1).
No checkpoint fetches a corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ligandra.core.registry import Registry
from ligandra.pretrained.spec import PreprocessingSpec

#: Registry of pretrained checkpoints. Add with ``@PRETRAINED.register("name")``.
PRETRAINED: Registry[PretrainedCheckpoint] = Registry("pretrained checkpoint")

#: The transfer modes a checkpoint may support (Section 7.3).
TRANSFER_MODES = frozenset({"frozen_head", "finetune", "rl_steer", "condition"})


class PreprocessingMismatchError(RuntimeError):
    """Raised when two representation pipelines disagree (different hashes)."""


class PretrainedCheckpoint(ABC):
    """A released foundation-model checkpoint + its preprocessing contract."""

    #: registry key (also set by the registry decorator)
    name: str = "checkpoint"
    #: "predictive_transformer" | "generative_lm" | "structure_diffusion"
    family: str = "predictive_transformer"

    @abstractmethod
    def spec(self) -> PreprocessingSpec:
        """Return this checkpoint's bound :class:`PreprocessingSpec`."""

    @abstractmethod
    def load(self) -> tuple[Any, PreprocessingSpec]:
        """Load and return ``(model, spec)``.

        ``model`` exposes ``embed(EncodedMol) -> np.ndarray`` for encoder-style
        checkpoints.  Loading must **not** fetch any external molecule corpus.
        """

    @property
    def preprocess_hash(self) -> str:
        return self.spec().hash()

    def transfer(self, target_set, mode: str = "frozen_head", **kwargs) -> Any:
        """Transfer onto ``target_set`` (the small target-specific set only).

        Subclasses override this.  The base implementation validates the mode and
        the no-corpus contract, then defers.
        """
        if mode not in TRANSFER_MODES:
            raise ValueError(
                f"Unknown transfer mode {mode!r}. Choose from {sorted(TRANSFER_MODES)}."
            )
        raise NotImplementedError(
            f"{type(self).__name__} (family={self.family!r}) does not implement "
            f"transfer(mode={mode!r}) yet."
        )


def assert_consistent_preprocessing(*objects: Any) -> str:
    """Assert every object shares one ``preprocess_hash``; return it.

    Accepts anything exposing a ``preprocess_hash`` attribute or a ``hash()``
    method (specs, featurizers, models).  This is the guard that enforces the
    core invariant across the fine-tune set, later inference, and any generator
    that reuses the model (Section 7.2 / acceptance criteria).
    """
    hashes: list[tuple[str, str]] = []
    for obj in objects:
        h = _extract_hash(obj)
        if h is None:
            raise TypeError(f"{obj!r} exposes no preprocess_hash.")
        hashes.append((getattr(obj, "__class__", type(obj)).__name__, h))
    if not hashes:
        raise ValueError("assert_consistent_preprocessing needs at least one object.")
    first = hashes[0][1]
    mismatched = [(n, h) for n, h in hashes if h != first]
    if mismatched:
        raise PreprocessingMismatchError(
            "Representation-pipeline mismatch — refusing to mix checkpoints/"
            f"tokenizers. Expected preprocess_hash={first}; got {mismatched}. "
            "Downstream molecules must be encoded exactly as the checkpoint was "
            "pretrained (Section 7.2)."
        )
    return first


def _extract_hash(obj: Any) -> str | None:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "preprocess_hash"):
        return str(obj.preprocess_hash)
    if isinstance(obj, PreprocessingSpec):
        return obj.hash()
    h = getattr(obj, "hash", None)
    if callable(h):
        return str(h())
    return None
