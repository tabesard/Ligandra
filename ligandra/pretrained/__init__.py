"""Pretrained-checkpoint registry & preprocessing contract (Section 7).

This package is where transfer learning is made *correct*:

* :class:`~ligandra.pretrained.spec.PreprocessingSpec` binds a checkpoint's
  exact input pipeline and yields a stable ``preprocess_hash``;
* :class:`~ligandra.pretrained.base.PretrainedCheckpoint` + the ``PRETRAINED``
  registry expose released checkpoints (``load()`` returns ``(model, spec)``);
* :func:`~ligandra.pretrained.base.assert_consistent_preprocessing` enforces
  that the fine-tune set, later inference, and any reusing generator share one
  ``preprocess_hash``;
* :mod:`~ligandra.pretrained.leakage` provides the data-leakage / applicability
  guards.

Importing this package registers the built-in checkpoints.
"""

from __future__ import annotations

# Import concrete checkpoints so their @register decorators run.
from ligandra.pretrained import checkpoints as _checkpoints  # noqa: F401,E402
from ligandra.pretrained.base import (
    PRETRAINED,
    PreprocessingMismatchError,
    PretrainedCheckpoint,
    assert_consistent_preprocessing,
)
from ligandra.pretrained.leakage import (
    OverlapReport,
    applicability_domain_flags,
    dedupe_against,
    inchikey,
    overlap_with_pretraining,
)
from ligandra.pretrained.spec import EncodedMol, PreprocessingSpec


def build_checkpoint(name: str, **params) -> PretrainedCheckpoint:
    """Instantiate a registered checkpoint (used by featurizers/models/UI)."""
    return PRETRAINED.create(name, **params)


__all__ = [
    "PRETRAINED",
    "PretrainedCheckpoint",
    "PreprocessingSpec",
    "EncodedMol",
    "PreprocessingMismatchError",
    "assert_consistent_preprocessing",
    "build_checkpoint",
    "OverlapReport",
    "overlap_with_pretraining",
    "dedupe_against",
    "applicability_domain_flags",
    "inchikey",
]
