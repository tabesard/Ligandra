"""Featurization layer. Importing registers all built-in featurizers."""

# Import concrete featurizers so their @register decorators run.
from ligandra.featurize import descriptors as _descriptors  # noqa: F401,E402
from ligandra.featurize import fingerprints as _fingerprints  # noqa: F401,E402
from ligandra.featurize import graph as _graph  # noqa: F401,E402
from ligandra.featurize import learned as _learned  # noqa: F401,E402
from ligandra.featurize.base import FEATURIZERS, Featurizer


def build_featurizer(name: str, **params) -> Featurizer:
    """Convenience constructor used by the pipeline/UI."""
    return FEATURIZERS.create(name, **params)


__all__ = ["FEATURIZERS", "Featurizer", "build_featurizer"]
