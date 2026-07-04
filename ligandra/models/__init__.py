"""Predictive-modeling layer. Importing registers the built-in models."""

# Import concrete models so their @register decorators run.
from ligandra.models import classical as _classical  # noqa: F401,E402
from ligandra.models import foundation as _foundation  # noqa: F401,E402
from ligandra.models import gnn as _gnn  # noqa: F401,E402
from ligandra.models.base import (
    MODELS,
    Leaderboard,
    LeaderboardEntry,
    PredictiveModel,
)


def build_model(name: str, **params) -> PredictiveModel:
    return MODELS.create(name, **params)


__all__ = [
    "MODELS",
    "PredictiveModel",
    "Leaderboard",
    "LeaderboardEntry",
    "build_model",
]
