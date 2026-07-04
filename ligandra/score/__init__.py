"""Scoring & multi-objective ranking layer."""

# Import concrete scorers so their @register decorators run.
from ligandra.score import scorers as _scorers  # noqa: F401,E402
from ligandra.score.base import (
    SCORERS,
    ScoringFunction,
    WeightedSumObjective,
    pareto_front,
)


def build_scorer(name: str, **params) -> ScoringFunction:
    return SCORERS.create(name, **params)


__all__ = [
    "SCORERS",
    "ScoringFunction",
    "WeightedSumObjective",
    "pareto_front",
    "build_scorer",
]
