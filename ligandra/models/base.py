"""Predictive-model contract, registry and leaderboard.

Every model — a linear regressor, a GNN, or a fine-tuned transformer — presents
the same interface, so the pipeline trains and benchmarks them identically and a
new model is a one-class addition.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ligandra.core.registry import Registry
from ligandra.core.types import TaskType
from ligandra.models.metrics import compute_metrics

#: Registry of predictive models. Add with ``@MODELS.register("name")``.
MODELS: Registry[PredictiveModel] = Registry("model")


class PredictiveModel(ABC):
    """Base class for QSAR/QSPR models."""

    task: TaskType = TaskType.REGRESSION

    @abstractmethod
    def fit(self, X, y) -> PredictiveModel:
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)``.  Default: point prediction with zero std.

        Ensemble models (e.g. RandomForest) override this with a real estimate.
        """
        pred = np.asarray(self.predict(X), dtype=float)
        return pred, np.zeros_like(pred)

    def finetune(self, pretrained: str, X, y) -> PredictiveModel:
        """Transfer-learning hook. Only deep models implement it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support finetuning from a "
            f"pretrained checkpoint."
        )

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> PredictiveModel:
        ...

    # -- shared evaluation ----------------------------------------------
    def evaluate(self, X, y_true) -> dict[str, float]:
        y_pred = self.predict(X)
        y_score = None
        if self.task == TaskType.CLASSIFICATION and hasattr(self, "predict_proba"):
            try:
                y_score = self.predict_proba(X)  # type: ignore[attr-defined]
            except Exception:
                y_score = None
        return compute_metrics(self.task, y_true, y_pred, y_score)


@dataclass
class LeaderboardEntry:
    model_name: str
    featurizer: str
    metrics: dict[str, float]
    checkpoint: str | None = None
    extra: dict = field(default_factory=dict)


class Leaderboard:
    """Collects held-out metrics for every trained model and ranks them."""

    def __init__(self, task: TaskType, primary_metric: str | None = None) -> None:
        self.task = task
        self.primary_metric = primary_metric or (
            "R2" if task == TaskType.REGRESSION else "ROC_AUC"
        )
        self.entries: list[LeaderboardEntry] = []

    def add(self, entry: LeaderboardEntry) -> None:
        self.entries.append(entry)

    def to_dataframe(self):
        import pandas as pd

        rows = []
        for e in self.entries:
            row = {"model": e.model_name, "featurizer": e.featurizer, **e.metrics}
            if e.checkpoint:
                row["checkpoint"] = e.checkpoint
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty and self.primary_metric in df.columns:
            higher_better = self.primary_metric not in {"MSE", "RMSE", "MAE"}
            df = df.sort_values(
                self.primary_metric, ascending=not higher_better
            ).reset_index(drop=True)
        return df

    def best(self) -> LeaderboardEntry | None:
        df = self.to_dataframe()
        if df.empty:
            return None
        return self.entries[
            next(
                i
                for i, e in enumerate(self.entries)
                if e.model_name == df.iloc[0]["model"]
                and e.featurizer == df.iloc[0]["featurizer"]
            )
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([e.__dict__ for e in self.entries], indent=2), encoding="utf-8"
        )
