"""Experiment tracking & reproducibility (M10).

A lightweight, dependency-free ``local`` backend writes a per-run manifest
(config, dataset hash, params, metrics, artifact paths) under ``runs/``.  An
optional ``mlflow`` backend logs the same information to MLflow when installed.
Both implement :class:`Tracker`, selected by config.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def hash_dataset(smiles: list[str], values: list[float] | None = None) -> str:
    """Deterministic content hash of a dataset (for reproducibility)."""
    h = hashlib.sha256()
    for i, s in enumerate(sorted(smiles)):
        h.update(s.encode("utf-8"))
        if values is not None:
            h.update(f"{values[i]:.6g}".encode())
    return h.hexdigest()[:16]


class Tracker(ABC):
    @abstractmethod
    def start_run(self, name: str, config: dict) -> Tracker: ...
    @abstractmethod
    def log_params(self, params: dict) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    @abstractmethod
    def log_artifact(self, path: str | Path) -> None: ...
    @abstractmethod
    def end_run(self) -> Path | None: ...


class LocalTracker(Tracker):
    """Writes ``runs/<name>_<timestamp>/manifest.json`` plus copied artifacts."""

    def __init__(self, output_dir: str = "runs") -> None:
        self.output_dir = Path(output_dir)
        self.run_dir: Path | None = None
        self._manifest: dict[str, Any] = {}

    def start_run(self, name: str, config: dict) -> Tracker:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_dir / f"{name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = {
            "name": name,
            "created": stamp,
            "config": config,
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        return self

    def log_params(self, params: dict) -> None:
        self._manifest["params"].update(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._manifest["metrics"].update(metrics)

    def log_artifact(self, path: str | Path) -> None:
        self._manifest["artifacts"].append(str(path))

    def end_run(self) -> Path | None:
        if self.run_dir is None:
            return None
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, default=str), encoding="utf-8"
        )
        return self.run_dir


class MLflowTracker(Tracker):  # pragma: no cover - optional dependency
    """MLflow-backed tracker (used when config.tracking.backend == 'mlflow')."""

    def __init__(self, experiment_name: str = "ligandra") -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)
        self._active = None

    def start_run(self, name: str, config: dict) -> Tracker:
        self._active = self._mlflow.start_run(run_name=name)
        self._mlflow.log_dict(config, "config.json")
        return self

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if _is_num(v)}, step=step
        )

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def end_run(self) -> Path | None:
        self._mlflow.end_run()
        return None


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def build_tracker(backend: str, **kwargs) -> Tracker:
    if backend == "mlflow":
        return MLflowTracker(kwargs.get("experiment_name", "ligandra"))
    return LocalTracker(kwargs.get("output_dir", "runs"))


__all__ = ["Tracker", "LocalTracker", "MLflowTracker", "build_tracker", "hash_dataset"]
