"""Classical QSAR models: linear family, RandomForest, SVR, XGBoost/LightGBM.

Ports the prototype's Linear/Ridge/Lasso and fills in the dead tree/NN tabs with
real, cross-validated implementations behind the shared ``PredictiveModel`` API.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ligandra.core.types import TaskType
from ligandra.models.base import MODELS, PredictiveModel


class _SklearnModel(PredictiveModel):
    """Adapter wrapping any scikit-learn estimator (with optional CV search)."""

    #: hyperparameter grid for GridSearchCV; empty means fit directly
    param_grid: dict = {}
    task = TaskType.REGRESSION

    def __init__(self, cv: int = 5, scoring: str | None = None, **params) -> None:
        self.cv = cv
        self.scoring = scoring
        self.params = params
        self._estimator = None
        self.best_params_: dict = {}

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y) -> PredictiveModel:
        from sklearn.model_selection import GridSearchCV

        est = self._make_estimator()
        n = len(y)
        if self.param_grid and n >= self.cv:
            scoring = self.scoring or (
                "neg_mean_squared_error"
                if self.task == TaskType.REGRESSION
                else "roc_auc"
            )
            search = GridSearchCV(
                est, self.param_grid, cv=min(self.cv, n), scoring=scoring
            )
            search.fit(X, y)
            self._estimator = search.best_estimator_
            self.best_params_ = search.best_params_
        else:
            est.fit(X, y)
            self._estimator = est
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        proba = self._estimator.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "estimator": self._estimator,
                    "task": self.task,
                    "best_params": self.best_params_,
                    "class": type(self).__name__,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> PredictiveModel:
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls()
        obj._estimator = blob["estimator"]
        obj.task = blob["task"]
        obj.best_params_ = blob.get("best_params", {})
        return obj


# --- Linear family (ported from the prototype) ---------------------------
@MODELS.register("linear")
class LinearModel(_SklearnModel):
    def _make_estimator(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**self.params)


@MODELS.register("ridge")
class RidgeModel(_SklearnModel):
    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

    def _make_estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(**self.params)


@MODELS.register("lasso")
class LassoModel(_SklearnModel):
    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

    def _make_estimator(self):
        from sklearn.linear_model import Lasso

        return Lasso(max_iter=10000, **self.params)


# --- Trees / kernels ------------------------------------------------------
@MODELS.register("random_forest")
class RandomForestModel(_SklearnModel):
    param_grid = {"n_estimators": [200], "max_depth": [None, 10, 20]}

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        # n_jobs left at the sklearn default (1): parallel predict in 1.9 emits a
        # spurious joblib warning, and QSAR datasets are small enough not to need it.
        return RandomForestRegressor(random_state=42, **self.params)

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Std across trees = a cheap epistemic-uncertainty / applicability signal."""
        preds = np.stack(
            [est.predict(X) for est in self._estimator.estimators_], axis=0
        )
        return preds.mean(0), preds.std(0)


@MODELS.register("svr")
class SVRModel(_SklearnModel):
    param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}

    def _make_estimator(self):
        from sklearn.svm import SVR

        return SVR(**self.params)


@MODELS.register("xgboost")
class XGBoostModel(_SklearnModel):
    param_grid = {"n_estimators": [300], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}

    def _make_estimator(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `xgboost` to use the xgboost model.") from exc
        return XGBRegressor(random_state=42, n_jobs=-1, **self.params)


@MODELS.register("lightgbm")
class LightGBMModel(_SklearnModel):
    param_grid = {"n_estimators": [300], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]}

    def _make_estimator(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `lightgbm` to use the lightgbm model.") from exc
        return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **self.params)
