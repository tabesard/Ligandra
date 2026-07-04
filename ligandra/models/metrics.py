"""Regression and classification metrics used by the leaderboard."""

from __future__ import annotations

import numpy as np

from ligandra.core.types import TaskType


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    out = {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    # Pearson is undefined for constant predictions.
    if np.std(y_pred) > 0 and np.std(y_true) > 0 and len(y_true) > 1:
        out["Pearson"] = float(pearsonr(y_true, y_pred)[0])
    else:
        out["Pearson"] = float("nan")
    return out


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, float] = {"MCC": float(matthews_corrcoef(y_true, y_pred))}
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
        out["PR_AUC"] = float(average_precision_score(y_true, y_score))
    else:
        out["ROC_AUC"] = float("nan")
        out["PR_AUC"] = float("nan")
    return out


def compute_metrics(task: TaskType, y_true, y_pred, y_score=None) -> dict[str, float]:
    if task == TaskType.CLASSIFICATION:
        return classification_metrics(y_true, y_pred, y_score)
    return regression_metrics(y_true, y_pred)
