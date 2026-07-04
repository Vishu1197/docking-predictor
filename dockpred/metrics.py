"""Regression metrics used for benchmarking, calibration and evaluation."""

from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties averaged) -- avoids a scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def regression_metrics(y_true, y_pred) -> dict:
    """Full metric suite in real docking-score units."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    var_y = float(np.var(y_true))
    expl = float(1.0 - np.var(err) / var_y) if var_y > 0 else float("nan")

    if y_true.std() > 0 and y_pred.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman = float(np.corrcoef(_rankdata(y_true), _rankdata(y_pred))[0, 1])
    else:
        pearson = spearman = float("nan")

    return {
        "RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2,
        "Pearson": pearson, "Spearman": spearman, "ExplainedVar": expl,
    }
