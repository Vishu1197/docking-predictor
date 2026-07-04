"""Final hold-out evaluation on the experimental docking scores.

Scores ``data/test/test.csv`` (which is *never* seen during training, tuning,
feature selection or model selection) with the frozen ensemble and produces:

* ``data/test/test_predicted.csv`` -- the original file byte-for-byte with a
  single ``predicted_score`` column inserted immediately after ``score1``;
* the full regression metric suite on the unseen set;
* predicted-vs-experimental scatter, residual, and error-distribution plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dockpred import config
from dockpred.ensemble import EnsemblePredictor
from dockpred.metrics import regression_metrics


def evaluate_holdout(
    test_path: str | Path | None = None,
    manifest: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    target_column: str = config.TARGET_COLUMN,
    device: str = "cpu",
) -> dict:
    test_path = Path(test_path) if test_path else config.PROJECT_ROOT / "data" / "test" / "test.csv"
    outputs_dir = Path(outputs_dir) if outputs_dir else config.PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    predictor = EnsemblePredictor.from_manifest(manifest, device=device)
    df = pd.read_csv(test_path)
    n_rows = len(df)

    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in {test_path}")

    y_true = df[target_column].to_numpy(dtype=np.float64)
    result = predictor.predict_frame(df)
    y_pred = result.ensemble

    # ---- write test_predicted.csv (predicted_score right after score1) ----
    out_df = df.copy()
    insert_at = list(out_df.columns).index(target_column) + 1
    out_df.insert(insert_at, "predicted_score", y_pred)
    out_csv = test_path.parent / "test_predicted.csv"
    out_df.to_csv(out_csv, index=False)

    # ---- metrics ----
    metrics = regression_metrics(y_true, y_pred)

    # ---- plots ----
    _plots(y_true, y_pred, outputs_dir)

    # ---- summary + residual stats ----
    resid = y_pred - y_true
    summary = {
        "test_path": str(test_path),
        "n_rows": int(n_rows),
        "deployed_model": predictor.manifest.get("deployed"),
        "members": predictor.members,
        "metrics": metrics,
        "residual_stats": {
            "mean": float(resid.mean()), "std": float(resid.std()),
            "min": float(resid.min()), "max": float(resid.max()),
            "p05": float(np.percentile(resid, 5)),
            "p50": float(np.percentile(resid, 50)),
            "p95": float(np.percentile(resid, 95)),
        },
        "output_csv": str(out_csv),
    }
    with open(outputs_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- verification ----
    cols = list(out_df.columns)
    assert len(out_df) == n_rows, "row count changed!"
    assert cols.index("predicted_score") == cols.index(target_column) + 1
    assert [c for c in cols if c != "predicted_score"] == list(df.columns)

    return summary


def _plots(y_true: np.ndarray, y_pred: np.ndarray, outputs_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid = y_pred - y_true
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    # scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
    ax.set_xlabel("Experimental docking score (kcal/mol)")
    ax.set_ylabel("Predicted docking score (kcal/mol)")
    ax.set_title("Predicted vs Experimental")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "test_scatter.png", dpi=130)
    plt.close(fig)

    # residual plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, resid, s=8, alpha=0.35, edgecolors="none")
    ax.axhline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Predicted docking score (kcal/mol)")
    ax.set_ylabel("Residual (pred - exp)")
    ax.set_title("Residual plot")
    fig.tight_layout()
    fig.savefig(outputs_dir / "test_residuals.png", dpi=130)
    plt.close(fig)

    # error distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(resid, bins=50, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Residual (pred - exp)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    fig.tight_layout()
    fig.savefig(outputs_dir / "test_error_hist.png", dpi=130)
    plt.close(fig)
