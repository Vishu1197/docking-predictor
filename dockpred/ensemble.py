"""The production ensemble predictor.

Loads the calibrated ensemble from ``models/ensemble_manifest.json`` and predicts
docking scores from raw (possibly partial) molecular descriptors. Supports both
weighted-average and Ridge-stacking ensembles, chosen automatically by the
training pipeline. All predictions are in real docking-score units (kcal/mol).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dockpred import config
from dockpred.base_models import load_base_models
from dockpred.data import FeaturePipeline


@dataclass
class PredictionResult:
    """Ensemble output plus per-model breakdown, all in kcal/mol."""

    ensemble: np.ndarray
    per_model: dict[str, np.ndarray]

    def to_frame(self, include_per_model: bool = True) -> pd.DataFrame:
        data = {"predicted_score": self.ensemble}
        if include_per_model:
            for name, preds in self.per_model.items():
                data[f"score_{name}"] = preds
        return pd.DataFrame(data)


class EnsemblePredictor:
    """Load the calibrated ensemble and predict docking scores from descriptors."""

    def __init__(self, manifest: dict, device: str = "cpu"):
        self.manifest = manifest
        self.device = device
        self.pipeline = FeaturePipeline.from_manifest(manifest)
        self.method = manifest.get("ensemble_method", "weighted_average")
        self.members: list[str] = list(manifest["members"])
        self.weights: dict[str, float] = manifest.get("weights") or {}
        self.meta_coef = manifest.get("meta_coef")
        self.meta_intercept = manifest.get("meta_intercept", 0.0)
        self.models = load_base_models(
            manifest["base_models"], self.pipeline.n_features, device=device)

        missing = set(self.members) - set(self.models)
        if missing:
            raise ValueError(f"Manifest references models with no artifact: {sorted(missing)}")

    # ---- construction -------------------------------------------------

    @classmethod
    def from_manifest(cls, path: str | Path | None = None, device: str = "cpu"):
        path = Path(path) if path else config.manifest_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Ensemble manifest not found: {path}\n"
                "Run `python main.py` to train and build it first.")
        with open(path) as f:
            manifest = json.load(f)
        return cls(manifest, device=device)

    # ---- prediction ---------------------------------------------------

    def predict_frame(self, df: pd.DataFrame) -> PredictionResult:
        """Predict docking scores for a frame of raw descriptors."""
        X = self.pipeline.transform(df)
        per_model = {name: self.models[name].predict(X) for name in self.members}
        P = np.column_stack([per_model[n] for n in self.members])

        if self.method == "stacking" and self.meta_coef is not None:
            coef = np.asarray(self.meta_coef, dtype=np.float64)
            ensemble = P @ coef + float(self.meta_intercept)
        else:
            w = np.array([self.weights.get(n, 1.0 / len(self.members))
                          for n in self.members], dtype=np.float64)
            w = w / w.sum()
            ensemble = P @ w
        return PredictionResult(ensemble=ensemble, per_model=per_model)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_frame(df).ensemble

    # ---- introspection ------------------------------------------------

    def describe(self) -> dict:
        return {
            "version": self.manifest.get("dockpred_version"),
            "created_utc": self.manifest.get("created_utc"),
            "deployed": self.manifest.get("deployed"),
            "ensemble_method": self.method,
            "n_features": self.pipeline.n_features,
            "members": self.members,
            "weights": self.weights,
            "base_model_metrics": self.manifest.get("base_model_metrics", {}),
            "ensemble_metrics": self.manifest.get("ensemble_metrics", {}),
            "pool_meta": self.manifest.get("pool_meta", {}),
        }
