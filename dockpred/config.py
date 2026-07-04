"""Filesystem layout and shared constants for dockpred.

All artifact locations are resolved relative to the project root (the
directory that contains ``models/`` and ``checkpoints/``) so the tool works
whether it is run from the repo, installed as a package, or pointed at a
custom artifact directory via ``DOCKPRED_ARTIFACTS``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of the dockpred package directory.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def artifacts_dir() -> Path:
    """Directory holding trained artifacts (models/, checkpoints/, manifest).

    Override with the ``DOCKPRED_ARTIFACTS`` environment variable to point the
    CLI at a different bundle of trained models.
    """
    env = os.environ.get("DOCKPRED_ARTIFACTS")
    if env:
        return Path(env).expanduser().resolve()
    return PROJECT_ROOT


def models_dir() -> Path:
    return artifacts_dir() / "models"


def checkpoints_dir() -> Path:
    return artifacts_dir() / "checkpoints"


def manifest_path() -> Path:
    return models_dir() / "ensemble_manifest.json"


# Preprocessing artifacts (fit during the retraining pipeline).
IMPUTER_FILE = "imputer.joblib"
SCALER_FILE = "scaler.joblib"
FEATURE_PIPELINE_FILE = "feature_pipeline.json"  # feature-only schema + scaling

# Target column name in the original dataset.
TARGET_COLUMN = "score1"

# Classical ML base models: name -> joblib filename under models/.
ML_MODEL_FILES = {
    "LightGBM": "LightGBM.joblib",
    "XGBoost": "XGBoost.joblib",
    "CatBoost": "CatBoost.joblib",
    "HistGradientBoosting": "HistGradientBoosting.joblib",
}

# Deep-learning base models: name -> checkpoint filename under checkpoints/.
# Only ResidualTabularNet is shipped in the ensemble; the weaker DNN /
# attention checkpoints exist but are intentionally excluded.
DL_MODEL_FILES = {
    "ResidualTabularNet": "ResidualTabularNet.pt",
}
