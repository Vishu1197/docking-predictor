"""Data loading, cleaning and the fitted feature-preprocessing pipeline.

This module is the single source of truth for turning the raw molecular
descriptor chunks (``data/processed/chunk_*.parquet``) into model-ready
matrices, and for persisting the exact preprocessing so inference on arbitrary
user input (full *or* partial descriptor sets) reproduces it faithfully.

Design decisions that fix the original project's *regression-to-the-mean*
failure:

* **Target outlier filtering.** The raw docking scores contain a small (~1.3%)
  tail of non-physical values (up to +423 kcal/mol from failed dockings). Those
  dominate an MSE loss and collapse every model onto the global mean. Training
  rows are restricted to a physically sensible score window.
* **Feature winsorisation.** A handful of descriptor cells reach ~3.5e6; they
  are clipped to per-feature quantile bounds before standardisation so linear /
  DL / distance models are not wrecked by them.
* **All chunks are used.** The pool samples rows from *every* one of the ~175
  chunks (not a handful), unless ``debug`` limits the file count.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dockpred import config


# ---------------------------------------------------------------------------
# Fitted preprocessing pipeline
# ---------------------------------------------------------------------------


@dataclass
class FeaturePipeline:
    """Median-impute + winsorise + standard-scale for the descriptor columns.

    The same object is fit during training and reloaded at inference. At
    inference it also *aligns* arbitrary user frames to the training schema so
    missing descriptors are created and imputed automatically.
    """

    feature_names: list[str]
    impute_values: np.ndarray          # per-feature median (raw units)
    clip_lo: np.ndarray                # per-feature lower winsor bound (raw)
    clip_hi: np.ndarray                # per-feature upper winsor bound (raw)
    center: np.ndarray                 # StandardScaler mean (post-clip)
    scale: np.ndarray                  # StandardScaler std (post-clip)
    mask_fill: np.ndarray              # scaled value of an imputed feature
    target_column: str = config.TARGET_COLUMN
    score_bounds: tuple[float, float] = (-15.0, 0.0)

    # ---- properties ---------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # kept for manifest/back-compat readers
    @property
    def target_center(self) -> float:
        return 0.0

    @property
    def target_scale(self) -> float:
        return 1.0

    # ---- fitting ------------------------------------------------------

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_names: list[str],
            target_column: str = config.TARGET_COLUMN,
            score_bounds: tuple[float, float] = (-15.0, 0.0),
            clip_quantiles: tuple[float, float] = (0.001, 0.999)) -> "FeaturePipeline":
        X = df[feature_names].to_numpy(dtype=np.float64, copy=True)
        X[~np.isfinite(X)] = np.nan

        impute = np.nanmedian(X, axis=0)
        impute = np.where(np.isfinite(impute), impute, 0.0)

        lo = np.nanquantile(X, clip_quantiles[0], axis=0)
        hi = np.nanquantile(X, clip_quantiles[1], axis=0)
        lo = np.where(np.isfinite(lo), lo, impute)
        hi = np.where(np.isfinite(hi), hi, impute)
        # guard degenerate bounds
        hi = np.where(hi <= lo, lo + 1e-9, hi)

        # impute then clip to compute scaler stats
        inds = np.where(np.isnan(X))
        X[inds] = np.take(impute, inds[1])
        Xc = np.clip(X, lo, hi)

        center = Xc.mean(axis=0)
        scale = Xc.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        mask_fill = ((np.clip(impute, lo, hi) - center) / scale).astype(np.float32)

        return cls(
            feature_names=list(feature_names),
            impute_values=impute, clip_lo=lo, clip_hi=hi,
            center=center, scale=scale, mask_fill=mask_fill,
            target_column=target_column, score_bounds=score_bounds,
        )

    # ---- (de)serialisation -------------------------------------------

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "impute_values": self.impute_values.tolist(),
            "clip_lo": self.clip_lo.tolist(),
            "clip_hi": self.clip_hi.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "mask_fill": self.mask_fill.astype(np.float64).tolist(),
            "scaler_type": "standard+winsor",
            "target_column": self.target_column,
            "target_transform": "identity",
            "score_bounds": list(self.score_bounds),
        }

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config.models_dir() / config.FEATURE_PIPELINE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        return path

    @classmethod
    def from_dict(cls, fp: dict) -> "FeaturePipeline":
        d = len(fp["feature_names"])
        impute = np.asarray(fp["impute_values"], dtype=np.float64)
        center = np.asarray(fp["center"], dtype=np.float64)
        scale = np.asarray(fp["scale"], dtype=np.float64)
        scale = np.where(scale < 1e-8, 1.0, scale)
        lo = np.asarray(fp.get("clip_lo", impute - 1e9), dtype=np.float64)
        hi = np.asarray(fp.get("clip_hi", impute + 1e9), dtype=np.float64)
        mask_fill = fp.get("mask_fill")
        if mask_fill is None:
            mask_fill = (np.clip(impute, lo, hi) - center) / scale
        bounds = fp.get("score_bounds", fp.get("outlier_bounds", [-15.0, 0.0]))
        return cls(
            feature_names=list(fp["feature_names"]),
            impute_values=impute, clip_lo=lo, clip_hi=hi,
            center=center, scale=scale,
            mask_fill=np.asarray(mask_fill, dtype=np.float32).reshape(d),
            target_column=fp.get("target_column", config.TARGET_COLUMN),
            score_bounds=(float(bounds[0]), float(bounds[1])),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FeaturePipeline":
        path = Path(path) if path else config.models_dir() / config.FEATURE_PIPELINE_FILE
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_manifest(cls, manifest: dict) -> "FeaturePipeline":
        return cls.from_dict(manifest["feature_pipeline"])

    # ---- alignment + transform ---------------------------------------

    def align(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Reindex an arbitrary frame to the training feature columns/order.

        Missing descriptors become all-NaN (imputed in ``transform``); extra
        columns (ids, labels, ...) are dropped. Returns ``(aligned, missing,
        extra)`` for user-facing reporting.
        """
        cols = set(df.columns)
        missing = [c for c in self.feature_names if c not in cols]
        extra = [c for c in df.columns if c not in set(self.feature_names)]
        aligned = df.reindex(columns=self.feature_names)
        return aligned, missing, extra

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Raw (possibly partial) descriptor frame -> scaled float32 matrix."""
        aligned, _, _ = self.align(df)
        X = aligned.to_numpy(dtype=np.float64, copy=True)
        nan = ~np.isfinite(X)
        if nan.any():
            X[nan] = np.take(self.impute_values, np.where(nan)[1])
        X = np.clip(X, self.clip_lo, self.clip_hi)
        X = (X - self.center) / self.scale
        return X.astype(np.float32)

    def transform_matrix(self, X_raw: np.ndarray) -> np.ndarray:
        """Same as :meth:`transform` for an already-aligned raw numpy matrix."""
        X = np.asarray(X_raw, dtype=np.float64, copy=True)
        nan = ~np.isfinite(X)
        if nan.any():
            X[nan] = np.take(self.impute_values, np.where(nan)[1])
        X = np.clip(X, self.clip_lo, self.clip_hi)
        X = (X - self.center) / self.scale
        return X.astype(np.float32)

    def missing_fraction(self, df: pd.DataFrame) -> float:
        _, missing, _ = self.align(df)
        return len(missing) / max(1, self.n_features)


# ---------------------------------------------------------------------------
# Raw data pool
# ---------------------------------------------------------------------------


@dataclass
class DataPool:
    """A materialised train/val/test split plus the fitted pipeline."""

    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    pipeline: FeaturePipeline
    meta: dict = field(default_factory=dict)


def list_chunks(processed_dir: str | Path | None = None) -> list[str]:
    processed_dir = Path(processed_dir) if processed_dir else config.PROJECT_ROOT / "data" / "processed"
    files = sorted(glob.glob(str(processed_dir / "chunk_*.parquet")))
    if not files:
        files = sorted(glob.glob(str(processed_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet chunks under {processed_dir}")
    return files


def build_pool(
    processed_dir: str | Path | None = None,
    target_column: str = config.TARGET_COLUMN,
    rows_per_chunk: int = 2200,
    score_bounds: tuple[float, float] = (-15.0, 0.0),
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    debug_chunks: int | None = None,
    random_state: int = 42,
) -> DataPool:
    """Sample rows from *every* chunk, clean, split and fit the pipeline.

    Parameters
    ----------
    rows_per_chunk
        Rows sampled per parquet chunk. With ~175 chunks the default yields a
        ~350k-row pool spanning the full dataset while staying tractable on CPU.
    debug_chunks
        If set, only the first N chunks are read (fast smoke tests).
    """
    rng = np.random.default_rng(random_state)
    files = list_chunks(processed_dir)
    if debug_chunks:
        files = files[:debug_chunks]

    parts: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        if target_column not in df.columns:
            continue
        # keep only physically sensible scores for training signal
        y = df[target_column].to_numpy(dtype=np.float64)
        keep = np.isfinite(y) & (y >= score_bounds[0]) & (y <= score_bounds[1])
        df = df.loc[keep]
        if len(df) > rows_per_chunk:
            idx = rng.choice(len(df), size=rows_per_chunk, replace=False)
            df = df.iloc[idx]
        parts.append(df)

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    feature_names = [c for c in pool.columns if c != target_column]
    y = pool[target_column].to_numpy(dtype=np.float64)

    n = len(pool)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_df = pool.iloc[:n_test]
    val_df = pool.iloc[n_test:n_test + n_val]
    train_df = pool.iloc[n_test + n_val:]

    # Fit pipeline on TRAIN ONLY (no leakage into val/test).
    pipeline = FeaturePipeline.fit(
        train_df, feature_names, target_column=target_column, score_bounds=score_bounds)

    def xt(d: pd.DataFrame) -> np.ndarray:
        return pipeline.transform(d[feature_names])

    return DataPool(
        feature_names=feature_names,
        X_train=xt(train_df), y_train=train_df[target_column].to_numpy(np.float64),
        X_val=xt(val_df), y_val=val_df[target_column].to_numpy(np.float64),
        X_test=xt(test_df), y_test=test_df[target_column].to_numpy(np.float64),
        pipeline=pipeline,
        meta={
            "n_chunks": len(files), "rows_per_chunk": rows_per_chunk,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "score_bounds": list(score_bounds),
        },
    )


def augment_missing(X: np.ndarray, mask_fill: np.ndarray, *, n_copies: int = 1,
                    max_drop: float = 0.5, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return masked copies of ``X`` simulating missing descriptors.

    For each copy, a random fraction (uniform in ``[0, max_drop]``) of features
    per row is replaced by ``mask_fill`` (the scaled value of a median-imputed
    feature). Concatenated with the originals this makes every downstream model
    robust to partial descriptor sets at inference time. Returns ``(mask_stack,
    keep_index)`` where ``keep_index`` maps each augmented row to its source row
    so the target can be tiled.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    out = [X]
    idx = [np.arange(n)]
    for _ in range(n_copies):
        Xc = X.copy()
        frac = rng.uniform(0.0, max_drop, size=n)[:, None]
        drop = rng.random((n, d)) < frac
        Xc[drop] = np.broadcast_to(mask_fill, (n, d))[drop]
        out.append(Xc)
        idx.append(np.arange(n))
    return np.concatenate(out, axis=0), np.concatenate(idx, axis=0)
