"""End-to-end retraining, benchmarking, ensembling and model selection.

Pipeline
--------
1. Build a train/val/test pool from *every* processed chunk (outlier-filtered).
2. Benchmark the full classical-ML zoo; Optuna-tune the strong GBMs.
3. Benchmark the deep-learning architectures with missing-feature masking.
4. Build weighted / stacking / blending ensembles and auto-select the best on
   the validation split by RMSE.
5. Persist every artifact plus a self-contained ``ensemble_manifest.json`` and a
   ``leaderboard.csv``.

The held-out ``data/test/test.csv`` is **never** read here -- it is scored only
by ``dockpred.evaluate`` after the ensemble is frozen.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dockpred import config
from dockpred._version import __version__
from dockpred.data import DataPool, augment_missing, build_pool
from dockpred.metrics import regression_metrics
from dockpred.zoo import ML_ZOO, ROW_CAPPED, TUNABLE, optuna_space

warnings.filterwarnings("ignore")

# FTTransformerLite tokenises all 466 features and can hard-crash (native
# segfault) inside torch's TransformerEncoder on CPU, so it is excluded from the
# default run. The architecture remains available in dockpred.nn_models.
DL_ARCHS = ["ResidualTabularNet", "WideDeepNet", "GatedAttentionNet", "TabularDNN"]


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def tune_model(name: str, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
               n_trials: int, random_state: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = optuna_space(name, trial)
        model = ML_ZOO[name](random_state, **params)
        model.fit(X, y)
        pred = model.predict(Xv)
        return float(np.sqrt(np.mean((pred - yv) ** 2)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    _log(f"  tuned {name}: best_rmse={study.best_value:.4f} params={study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# ML benchmark
# ---------------------------------------------------------------------------


def benchmark_ml(pool: DataPool, *, tune: bool = True, n_trials: int = 25,
                 tune_rows: int = 60_000, augment: bool = True,
                 preset_params: dict | None = None,
                 random_state: int = 42) -> tuple[dict, dict]:
    """Fit + evaluate every ML model. Returns ``(results, tuned_params)``.

    ``preset_params`` supplies already-tuned hyper-parameters (e.g. from a prior
    Optuna run) and short-circuits tuning for those models.
    """
    Xtr, ytr = pool.X_train, pool.y_train
    Xv, yv = pool.X_val, pool.y_val

    # Robustness augmentation: add one masked copy of the training data so the
    # models learn to cope with median-imputed (missing) descriptors.
    if augment:
        Xaug, idx = augment_missing(Xtr, pool.pipeline.mask_fill, n_copies=1,
                                    max_drop=0.5, random_state=random_state)
        yaug = ytr[idx]
    else:
        Xaug, yaug = Xtr, ytr
    _log(f"ML training rows: {len(Xaug)} (augmented from {len(Xtr)})")

    rng = np.random.default_rng(random_state)
    tuned: dict[str, dict] = dict(preset_params or {})
    if tune:
        # tune on a bounded subsample (no augmentation) for speed
        sub = rng.choice(len(Xtr), size=min(tune_rows, len(Xtr)), replace=False)
        for name in TUNABLE:
            if name in tuned:
                _log(f"using preset params for {name}")
                continue
            _log(f"tuning {name} ({n_trials} trials on {len(sub)} rows)...")
            tuned[name] = tune_model(name, Xtr[sub], ytr[sub], Xv, yv,
                                     n_trials=n_trials, random_state=random_state)

    results: dict[str, dict] = {}
    for name, factory in ML_ZOO.items():
        params = tuned.get(name, {})
        # row cap for slow models
        if name in ROW_CAPPED and len(Xaug) > ROW_CAPPED[name]:
            cap = rng.choice(len(Xaug), size=ROW_CAPPED[name], replace=False)
            Xfit, yfit = Xaug[cap], yaug[cap]
        else:
            Xfit, yfit = Xaug, yaug
        try:
            model = factory(random_state, **params)
            t0 = time.time()
            model.fit(Xfit, yfit)
            train_time = time.time() - t0
            t1 = time.time()
            vp = np.asarray(model.predict(Xv), dtype=np.float64)
            infer_time = time.time() - t1
            m = regression_metrics(yv, vp)
            path = config.models_dir() / f"{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            results[name] = {
                "name": name, "kind": "ml", "artifact": path.name,
                "params": params, "val_metrics": m, "val_pred": vp,
                "train_time": train_time, "infer_time": infer_time,
            }
            _log(f"  {name:22s} val_RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} "
                 f"Pearson={m['Pearson']:.4f} ({train_time:.1f}s)")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name:22s} FAILED: {e}")
    return results, tuned


# ---------------------------------------------------------------------------
# DL benchmark
# ---------------------------------------------------------------------------


def benchmark_dl(pool: DataPool, *, archs: list[str] | None = None, epochs: int = 25,
                 device: str = "cpu", max_rows: int | None = None,
                 arch_max_rows: dict | None = None, random_state: int = 42) -> dict:
    from dockpred.dl import train_network

    archs = archs or DL_ARCHS
    arch_max_rows = arch_max_rows or {}
    rng = np.random.default_rng(random_state)
    results: dict[str, dict] = {}
    for name in archs:
        cap = arch_max_rows.get(name, max_rows)
        Xtr, ytr = pool.X_train, pool.y_train
        if cap and len(Xtr) > cap:
            sel = rng.choice(len(Xtr), size=cap, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        _log(f"training DL {name} on {len(Xtr)} rows ...")
        try:
            r = train_network(
                name, Xtr, ytr, pool.X_val, pool.y_val,
                pool.pipeline.mask_fill, epochs=epochs, device=device,
                random_state=random_state, verbose=True)
            results[name] = r
            m = r["val_metrics"]
            _log(f"  {name:22s} val_RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} "
                 f"Pearson={m['Pearson']:.4f} ({r['train_time']:.1f}s)")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name:22s} FAILED: {e}")
    return results


# ---------------------------------------------------------------------------
# Ensembling + selection
# ---------------------------------------------------------------------------


def _nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least-squares blend weights, normalised to sum 1."""
    from scipy.optimize import nnls
    w, _ = nnls(P, y)
    if w.sum() <= 0:
        w = np.ones(P.shape[1])
    return w / w.sum()


def build_ensembles(results: dict, pool: DataPool, *, top_k: int = 6,
                    random_state: int = 42) -> dict:
    """Construct candidate ensembles and evaluate each on the validation split."""
    names = sorted(results, key=lambda n: results[n]["val_metrics"]["RMSE"])
    top = names[:top_k]
    P = np.column_stack([results[n]["val_pred"] for n in top])  # (n_val, k)
    yv = pool.y_val

    candidates: dict[str, dict] = {}

    # 1) equal average
    candidates["Ensemble:equal"] = {
        "method": "weighted_average", "members": top,
        "weights": {n: 1.0 / len(top) for n in top},
        "pred": P.mean(axis=1),
    }

    # 2) inverse-RMSE weighted
    inv = np.array([1.0 / max(results[n]["val_metrics"]["RMSE"], 1e-8) for n in top])
    inv /= inv.sum()
    candidates["Ensemble:inverse_rmse"] = {
        "method": "weighted_average", "members": top,
        "weights": {n: float(w) for n, w in zip(top, inv)},
        "pred": P @ inv,
    }

    # 3) NNLS-optimised blend
    try:
        w = _nnls_weights(P, yv)
        candidates["Ensemble:nnls_blend"] = {
            "method": "weighted_average", "members": top,
            "weights": {n: float(wi) for n, wi in zip(top, w)},
            "pred": P @ w,
        }
    except Exception as e:  # noqa: BLE001
        _log(f"  nnls blend skipped: {e}")

    # 4) Ridge stacking meta-learner (fit on val out-of-sample predictions)
    try:
        from sklearn.linear_model import Ridge
        meta = Ridge(alpha=1.0)
        meta.fit(P, yv)
        candidates["Ensemble:stacking_ridge"] = {
            "method": "stacking", "members": top,
            "meta_coef": meta.coef_.tolist(), "meta_intercept": float(meta.intercept_),
            "pred": meta.predict(P),
        }
    except Exception as e:  # noqa: BLE001
        _log(f"  stacking skipped: {e}")

    for name, c in candidates.items():
        c["val_metrics"] = regression_metrics(yv, c["pred"])
        m = c["val_metrics"]
        _log(f"  {name:26s} val_RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} "
             f"Pearson={m['Pearson']:.4f}")
    return candidates


def _leaderboard_row(name: str, kind: str, m: dict, train_time=None, infer_time=None) -> dict:
    return {
        "model": name, "kind": kind,
        "RMSE": m["RMSE"], "MSE": m["MSE"], "MAE": m["MAE"], "R2": m["R2"],
        "Pearson": m["Pearson"], "Spearman": m["Spearman"],
        "ExplainedVar": m["ExplainedVar"],
        "train_time_s": train_time, "infer_time_s": infer_time,
    }


# Optuna-tuned hyper-parameters from the full HPO run (captured so subsequent
# runs need not repeat the ~1h search). Pass preset_params=None to re-tune.
TUNED_PARAMS = {
    "LightGBM": {"n_estimators": 1000, "learning_rate": 0.02886538767143831,
                 "num_leaves": 61, "subsample": 0.9075938078265732,
                 "colsample_bytree": 0.6223441463224711, "min_child_samples": 70,
                 "reg_lambda": 0.5661324486824431},
    "XGBoost": {"n_estimators": 700, "learning_rate": 0.028831576255828915,
                "max_depth": 8, "subsample": 0.7462519493463221,
                "colsample_bytree": 0.7954038992607804, "min_child_weight": 8,
                "reg_lambda": 0.01773943337541543},
    "HistGradientBoosting": {"max_iter": 1000, "learning_rate": 0.04542131010106149,
                             "max_leaf_nodes": 32, "l2_regularization": 2.5010943210773178},
    "CatBoost": {"iterations": 1000, "learning_rate": 0.05268900643192506,
                 "depth": 10, "l2_leaf_reg": 7.569163777249017},
}


def run_training(
    *, debug_chunks: int | None = None, rows_per_chunk: int = 2200,
    tune: bool = True, n_trials: int = 25, dl_epochs: int = 25,
    device: str = "cpu", random_state: int = 42,
    preset_params: dict | None = None, dl_max_rows: int | None = 100_000,
    outputs_dir: str | Path | None = None,
) -> dict:
    """Full pipeline; returns a summary dict and writes all artifacts."""
    outputs_dir = Path(outputs_dir) if outputs_dir else config.PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("BUILDING DATA POOL (all chunks, outlier-filtered)")
    pool = build_pool(rows_per_chunk=rows_per_chunk, debug_chunks=debug_chunks,
                      random_state=random_state)
    _log(f"pool: {pool.meta}")
    pool.pipeline.save()
    _log(f"saved feature pipeline ({pool.pipeline.n_features} features)")

    _log("=" * 60)
    _log("BENCHMARKING CLASSICAL ML")
    ml_results, tuned = benchmark_ml(pool, tune=tune, n_trials=n_trials,
                                     preset_params=preset_params,
                                     random_state=random_state)

    _log("=" * 60)
    _log("BENCHMARKING DEEP LEARNING")
    # FTTransformer tokenises every feature -> cap it harder to stay CPU-tractable.
    dl_results = benchmark_dl(pool, epochs=dl_epochs, device=device,
                              max_rows=dl_max_rows,
                              arch_max_rows={"FTTransformerLite": 40_000},
                              random_state=random_state)

    all_results = {**ml_results, **dl_results}
    return _finalize(all_results, pool, outputs_dir, random_state)


def _finalize(all_results: dict, pool: DataPool, outputs_dir: Path,
              random_state: int) -> dict:
    """Build ensembles, leaderboard, manifest and feature importance."""
    _log("=" * 60)
    _log("BUILDING ENSEMBLES")
    ensembles = build_ensembles(all_results, pool, random_state=random_state)

    # ---- leaderboard (validation) ----
    rows = []
    for n, r in all_results.items():
        rows.append(_leaderboard_row(n, r["kind"], r["val_metrics"],
                                     r.get("train_time"), r.get("infer_time")))
    for n, c in ensembles.items():
        rows.append(_leaderboard_row(n, "ensemble", c["val_metrics"]))
    lb = pd.DataFrame(rows).sort_values(
        ["RMSE", "MSE", "R2", "Pearson", "Spearman"],
        ascending=[True, True, False, False, False]).reset_index(drop=True)
    lb_path = outputs_dir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    _log(f"wrote leaderboard -> {lb_path}")
    _log("\n" + lb.to_string(index=False))

    _log(f"BEST single/ensemble by validation RMSE: {lb.iloc[0]['model']}")

    # Deploy the best ensemble (ensembles generalise better than any single model).
    # Robustness rule: an unconstrained Ridge stack can assign *negative* weights
    # to correlated base models -- meta-overfitting that hurts out-of-distribution
    # generalisation. When a non-negative convex blend is within a small tolerance
    # of the stack on validation, prefer the blend (a principled choice made
    # without reference to any test set).
    ranked = sorted(ensembles, key=lambda n: ensembles[n]["val_metrics"]["RMSE"])
    best_name = ranked[0]
    best_rmse = ensembles[best_name]["val_metrics"]["RMSE"]
    deploy_name = best_name
    if ensembles[best_name].get("method") == "stacking":
        coefs = ensembles[best_name].get("meta_coef", [])
        has_negative = any(c < 0 for c in coefs)
        if has_negative:
            for cand in ranked:
                c = ensembles[cand]
                if c.get("method") == "weighted_average" and all(
                    w >= 0 for w in c.get("weights", {}).values()
                ) and c["val_metrics"]["RMSE"] <= best_rmse * 1.01:
                    deploy_name = cand
                    _log(f"stack has negative weights; preferring robust convex "
                         f"blend '{cand}' (val RMSE {c['val_metrics']['RMSE']:.4f} "
                         f"vs stack {best_rmse:.4f})")
                    break
    deploy = ensembles[deploy_name]

    members = deploy["members"]
    base_specs = []
    for n in members:
        r = all_results[n]
        spec = {"name": n, "kind": r["kind"], "artifact": r["artifact"]}
        if r["kind"] == "dl":
            spec.update({"y_mean": r["y_mean"], "y_std": r["y_std"],
                         "arch_kwargs": r["arch_kwargs"]})
        base_specs.append(spec)

    manifest = {
        "dockpred_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deployed": deploy_name,
        "ensemble_method": deploy["method"],
        "members": members,
        "weights": deploy.get("weights"),
        "meta_coef": deploy.get("meta_coef"),
        "meta_intercept": deploy.get("meta_intercept"),
        "base_models": base_specs,
        "base_model_metrics": {n: all_results[n]["val_metrics"] for n in members},
        "ensemble_metrics": deploy["val_metrics"],
        "leaderboard": rows,
        "pool_meta": pool.meta,
        "feature_pipeline": pool.pipeline.to_dict(),
    }
    out = config.manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(f"wrote manifest -> {out}  (deployed: {deploy_name}, members={members})")

    _save_feature_importance(all_results, pool, outputs_dir)

    return {"leaderboard": lb, "manifest": manifest, "pool": pool,
            "all_results": all_results, "ensembles": ensembles}


def resume_from_disk(
    *, rows_per_chunk: int = 1000, dl_epochs: int = 12, dl_max_rows: int = 100_000,
    retrain_dl: bool = True, device: str = "cpu", random_state: int = 42,
    outputs_dir: str | Path | None = None,
) -> dict:
    """Rebuild the ensemble from already-trained ML artifacts.

    Reuses the persisted ML ``.joblib`` models (recomputing their validation
    predictions on the deterministically rebuilt pool) and optionally retrains
    the DL architectures, then finalises the leaderboard + manifest. Used to
    recover from a mid-run native crash without repeating ML training.
    """
    outputs_dir = Path(outputs_dir) if outputs_dir else config.PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _log("RESUME: rebuilding data pool (deterministic)")
    pool = build_pool(rows_per_chunk=rows_per_chunk, random_state=random_state)
    pool.pipeline.save()

    all_results: dict = {}
    _log("RESUME: loading ML models from disk + recomputing val predictions")
    for name in ML_ZOO:
        path = config.models_dir() / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        vp = np.asarray(model.predict(pool.X_val), dtype=np.float64)
        all_results[name] = {
            "name": name, "kind": "ml", "artifact": path.name, "params": {},
            "val_metrics": regression_metrics(pool.y_val, vp), "val_pred": vp,
            "train_time": None, "infer_time": None,
        }
        _log(f"  {name:22s} val_RMSE={all_results[name]['val_metrics']['RMSE']:.4f}")

    if retrain_dl:
        _log("RESUME: retraining stable DL architectures")
        dl_results = benchmark_dl(pool, epochs=dl_epochs, device=device,
                                  max_rows=dl_max_rows, random_state=random_state)
        all_results.update(dl_results)

    return _finalize(all_results, pool, outputs_dir, random_state)


def _save_feature_importance(all_results: dict, pool: DataPool, outputs_dir: Path) -> None:
    for cand in ["LightGBM", "XGBoost", "HistGradientBoosting", "CatBoost",
                 "RandomForest", "ExtraTrees"]:
        r = all_results.get(cand)
        if not r:
            continue
        try:
            model = joblib.load(config.models_dir() / r["artifact"])
            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                continue
            fi = pd.DataFrame({"feature": pool.feature_names, "importance": imp})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            fi.to_csv(outputs_dir / "feature_importance.csv", index=False)
            _log(f"feature importance from {cand} -> outputs/feature_importance.csv")
            _log("top 15 features:\n" + fi.head(15).to_string(index=False))
            return
        except Exception:  # noqa: BLE001
            continue
