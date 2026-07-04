"""Classical ML model zoo and Optuna tuning for docking-score regression.

Provides factory functions for every benchmarked regressor and Optuna search
spaces for the strong gradient-boosting models. Kept separate from the training
orchestrator so the set of algorithms is easy to see and extend.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Optional heavy deps imported lazily inside factories so importing the module
# never fails if one is absent.


def _ridge(rs, **kw):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=kw.get("alpha", 10.0), random_state=rs)


def _lasso(rs, **kw):
    from sklearn.linear_model import Lasso
    return Lasso(alpha=kw.get("alpha", 0.01), random_state=rs,
                 max_iter=2000, tol=1e-3, selection="random")


def _elasticnet(rs, **kw):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=kw.get("alpha", 0.01), l1_ratio=kw.get("l1_ratio", 0.5),
                      random_state=rs, max_iter=2000, tol=1e-3, selection="random")


def _random_forest(rs, **kw):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=kw.get("n_estimators", 150), max_depth=kw.get("max_depth", 24),
        min_samples_leaf=kw.get("min_samples_leaf", 5), n_jobs=-1, random_state=rs)


def _extra_trees(rs, **kw):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=kw.get("n_estimators", 150), max_depth=kw.get("max_depth", 24),
        min_samples_leaf=kw.get("min_samples_leaf", 5), n_jobs=-1, random_state=rs)


def _adaboost(rs, **kw):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(n_estimators=kw.get("n_estimators", 200),
                             learning_rate=kw.get("learning_rate", 0.5), random_state=rs)


def _knn(rs, **kw):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=kw.get("n_neighbors", 15),
                               weights=kw.get("weights", "distance"), n_jobs=-1)


def _svr(rs, **kw):
    from sklearn.svm import SVR
    return SVR(C=kw.get("C", 10.0), gamma=kw.get("gamma", "scale"),
               epsilon=kw.get("epsilon", 0.1))


def _histgb(rs, **kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=kw.get("learning_rate", 0.06),
        max_iter=kw.get("max_iter", 600),
        max_leaf_nodes=kw.get("max_leaf_nodes", 63),
        l2_regularization=kw.get("l2_regularization", 0.0),
        max_depth=kw.get("max_depth", None),
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=rs)


def _xgboost(rs, **kw):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=kw.get("n_estimators", 700),
        learning_rate=kw.get("learning_rate", 0.05),
        max_depth=kw.get("max_depth", 8),
        subsample=kw.get("subsample", 0.8),
        colsample_bytree=kw.get("colsample_bytree", 0.8),
        min_child_weight=kw.get("min_child_weight", 5),
        reg_lambda=kw.get("reg_lambda", 1.0),
        n_jobs=-1, random_state=rs, tree_method="hist")


def _lightgbm(rs, **kw):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=kw.get("n_estimators", 900),
        learning_rate=kw.get("learning_rate", 0.05),
        num_leaves=kw.get("num_leaves", 63),
        subsample=kw.get("subsample", 0.8),
        colsample_bytree=kw.get("colsample_bytree", 0.8),
        min_child_samples=kw.get("min_child_samples", 30),
        reg_lambda=kw.get("reg_lambda", 1.0),
        n_jobs=-1, random_state=rs, verbose=-1)


def _catboost(rs, **kw):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=kw.get("iterations", 800),
        learning_rate=kw.get("learning_rate", 0.05),
        depth=kw.get("depth", 8),
        l2_leaf_reg=kw.get("l2_leaf_reg", 3.0),
        random_seed=rs, verbose=False, allow_writing_files=False,
        thread_count=-1)


# name -> factory(random_state, **params)
ML_ZOO: dict[str, Callable] = {
    "Ridge": _ridge,
    "Lasso": _lasso,
    "ElasticNet": _elasticnet,
    "KNN": _knn,
    "SVR": _svr,
    "AdaBoost": _adaboost,
    "RandomForest": _random_forest,
    "ExtraTrees": _extra_trees,
    "HistGradientBoosting": _histgb,
    "XGBoost": _xgboost,
    "LightGBM": _lightgbm,
    "CatBoost": _catboost,
}

# Models that are slow / memory-heavy on hundreds of thousands of rows and are
# fit on a capped subsample during benchmarking.
ROW_CAPPED = {"SVR": 8_000, "KNN": 120_000, "RandomForest": 60_000,
              "ExtraTrees": 60_000, "AdaBoost": 60_000,
              "Lasso": 100_000, "ElasticNet": 100_000}

# Strong models worth Optuna tuning.
TUNABLE = ["LightGBM", "XGBoost", "HistGradientBoosting", "CatBoost"]


def optuna_space(name: str, trial) -> dict:
    if name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1400, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            max_depth=trial.suggest_int("max_depth", 5, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "HistGradientBoosting":
        return dict(
            max_iter=trial.suggest_int("max_iter", 300, 1000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 31, 255),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 5.0, log=True),
        )
    if name == "CatBoost":
        return dict(
            iterations=trial.suggest_int("iterations", 400, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            depth=trial.suggest_int("depth", 5, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        )
    raise ValueError(name)
