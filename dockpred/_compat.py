"""Platform compatibility shims applied at import time.

On Windows 11 the ``wmic`` utility used by joblib/loky to count *physical* CPU
cores has been removed, so ``joblib.cpu_count(only_physical_cores=True)`` spawns
a subprocess that fails (~0.7s) on **every** call and is never cached. Scikit's
tree ensembles and HistGradientBoosting call it repeatedly, spawning dozens of
``conhost``/``wmic`` processes and effectively hanging training.

We patch the probe to return the logical core count immediately. Import this
module before any sklearn/joblib parallelism runs.
"""

from __future__ import annotations

import os


def apply() -> None:
    cores = os.cpu_count() or 4
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cores))

    # Models were fit on numpy arrays; predicting on numpy triggers a noisy
    # sklearn/lightgbm "valid feature names" warning that is not actionable.
    import warnings

    warnings.filterwarnings("ignore", message=".*valid feature names.*")
    warnings.filterwarnings("ignore", message=".*was fitted with feature names.*")
    try:
        import joblib.externals.loky.backend.context as ctx

        ctx._count_physical_cores = lambda: (cores, None)
        # populate the module-level cache if present
        for attr in ("physical_cores_cache", "_physical_cores_cache"):
            if hasattr(ctx, attr):
                setattr(ctx, attr, cores)
    except Exception:  # noqa: BLE001 - best effort; never block startup
        pass


apply()
