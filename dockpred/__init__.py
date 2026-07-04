"""dockpred - Molecular docking score prediction.

A production inference tool that predicts protein-ligand docking scores
from precomputed molecular descriptors, using a performance-weighted
ensemble of gradient-boosting models and a deep residual network.

The public entry point is :class:`dockpred.ensemble.EnsemblePredictor`.
"""

from dockpred import _compat as _compat  # noqa: F401 - applies platform shims
from dockpred._version import __version__

__all__ = ["__version__", "EnsemblePredictor"]


def __getattr__(name):
    # Lazy import so `import dockpred` stays cheap and does not pull in
    # torch / sklearn until a predictor is actually constructed.
    if name == "EnsemblePredictor":
        from dockpred.ensemble import EnsemblePredictor

        return EnsemblePredictor
    raise AttributeError(f"module 'dockpred' has no attribute {name!r}")
