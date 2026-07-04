"""Uniform loading/prediction wrappers for the individual base models.

Every base model exposes ``predict(X) -> np.ndarray`` where ``X`` is the
**scaled** ``(n, d)`` feature matrix produced by
``dockpred.data.FeaturePipeline.transform`` and the output is in raw
docking-score units (kcal/mol).

* :class:`SklearnModel` wraps a joblib-persisted regressor (GBMs, linear, ...).
* :class:`TorchModel` wraps a DL checkpoint; the network predicts a standardised
  target and the wrapper de-standardises with the saved ``y_mean``/``y_std``.

Both are reconstructed from the ``base_models`` list embedded in the ensemble
manifest, so inference rebuilds exactly the models the benchmark selected.
"""

from __future__ import annotations

from typing import Protocol

import joblib
import numpy as np

from dockpred import config


class BaseModel(Protocol):
    name: str

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SklearnModel:
    """Wraps a joblib-persisted regressor that predicts raw kcal/mol."""

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator

    @classmethod
    def load(cls, name: str, filename: str) -> "SklearnModel":
        path = config.models_dir() / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing ML model artifact: {path}")
        return cls(name, joblib.load(path))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=np.float64).ravel()


class TorchModel:
    """Wraps a DL checkpoint; de-standardises to raw kcal/mol."""

    def __init__(self, name: str, module, y_mean: float, y_std: float,
                 device: str = "cpu", batch_size: int = 8192):
        self.name = name
        self.module = module
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def load(cls, name: str, filename: str, input_dim: int, y_mean: float,
             y_std: float, arch_kwargs: dict | None = None,
             device: str = "cpu") -> "TorchModel":
        import torch

        from dockpred.nn_models import build_network

        path = config.checkpoints_dir() / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing DL checkpoint: {path}")
        module = build_network(name, input_dim, **(arch_kwargs or {}))
        module.load_state_dict(torch.load(path, map_location=device))
        module.to(device)
        module.eval()
        return cls(name, module, y_mean, y_std, device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        preds = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                stop = start + self.batch_size
                batch = torch.from_numpy(
                    np.ascontiguousarray(X[start:stop], dtype=np.float32)
                ).to(self.device)
                out = self.module(batch).cpu().numpy().ravel()
                preds[start:stop] = out * self.y_std + self.y_mean
        return preds


def load_base_models(base_specs: list[dict], input_dim: int,
                     device: str = "cpu") -> dict[str, BaseModel]:
    """Load every base model described by the manifest's ``base_models`` list."""
    models: dict[str, BaseModel] = {}
    for spec in base_specs:
        name, kind = spec["name"], spec["kind"]
        if kind == "ml":
            models[name] = SklearnModel.load(name, spec["artifact"])
        elif kind == "dl":
            models[name] = TorchModel.load(
                name, spec["artifact"], input_dim,
                y_mean=spec["y_mean"], y_std=spec["y_std"],
                arch_kwargs=spec.get("arch_kwargs"), device=device)
        else:
            raise ValueError(f"Unknown base model kind '{kind}' for '{name}'")
    return models
