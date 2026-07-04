"""Deep-learning training with missing-descriptor robustness.

Trains the tabular networks in :mod:`dockpred.nn_models` on the scaled feature
matrix. Two ingredients make the networks robust to partial descriptor sets at
inference time:

* **Feature masking augmentation.** Each mini-batch has a random fraction of its
  features replaced by ``mask_fill`` (the scaled value of a median-imputed
  feature) -- a denoising / feature-dropout objective that teaches the network
  to predict from incomplete inputs.
* **Standardised target.** The network regresses the z-scored target; the
  ``y_mean``/``y_std`` are saved with the checkpoint and undone at inference.

AdamW + OneCycle LR + gradient clipping + early stopping on validation RMSE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dockpred import config
from dockpred.metrics import regression_metrics
from dockpred.nn_models import build_network


def _batches(n: int, bs: int, rng: np.random.Generator, shuffle: bool):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def train_network(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    mask_fill: np.ndarray,
    *,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    mask_prob: float = 0.15,
    max_drop: float = 0.5,
    patience: int = 6,
    device: str = "cpu",
    random_state: int = 42,
    arch_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Train one architecture; return checkpoint path, metrics and timings."""
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)
    device_t = torch.device(device)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    yt = ((y_train - y_mean) / y_std).astype(np.float32)
    yv = ((y_val - y_mean) / y_std).astype(np.float32)

    Xtr = torch.from_numpy(X_train.astype(np.float32))
    ytr = torch.from_numpy(yt)
    Xva = torch.from_numpy(X_val.astype(np.float32)).to(device_t)
    mask_t = torch.from_numpy(mask_fill.astype(np.float32)).to(device_t)

    input_dim = X_train.shape[1]
    arch_kwargs = arch_kwargs or {}
    net = build_network(name, input_dim, **arch_kwargs).to(device_t)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X_train)
    steps = max(1, (n + batch_size - 1) // batch_size) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.SmoothL1Loss()  # Huber: robust to any residual heavy tails

    best_rmse = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()

    for epoch in range(epochs):
        net.train()
        for bidx in _batches(n, batch_size, rng, shuffle=True):
            xb = Xtr[bidx].to(device_t)
            yb = ytr[bidx].to(device_t)
            # per-row random masking augmentation
            if mask_prob > 0:
                frac = torch.from_numpy(
                    rng.uniform(0.0, max_drop, size=len(bidx)).astype(np.float32)
                ).to(device_t).unsqueeze(1)
                use = torch.from_numpy(
                    (rng.random(len(bidx)) < mask_prob).astype(np.float32)
                ).to(device_t).unsqueeze(1)
                drop = (torch.rand_like(xb) < frac) & (use > 0)
                xb = torch.where(drop, mask_t.expand_as(xb), xb)
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            vp = net(Xva).cpu().numpy() * y_std + y_mean
        rmse = float(np.sqrt(np.mean((vp - y_val) ** 2)))
        if verbose:
            print(f"  [{name}] epoch {epoch + 1:2d}/{epochs}  val_rmse={rmse:.4f}")
        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [{name}] early stop at epoch {epoch + 1}")
                break

    train_time = time.time() - t0
    if best_state is not None:
        net.load_state_dict(best_state)

    # final val predictions + timing
    net.eval()
    t1 = time.time()
    with torch.no_grad():
        val_pred = net(Xva).cpu().numpy() * y_std + y_mean
    infer_time = time.time() - t1

    ckpt_dir = config.checkpoints_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    torch.save(net.state_dict(), ckpt_path)

    return {
        "name": name, "kind": "dl", "artifact": ckpt_path.name,
        "y_mean": y_mean, "y_std": y_std, "arch_kwargs": arch_kwargs,
        "val_metrics": regression_metrics(y_val, val_pred),
        "val_pred": val_pred, "train_time": train_time, "infer_time": infer_time,
    }


def predict_network(name: str, ckpt_path: str | Path, input_dim: int,
                    y_mean: float, y_std: float, X: np.ndarray,
                    arch_kwargs: dict | None = None, device: str = "cpu",
                    batch_size: int = 8192) -> np.ndarray:
    net = build_network(name, input_dim, **(arch_kwargs or {}))
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device).eval()
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size], dtype=np.float32)).to(device)
            out[s:s + batch_size] = net(xb).cpu().numpy().ravel() * y_std + y_mean
    return out
