# predict_only.py
# -----------------------------------------------------
# Docking Score Prediction Tool
# Predicts docking scores using pre-trained FNN + RF models
# Author: Vishal Chanda (Vishu1197)
# -----------------------------------------------------

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd

# optional torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------- CONFIG ----------
ARTIFACT_DIR = "artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "standard_scaler.save")
RF_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_rf_model.joblib")
FNN_STATE_PATH = os.path.join(ARTIFACT_DIR, "fnn_torch_state.pth")
FNN_INFO_PATH = os.path.join(ARTIFACT_DIR, "fnn_model_info.json")

# FNN class (same architecture as training)
class FNNSkip(nn.Module):
    def __init__(self, input_dim, trans_dim=16, p_dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc_trans = nn.Linear(32, trans_dim)
        self.bn_trans = nn.BatchNorm1d(trans_dim)
        self.proj = nn.Linear(input_dim, trans_dim)
        self.reg = nn.Linear(trans_dim, 1)
        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        x1 = self.act(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        x2 = self.act(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)
        trans = self.act(self.bn_trans(self.fc_trans(x2)))
        trans = self.dropout(trans)
        proj = self.act(self.proj(x))
        added = trans + proj
        out = self.reg(added)
        return out, added

def load_artifacts():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler missing: {SCALER_PATH}")
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF model missing: {RF_MODEL_PATH}")

    scaler = joblib.load(SCALER_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    fnn = None
    fnn_info = None
    if TORCH_AVAILABLE and os.path.exists(FNN_STATE_PATH) and os.path.exists(FNN_INFO_PATH):
        with open(FNN_INFO_PATH, "r") as fh:
            fnn_info = json.load(fh)
        fnn = FNNSkip(input_dim=fnn_info["input_dim"], trans_dim=fnn_info["transformed_dim"])
        fnn.load_state_dict(torch.load(FNN_STATE_PATH, map_location="cpu"))
        fnn.eval()
    return scaler, rf, fnn, fnn_info

def predict(input_csv, output_csv):
    print("Loading input:", input_csv)
    df = pd.read_csv(input_csv)
    reserved = {"ligand", "protein", "score1"}
    feat_cols = [c for c in df.columns if c not in reserved]
    if len(feat_cols) == 0:
        raise ValueError("No descriptor columns found in input CSV.")

    scaler, rf, fnn, fnn_info = load_artifacts()

    # scale
    X_raw = df[feat_cols].values.astype(float)
    X_scaled = scaler.transform(X_raw)

    # fnn transform if available
    if fnn is not None:
        import torch
        with torch.no_grad():
            _, X_trans = fnn(torch.tensor(X_scaled, dtype=torch.float32))
        X_final = X_trans.numpy()
    else:
        X_final = X_scaled

    preds = rf.predict(X_final)
    df["predicted_scores"] = preds

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Saved predictions to:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch docking score predictor")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV (no score1 required)")
    parser.add_argument("--output_csv", type=str, default="predicted_docking_scores.csv", help="Output CSV path")
    args = parser.parse_args()
    predict(args.input_csv, args.output_csv)
