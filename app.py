# app_predict.py
# -----------------------------------------------------
# Docking Score Prediction Tool
# Predicts docking scores using pre-trained FNN + RF models
# Author: Vishal Chanda (Vishu1197)
# -----------------------------------------------------


import os
import joblib
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Optional torch for FNN transform
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

st.set_page_config(page_title="Docking Score Predictor", layout="wide")
st.title("🧠 Docking Score Predictor")
st.markdown(
    "Upload a CSV with ligand, protein and descriptor columns (no `score1`). "
    "The app will predict docking scores using a pre-trained scaler + FNN (optional) + Random Forest."
)

# ---------- Load models (cached) ----------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler missing: {SCALER_PATH}")
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF model missing: {RF_MODEL_PATH}")

    artifacts = {
        "scaler": joblib.load(SCALER_PATH),
        "rf": joblib.load(RF_MODEL_PATH),
        "fnn": None,
        "fnn_info": None
    }

    if TORCH_AVAILABLE and os.path.exists(FNN_STATE_PATH) and os.path.exists(FNN_INFO_PATH):
        with open(FNN_INFO_PATH, "r") as fh:
            fnn_info = json.load(fh)

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

        fnn = FNNSkip(input_dim=fnn_info["input_dim"], trans_dim=fnn_info["transformed_dim"])
        fnn.load_state_dict(torch.load(FNN_STATE_PATH, map_location="cpu"))
        fnn.eval()
        artifacts["fnn"] = fnn
        artifacts["fnn_info"] = fnn_info

    return artifacts

# ---------- UI: file upload ----------
with st.sidebar:
    st.header("Options")
    uploaded = st.file_uploader("Upload descriptors CSV (no score1)", type="csv")
    use_sample = st.checkbox("Use repo sample (artifacts/sample_input_no_score.csv)", value=False)
    run = st.button("Predict")

if run:
    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

    # load dataframe
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample and os.path.exists(os.path.join(ARTIFACT_DIR, "sample_input_no_score.csv")):
        df = pd.read_csv(os.path.join(ARTIFACT_DIR, "sample_input_no_score.csv"))
    else:
        st.warning("Please upload a CSV file or enable the sample option.")
        st.stop()

    # prepare features: exclude ligand/protein if present
    reserved = {"ligand", "protein", "score1"}
    feat_cols = [c for c in df.columns if c not in reserved]
    if len(feat_cols) == 0:
        st.error("No descriptor columns found in the uploaded CSV.")
        st.stop()

    # scale
    scaler = artifacts["scaler"]
    X_raw = df[feat_cols].values.astype(float)
    X_scaled = scaler.transform(X_raw)

    # FNN transform if available
    if artifacts.get("fnn") is not None:
        with torch.no_grad():
            _, X_trans = artifacts["fnn"](torch.tensor(X_scaled, dtype=torch.float32))
        X_final = X_trans.numpy()
    else:
        X_final = X_scaled
        if not TORCH_AVAILABLE:
            st.warning("Torch not available — using scaled features without FNN transform.")

    # predict
    rf = artifacts["rf"]
    preds = rf.predict(X_final)
    df["predicted_scores"] = preds

    # show basic outputs
    st.success(f"Predicted {len(df)} rows.")
    st.dataframe(df.head(10))

    st.subheader("Top Predictions (lowest predicted score first)")
    top_n = st.slider("Top N", 5, 100, 10)
    st.table(df.sort_values("predicted_scores").head(top_n)[["ligand","protein","predicted_scores"]])

    st.subheader("Predicted score distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(preds, bins=30)
    ax.set_xlabel("Predicted docking score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predicted_docking_scores.csv", mime="text/csv")
else:
    st.info("Upload descriptors and click Predict to run.")

