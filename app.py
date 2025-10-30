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
SAMPLE_PATH = os.path.join(ARTIFACT_DIR, "test.csv")

# ---------- PAGE ----------
st.set_page_config(
    page_title="Docking Score Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:16px;">
      <div style="font-size:46px; line-height:1;">🧠</div>
      <div>
        <h1 style="margin:0 0 6px 0;">Docking Score Predictor</h1>
        <div style="color:#a6adb4;">Predict docking scores from ligand+protein descriptors — fast, reproducible, and easy.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Upload & Options")
    uploaded = st.file_uploader("Upload descriptors CSV (no `score1`)", type="csv", help="CSV must include descriptor columns (e.g., molecular_weight, xlogp, tpsa, ...).")
    use_sample = st.checkbox("Use repo sample CSV", value=False)
    top_n_default = st.number_input("Top N (table)", min_value=3, max_value=200, value=10, step=1)
    run = st.button("Predict")
    st.info("Tip: max file size on Streamlit is 200MB. Use small files for quick demo.")

# ---------- Helpers & model loading ----------
@st.cache_resource
def load_artifacts():
    """Load scaler and RF, optionally FNN. Raises FileNotFoundError if missing core artifacts."""
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

@st.cache_data
def load_csv_from_filelike(file):
    return pd.read_csv(file)

# ---------- Main logic ----------
if run:
    # load artifacts with spinner
    try:
        with st.spinner("Loading models and scaler..."):
            artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

    # read dataframe
    if uploaded is not None:
        try:
            df = load_csv_from_filelike(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    elif use_sample and os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
    else:
        st.warning("Please upload a CSV or enable 'Use repo sample'.")
        st.stop()

    # validate & prepare features
    reserved = {"ligand", "protein", "score1"}
    feat_cols = [c for c in df.columns if c not in reserved]
    if len(feat_cols) == 0:
        st.error("No descriptor columns detected in CSV. Make sure numeric descriptor columns are present.")
        st.stop()

    # show quick dataset card
    n_rows = len(df)
    n_feats = len(feat_cols)
    col1, col2, col3 = st.columns([1,1,2])
    col1.metric("Rows", f"{n_rows}")
    col2.metric("Descriptors", f"{n_feats}")
    col3.write("**First descriptors**")
    col3.write(", ".join(feat_cols[:10]) + (", ..." if len(feat_cols)>10 else ""))

    # scale
    scaler = artifacts["scaler"]
    try:
        X_raw = df[feat_cols].values.astype(float)
    except Exception as e:
        st.error("Could not convert descriptor columns to numeric array. Check for missing/non-numeric values.")
        st.stop()

    X_scaled = scaler.transform(X_raw)

    # FNN transform if available
    if artifacts.get("fnn") is not None:
        st.info("Applying FNN encoder (optional).")
        with st.spinner("Transforming features with FNN..."):
            with torch.no_grad():
                X_final_t = artifacts["fnn"](torch.tensor(X_scaled, dtype=torch.float32))[1]
            X_final = X_final_t.numpy()
    else:
        X_final = X_scaled
        if not TORCH_AVAILABLE:
            st.info("Torch not available — using scaled features only.")

    # predict
    with st.spinner("Running Random Forest predictions..."):
        preds = artifacts["rf"].predict(X_final)
    df["predicted_scores"] = preds

    # layout results: left = table & download, right = plots
    left, right = st.columns([1.5,1])

    with left:
        st.subheader("Preview predictions")
        st.dataframe(df.head(12), use_container_width=True)

        st.markdown("### Top predictions (lowest score = best)")
        top_n = int(top_n_default)
        try:
            top_df = df.sort_values("predicted_scores").head(top_n)[["ligand","protein","predicted_scores"]]
        except Exception:
            top_df = df.head(top_n)[["ligand","protein","predicted_scores"]]
        st.table(top_df)

        # download button
        buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions (CSV)", data=buf, file_name="predicted_docking_scores.csv", mime="text/csv")

    with right:
        st.subheader("Predicted score distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(preds, bins=30, edgecolor="#333", alpha=0.8)
        ax.set_xlabel("Predicted docking score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # scatter if user provided true scores (optional)
        if "score1" in df.columns:
            st.subheader("True vs Predicted (available score1)")
            y_true = df["score1"].values
            y_pred = df["predicted_scores"].values
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.scatter(y_true, y_pred, alpha=0.7, s=30)
            m, b = np.polyfit(y_true, y_pred, 1)
            xs = np.linspace(min(y_true), max(y_true), 100)
            ax2.plot(xs, m*xs + b, color="red", linewidth=1.6)
            ax2.set_xlabel("True docking score")
            ax2.set_ylabel("Predicted docking score")
            st.pyplot(fig2)

    st.success("✅ Predictions complete.")

    # show a small footer with artifact versions if present
    manifest_lines = []
    if os.path.exists(FNN_INFO_PATH):
        try:
            with open(FNN_INFO_PATH, "r") as fh:
                info = json.load(fh)
            manifest_lines.append(f"FNN input_dim={info.get('input_dim')}, transformed_dim={info.get('transformed_dim')}")
        except Exception:
            pass
    st.markdown("----")
    st.caption("Model artifacts loaded from `artifacts/`. Replace files in the repo to update models.")

else:
    # landing / idle state (nice help text)
    st.write(
        """
        **How to use:**  
        1. Upload a CSV file containing ligand, protein and numeric descriptor columns (do **not** include `score1`).  
        2. Click **Predict**.  
        3. Download predictions or view plots.
        """
    )
    st.write("---")
    st.info("Need a quick demo? Check the 'Use repo sample CSV' box in the sidebar (if sample exists).")

# -------------------------------------------------------------
# 📚 Citation Footer (styled version)
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: justify; 
        font-size: 15px; 
        line-height: 1.6; 
        color: #333; 
        background-color: #f4fdf4; 
        border-left: 6px solid #2e7d32; 
        padding: 15px 20px; 
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    ">
        <div style="color: #2e7d32; font-weight: 600; font-size: 16px; margin-bottom: 6px;">
            📚 Please cite this <span style="color:#2e7d32;">tool</span> as:
        </div>
        Chanda, V., Hanumantharayudu, P. T., Keshri, V., Haldar, A., Muralidaran, Y., & Mishra, P. (2025). 
        <em>Unveiling natural antiviral agents against dengue virus: A hybrid machine learning and molecular dynamics approach</em> 
        [Manuscript under review]. 
        Department of Biotechnology, School of Applied Sciences, REVA University, Bengaluru, India.
    </div>
    """,
    unsafe_allow_html=True,
)

