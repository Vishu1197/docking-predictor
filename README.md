# 🧠 Docking Score Prediction Tool  
### AI-powered platform for predicting molecular docking scores using pre-trained ML/DL models

---

## 🌍 Overview

The **Docking Score Prediction Tool** is an intelligent web application built with, designed to predict molecular docking scores from a set of **molecular descriptors** — *without performing traditional docking simulations*.  

This tool provides researchers with a fast, cost-effective, and reproducible way to estimate the binding affinity of ligands to proteins using **pre-trained hybrid models** combining **Feed-Forward Neural Networks (FNN)** and **Random Forest (RF) regression**.

---

## 🧬 Scientific Background

Molecular docking is a critical step in computational drug discovery — used to predict the preferred orientation and binding strength between a ligand and a target protein.  
Traditional docking methods (AutoDock, Glide, GOLD, etc.) are **computationally intensive** and require extensive parameter tuning.

This AI-driven approach learns from thousands of known docking results to establish a **predictive relationship between molecular descriptors** and docking scores.  
Once trained, the model can **instantly estimate docking scores** for unseen ligand–protein pairs, significantly accelerating the early stages of virtual screening.

---

## 🚀 Features

✅ Predicts docking scores directly from molecular descriptors (no `docking score` required)  
✅ Uses a **StandardScaler + FNN encoder + Random Forest** hybrid model  
✅ Works for unseen ligands and proteins (trained for generalization)  
✅ Interactive visualization of score distribution  
✅ Modular design — easily update models, scalers, or encoders in the `artifacts/` folder  
✅ Deployable anywhere (Local server, or private lab network)

---

## 🧩 Model Architecture

| Component | Description |
|------------|--------------|
| **StandardScaler** (`standard_scaler.save`) | Normalizes ligand and protein descriptors |
| **FNN Encoder** (`fnn_torch_state.pth`, `fnn_model_info.json`) | Nonlinear transformation network with skip connections |
| **Random Forest Regressor** (`best_rf_model.joblib`) | Predicts final docking score from FNN-transformed features |

**Workflow:**
1. Input descriptors → StandardScaler normalization  
2. Normalized features → FNN feature transformation  
3. Transformed latent vector → Random Forest Regression  
4. Output → Predicted docking score  

---

## 📄 Input File Format

Prepare a CSV file with the following columns:

| Column | Description |
|---------|--------------|
| `ligand` | Ligand ID or PubChem CID |
| `protein` | Target protein name or ID |
| `molecular_weight` | Ligand molecular weight |
| `xlogp` | LogP (lipophilicity) |
| `tpsa` | Topological Polar Surface Area |
| `hbond_donor_count` | Hydrogen bond donor count |
| `hbond_acceptor_count` | Hydrogen bond acceptor count |
| `charge` | Net molecular charge |
| `rotatable_bond_count` | Number of rotatable bonds |
| `prot_nhd`, `prot_nha`, `prot_mw` | Protein descriptors |

⚠️ Do **not** include a `docking score` column — the model predicts it.  
Rows with missing numeric values should be pre-cleaned before upload.

---

## 📊 Output Format

After prediction, the tool adds a new column `predicted_scores` and allows CSV download:

```csv
ligand,protein,molecular_weight,xlogp,tpsa,hbond_donor_count,...,predicted_scores
1234,NS1,314.25,2.1,89.2,2,4,...,-8.57
5678,NS3,290.44,1.8,102.5,3,5,...,-7.92


## 📄 Citation

<div style="background-color:#e6ffed; padding:15px; border-left:5px solid #2ecc71;">

<strong>📢 If you use <code>docking-predictor</code> in your research, please cite:</strong>

<br><br>

**Chanda, V., Hanumantharayudu, P.T., Keshri, V. et al. (2025). _Unveiling natural antiviral agents against dengue virus: a hybrid machine learning and molecular dynamics approach._**  
*Network Modeling Analysis in Health Informatics and Bioinformatics, 14, 164 (2025).*  
📌 **https://doi.org/10.1007/s13721-025-00670-7**

<br><br>

You may also cite the GitHub repository:  
**Chanda, V. (2025). _docking-predictor_ (Version 1.0). GitHub. https://github.com/Vishu1197/docking-predictor**

</div>

