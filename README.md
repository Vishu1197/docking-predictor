# 🧠 Docking Score Prediction Tool  
### AI-powered platform for predicting molecular docking scores using pre-trained ML/DL models

---

## 🌍 Overview

The **Docking Score Prediction Tool** is an intelligent web application designed to predict **molecular docking scores using machine learning**, without performing traditional docking simulations.

It provides researchers with a **fast, cost-effective, and reproducible** method to estimate ligand–protein binding affinity using a hybrid ML architecture combining StandardScaler, a Feed-Forward Neural Network (FNN) Encoder, and a Random Forest Regressor.

---

## 🧬 Scientific Background

Molecular docking is widely used in computational drug discovery to assess ligand–protein interactions.  
Traditional docking tools (AutoDock, Glide, GOLD, etc.) are:

- Computationally expensive  
- Time-intensive  
- Sensitive to parameter tuning  

Our ML-based system is trained on thousands of docking results to learn a predictive mapping between molecular descriptors and docking scores.  
This allows **instant prediction** for new ligand–protein pairs, enabling rapid virtual screening.

---

## 🚀 Features

- Predicts docking scores directly from numerical descriptors  
- No need for computational docking  
- Hybrid ML model (StandardScaler → FNN → Random Forest)  
- Generalizes to unseen ligands and proteins  
- Clean user interface with interactive visualization  
- Easily replaceable model and scaler components in `artifacts/`  
- Works locally or in private lab environments  

---

## 🧩 Model Architecture

| Component | Description |
|----------|-------------|
| **StandardScaler** (`standard_scaler.save`) | Normalizes ligand + protein descriptors |
| **FNN Encoder** (`fnn_torch_state.pth`, `fnn_model_info.json`) | Non-linear transformation network with skip connections |
| **Random Forest Regressor** (`best_rf_model.joblib`) | Predicts docking score from encoded features |

### Workflow:
1. Input descriptors → StandardScaler  
2. Normalized data → FNN encoder  
3. Latent representation → Random Forest  
4. Output → Predicted docking score  

---

## 📄 Input File Format

Upload a CSV containing the following columns:

| Column | Meaning |
|--------|---------|
| `ligand` | Ligand ID or PubChem CID |
| `protein` | Protein ID or name |
| `molecular_weight` | Ligand molecular weight |
| `xlogp` | Lipophilicity (LogP) |
| `tpsa` | Topological Polar Surface Area |
| `hbond_donor_count` | H-bond donors |
| `hbond_acceptor_count` | H-bond acceptors |
| `charge` | Net molecular charge |
| `rotatable_bond_count` | Rotatable bonds |
| `prot_nhd`, `prot_nha`, `prot_mw` | Protein descriptors |

⚠️ Do **not** include `docking score` — the model predicts it.

---

## 📊 Output Format

After prediction, the tool adds a new column `predicted_scores` to the CSV:

```csv
ligand,protein,molecular_weight,xlogp,tpsa,hbond_donor_count,...,predicted_scores
1234,NS1,314.25,2.1,89.2,2,4,...,-8.57
5678,NS3,290.44,1.8,102.5,3,5,...,-7.92
```

---

## 📄 Citation

> ![Cite](https://img.shields.io/badge/Cite%20This%20Tool-Green?style=for-the-badge&color=2ecc71)
>
> **Chanda, V., Hanumantharayudu, P. T., Keshri, V. et al. (2025).  
> _Unveiling natural antiviral agents against dengue virus: a hybrid machine learning and molecular dynamics approach._  
> Network Modeling Analysis in Health Informatics and Bioinformatics, 14, 164 (2025).**  
> DOI: https://doi.org/10.1007/s13721-025-00670-7
>
> **GitHub Repository Citation:**  
> Chanda, V. (2025). *docking-predictor* (Version 1.0). GitHub.  
> https://github.com/Vishu1197/docking-predictor

