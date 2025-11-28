🧠 Docking Score Prediction Tool
AI-powered platform for predicting molecular docking scores using pre-trained ML/DL models
🌍 Overview

The Docking Score Prediction Tool is an intelligent web application designed to predict molecular docking scores using machine learning, without performing traditional docking simulations.

It provides researchers with a fast, cost-effective, and reproducible way to estimate ligand–protein binding affinity using a hybrid model combining:

StandardScaler

Feed-Forward Neural Network (FNN) Encoder

Random Forest Regressor

This approach eliminates the computational cost of conventional docking tools while maintaining high predictive performance.

🧬 Scientific Background

Molecular docking is widely used in computational drug discovery to predict the orientation and binding strength between ligand–protein complexes.
Traditional algorithms (AutoDock, Glide, GOLD, etc.) are:

⚠️ Computationally heavy

⚠️ Time-consuming

⚠️ Sensitive to parameter tuning

Our ML-based workflow uses thousands of precomputed docking results to learn a generalizable mapping between molecular descriptors and docking scores.

Once trained, it can instantly estimate docking scores for unseen ligands and proteins, dramatically accelerating early-stage virtual screening.

🚀 Features

✅ Predicts docking scores directly from molecular descriptors
✅ No docking simulations needed
✅ Hybrid StandardScaler → FNN → Random Forest model
✅ Generalizes to unseen ligands & proteins
✅ Clean UI with interactive results visualization
✅ Replaceable model components (inside artifacts/)
✅ Deployable locally or in private lab networks

🧩 Model Architecture
Component	Description
StandardScaler (standard_scaler.save)	Normalizes ligand + protein descriptors
FNN Encoder (fnn_torch_state.pth, fnn_model_info.json)	Non-linear transformation network (with skip connections)
Random Forest Regressor (best_rf_model.joblib)	Predicts final docking score from encoded features
Workflow

Input descriptors → StandardScaler

Normalized data → FNN encoder

Latent representation → Random Forest model

Output → Predicted docking score

📄 Input File Format

Upload a CSV containing the following columns:

Column	Meaning
ligand	Ligand ID or PubChem CID
protein	Protein ID or name
molecular_weight	MW of ligand
xlogp	LogP / hydrophobicity
tpsa	Topological Polar Surface Area
hbond_donor_count	# donors
hbond_acceptor_count	# acceptors
charge	Net molecular charge
rotatable_bond_count	# rotatable bonds
prot_nhd, prot_nha, prot_mw	Protein descriptors

⚠️ Do not include a docking score column — the model predicts it.

📊 Output Format

After prediction, the tool adds a new column predicted_scores and allows CSV download:

ligand,protein,molecular_weight,xlogp,tpsa,hbond_donor_count,...,predicted_scores
1234,NS1,314.25,2.1,89.2,2,4,...,-8.57
5678,NS3,290.44,1.8,102.5,3,5,...,-7.92

📄 Citation
<img src="https://img.shields.io/badge/Cite%20This%20Tool-Green?style=for-the-badge&color=2ecc71" />

Chanda, V., Hanumantharayudu, P.T., Keshri, V. et al. (2025).
Unveiling natural antiviral agents against dengue virus: a hybrid machine learning and molecular dynamics approach.
Network Modeling Analysis in Health Informatics and Bioinformatics, 14, 164 (2025).
🔗 https://doi.org/10.1007/s13721-025-00670-7

GitHub Repository:
Chanda, V. (2025). docking-predictor (Version 1.0). GitHub.
https://github.com/Vishu1197/docking-predictor
