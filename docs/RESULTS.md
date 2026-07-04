# Results

## Deployed model

**`Ensemble:nnls_blend`** — a non-negative (convex) weighted blend of the six
strongest base models:

| Member | Type | Validation RMSE |
|--------|------|-----------------|
| CatBoost | Gradient boosting | 0.897 |
| XGBoost | Gradient boosting | 0.899 |
| LightGBM | Gradient boosting | 0.903 |
| HistGradientBoosting | Gradient boosting | 0.906 |
| GatedAttentionNet | Deep learning | 0.913 |
| ResidualTabularNet | Deep learning | 0.920 |

Model selection is automatic and driven **only by the validation split**
(RMSE → MSE → R² → Pearson → Spearman). An unconstrained Ridge stack edged the
blend on validation (0.8943 vs 0.8947) but assigned *negative* weights to
correlated members — a sign of meta-overfitting — so the robust convex blend is
deployed instead. This choice is made without reference to the test set.

## Full benchmark (validation split, all 175 chunks)

| Model | Kind | RMSE | MSE | R² | Pearson | Spearman |
|-------|------|------|-----|----|---------|----------|
| Ensemble:nnls_blend (deployed) | ensemble | 0.895 | 0.800 | 0.631 | 0.795 | 0.803 |
| CatBoost | ml | 0.897 | 0.805 | 0.629 | 0.793 | 0.801 |
| XGBoost | ml | 0.899 | 0.808 | 0.628 | 0.792 | 0.801 |
| LightGBM | ml | 0.903 | 0.815 | 0.624 | 0.790 | 0.799 |
| HistGradientBoosting | ml | 0.906 | 0.822 | 0.621 | 0.788 | 0.797 |
| GatedAttentionNet | dl | 0.913 | 0.834 | 0.616 | 0.786 | 0.797 |
| ResidualTabularNet | dl | 0.920 | 0.847 | 0.610 | 0.781 | 0.795 |
| WideDeepNet | dl | 0.930 | 0.866 | 0.601 | 0.776 | 0.791 |
| TabularDNN | dl | 0.931 | 0.866 | 0.601 | 0.776 | 0.790 |
| RandomForest | ml | 0.942 | 0.887 | 0.591 | 0.769 | 0.781 |
| ExtraTrees | ml | 0.943 | 0.890 | 0.590 | 0.769 | 0.779 |
| KNN | ml | 0.988 | 0.975 | 0.551 | 0.745 | 0.754 |
| SVR | ml | 1.000 | 1.000 | 0.539 | 0.736 | 0.760 |
| Ridge | ml | 1.147 | 1.315 | 0.394 | 0.628 | 0.678 |
| ElasticNet | ml | 1.149 | 1.320 | 0.392 | 0.626 | 0.676 |
| Lasso | ml | 1.150 | 1.323 | 0.391 | 0.625 | 0.675 |
| AdaBoost | ml | 1.513 | 2.290 | −0.055 | 0.480 | 0.502 |

Gradient boosting dominates; the masked-trained DL models are competitive; linear
models and AdaBoost lag. All models were trained with feature-masking
augmentation so the ensemble degrades gracefully under missing descriptors.

## Final hold-out evaluation (`data/test/test.csv`, 694 experimental scores)

This set was **never** used in training, tuning, feature selection or model
selection — only for this final evaluation.

| Metric | Value |
|--------|-------|
| MSE | 1.816 |
| RMSE | 1.348 |
| MAE | 1.063 |
| R² | 0.044 |
| Pearson | 0.497 |
| Spearman | 0.469 |
| Explained variance | 0.195 |

See `outputs/test_scatter.png`, `outputs/test_residuals.png`,
`outputs/test_error_hist.png`.

### Analysis — where it works and where it struggles

- **Ranking is the strength.** Pearson ≈ 0.50 and Spearman ≈ 0.47 on genuinely
  unseen *experimental* scores mean the model orders compounds moderately well —
  the property that matters for virtual screening / hit triage.
- **Absolute calibration is the weakness.** Predictions are biased high by
  ≈ +0.77 kcal/mol and slightly compressed (σ 1.08 vs 1.38). This is a
  **computational-vs-experimental domain shift**: the training targets are
  computed docking scores, whereas the test targets are experimental, and the
  test molecules are systematically smaller (mean MW 322 vs 402 in training).
  An affine recalibration on held-out labels would lift R² to ≈ 0.25, but that
  would require using the test labels and is deliberately not done.
- Choosing the robust convex blend over the Ridge stack raised hold-out R² from
  −0.15 to +0.04 and cut RMSE from 1.48 to 1.35 — evidence that avoiding
  meta-overfitting helps out-of-distribution generalisation.

## Top descriptors (LightGBM importance)

Heavy-atom count, molecular complexity, rotatable-bond count, TPSA, XLogP,
molecular weight and H-bond acceptor/donor counts — the classical determinants
of protein–ligand binding. Full list in `outputs/feature_importance.csv`.
