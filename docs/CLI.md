# dockpred CLI Reference

Install the CLI with `pip install -e .` (or run `python -m dockpred <command>`).

## `dockpred predict`

Predict docking scores for a table of molecular descriptors. Works with a full
466-descriptor set **or any subset** — missing descriptors are detected, aligned
to the training schema, and imputed automatically.

```
dockpred predict INPUT [-o OUTPUT] [--id-column COL] [--per-model]
                        [--manifest PATH] [--device cpu|cuda]
```

| Argument | Description |
|----------|-------------|
| `INPUT` | CSV / TSV / Parquet of descriptors (positional, required). |
| `-o, --output` | Output path. Default: `<input>_predicted.csv`. |
| `--id-column` | A column carried through to the output as a row identifier. |
| `--per-model` | Also emit each base model's prediction (`score_<Model>`). |
| `--manifest` | Custom `ensemble_manifest.json`. Default: `models/ensemble_manifest.json`. |
| `--device` | Torch device for the DL member. Default: `cpu`. |

Output: a table with a `predicted_score` column, plus a `.report.txt` next to it
summarising descriptors detected, confidence and estimated accuracy.

**Confidence levels** (driven by the fraction of missing descriptors):

| Missing fraction | Confidence | Estimated accuracy |
|------------------|------------|--------------------|
| 0 %              | High       | High               |
| < 10 %           | High       | Moderate           |
| 10–35 %          | Moderate   | Moderate           |
| 35–70 %          | Moderate   | Reduced            |
| > 70 %           | Low        | Low                |

## `dockpred evaluate`

Score a labelled dataset (default `data/test/test.csv`) against its experimental
`score1` column and produce the final deliverables.

```
dockpred evaluate [-i INPUT] [--target-column score1] [--manifest PATH] [--device D]
```

Writes:
- `data/test/test_predicted.csv` — original file + `predicted_score` after `score1`.
- `outputs/test_metrics.json` — full metric suite + residual stats.
- `outputs/test_scatter.png`, `test_residuals.png`, `test_error_hist.png`.

## `dockpred info`

Show the deployed ensemble: method, members, per-model validation metrics and
blend weights. Add `--json` for machine-readable output.

```
dockpred info [--json]
```
