"""Command-line interface for dockpred.

Subcommands
-----------
predict    Predict docking scores for a CSV/Parquet of molecular descriptors.
           Works with full OR partial descriptor sets -- missing descriptors are
           detected, aligned and imputed automatically.
evaluate   Score data/test/test.csv against experimental scores and write the
           final test_predicted.csv, metrics and diagnostic plots.
info       Show the deployed ensemble: composition, weights and metrics.

Examples
--------
    dockpred predict molecule.csv
    dockpred predict descriptors.parquet -o scores.csv
    dockpred evaluate
    dockpred info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dockpred._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dockpred",
        description="Predict protein-ligand docking scores from molecular "
                    "descriptors using an auto-selected ML+DL ensemble.")
    parser.add_argument("--version", action="version", version=f"dockpred {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict", help="Predict docking scores for descriptors.")
    pp.add_argument("input", help="Input CSV/TSV/Parquet of descriptors.")
    pp.add_argument("-o", "--output", default=None,
                    help="Output path (default: <input>_predicted.csv).")
    pp.add_argument("--id-column", default=None,
                    help="Column carried through to the output as a row id.")
    pp.add_argument("--per-model", action="store_true",
                    help="Include each base model's prediction as extra columns.")
    pp.add_argument("--manifest", default=None)
    pp.add_argument("--device", default="cpu")

    pe = sub.add_parser("evaluate", help="Final hold-out evaluation on test.csv.")
    pe.add_argument("-i", "--input", default=None,
                    help="Labelled CSV/Parquet (default: data/test/test.csv).")
    pe.add_argument("--target-column", default="score1")
    pe.add_argument("--manifest", default=None)
    pe.add_argument("--device", default="cpu")

    pi = sub.add_parser("info", help="Show ensemble composition and weights.")
    pi.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    pi.add_argument("--manifest", default=None)
    pi.add_argument("--device", default="cpu")
    return parser


# --------------------------------------------------------------------------


def _confidence(missing_frac: float) -> tuple[str, str]:
    """Map missing-descriptor fraction to (confidence, estimated accuracy)."""
    if missing_frac == 0.0:
        return "High", "High"
    if missing_frac < 0.10:
        return "High", "Moderate"
    if missing_frac < 0.35:
        return "Moderate", "Moderate"
    if missing_frac < 0.70:
        return "Moderate", "Reduced"
    return "Low", "Low"


def cmd_predict(args) -> int:
    from dockpred import io_utils
    from dockpred.ensemble import EnsemblePredictor

    predictor = EnsemblePredictor.from_manifest(args.manifest, device=args.device)
    df = io_utils.read_table(args.input)

    aligned, missing, extra = predictor.pipeline.align(df)
    n_expected = predictor.pipeline.n_features
    n_detected = n_expected - len(missing)
    missing_frac = len(missing) / max(1, n_expected)
    confidence, est_acc = _confidence(missing_frac)

    result = predictor.predict_frame(df)
    out = result.to_frame(include_per_model=args.per_model)
    if args.id_column and args.id_column in df.columns:
        out.insert(0, args.id_column, df[args.id_column].to_numpy())

    output = args.output or str(Path(args.input).with_name(
        Path(args.input).stem + "_predicted.csv"))
    io_utils.write_table(out, output)

    # ---- formatted report ----
    preds = out["predicted_score"].to_numpy()
    bar = "=" * 49
    lines = [
        bar, "Docking Score Prediction", bar,
        f"Input file:            {args.input}",
        f"Rows:                  {len(df)}",
        f"Descriptors detected:  {n_detected} / {n_expected}",
        f"Missing descriptors:   {len(missing)}",
        "",
        f"Prediction (mean):     {preds.mean():.2f} kcal/mol",
        f"Prediction range:      [{preds.min():.2f}, {preds.max():.2f}] kcal/mol",
        f"Prediction Confidence: {confidence}",
        f"Estimated Accuracy:    {est_acc}",
    ]
    if missing:
        lines += [
            "",
            "Note:",
            "  Some expected molecular descriptors were not provided.",
            "  Missing descriptors were estimated using the trained",
            "  preprocessing pipeline. The predicted docking score may",
            "  have reduced absolute accuracy, but relative ranking",
            "  between candidate molecules remains largely consistent.",
        ]
    lines += ["", f"Saved predictions:     {output}", bar]
    report = "\n".join(lines)
    print(report)

    # persist the report next to the output
    report_path = Path(output).with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    return 0


def cmd_evaluate(args) -> int:
    from dockpred.evaluate import evaluate_holdout

    summary = evaluate_holdout(
        test_path=args.input, manifest=args.manifest,
        target_column=args.target_column, device=args.device)

    m = summary["metrics"]
    print("=" * 49)
    print("Final Hold-out Evaluation (unseen experimental set)")
    print("=" * 49)
    print(f"  rows:          {summary['n_rows']}")
    print(f"  deployed:      {summary['deployed_model']}")
    print(f"  members:       {', '.join(summary['members'])}")
    print("  ---- metrics ----")
    for k in ["MSE", "RMSE", "MAE", "R2", "Pearson", "Spearman", "ExplainedVar"]:
        print(f"  {k:14s} {m[k]:.4f}")
    print(f"\n  wrote {summary['output_csv']}")
    print("=" * 49)
    return 0


def cmd_info(args) -> int:
    from dockpred.ensemble import EnsemblePredictor

    predictor = EnsemblePredictor.from_manifest(args.manifest, device=args.device)
    desc = predictor.describe()
    if args.json:
        print(json.dumps(desc, indent=2))
        return 0

    print(f"dockpred ensemble  (v{desc.get('version')})")
    print(f"  deployed:    {desc.get('deployed')}  [{desc.get('ensemble_method')}]")
    print(f"  created:     {desc.get('created_utc')}")
    print(f"  features:    {desc.get('n_features')}")
    print("\n  base models (validation RMSE | R2):")
    bm = desc.get("base_model_metrics", {})
    for name in desc["members"]:
        m = bm.get(name, {})
        w = desc["weights"].get(name)
        wtxt = f" w={w:.3f}" if w is not None else ""
        print(f"    {name:22s} RMSE {m.get('RMSE', float('nan')):6.3f} "
              f"| R2 {m.get('R2', float('nan')):6.3f}{wtxt}")
    em = desc.get("ensemble_metrics", {})
    if em:
        print(f"\n  ENSEMBLE  RMSE {em.get('RMSE'):.3f} | R2 {em.get('R2'):.3f} "
              f"| Pearson {em.get('Pearson'):.3f} | Spearman {em.get('Spearman'):.3f}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"predict": cmd_predict, "evaluate": cmd_evaluate, "info": cmd_info}
    try:
        return handlers[args.command](args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
