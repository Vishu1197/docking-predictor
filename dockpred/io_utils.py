"""Tabular input/output helpers (CSV and Parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame based on extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(
        f"Unsupported input extension '{suffix}'. Use .csv, .tsv or .parquet."
    )


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV or Parquet based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
