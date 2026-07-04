"""End-to-end smoke tests for the dockpred inference stack.

These require the trained artifacts (models/, checkpoints/) and the built
manifest (run ``python main.py`` once). Tests skip cleanly if the manifest is
absent so the suite still passes on a fresh checkout without weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dockpred import config
from dockpred.data import FeaturePipeline

ROOT = config.PROJECT_ROOT
MANIFEST = config.manifest_path()
EXAMPLE = ROOT / "examples" / "sample_labeled.csv"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="ensemble manifest not built; run `python main.py`"
)


@pytest.fixture(scope="module")
def predictor():
    from dockpred.ensemble import EnsemblePredictor

    return EnsemblePredictor.from_manifest()


def test_manifest_shape(predictor):
    assert predictor.pipeline.n_features == 466
    assert len(predictor.members) >= 1


def test_pipeline_alignment_and_transform():
    fp = FeaturePipeline.load()
    # A frame missing half its columns still transforms to the full width.
    n = fp.n_features
    df = pd.DataFrame(np.random.randn(4, n // 2),
                      columns=fp.feature_names[: n // 2])
    X = fp.transform(df)
    assert X.shape == (4, n)
    assert np.isfinite(X).all()


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example input missing")
def test_predict_example(predictor):
    df = pd.read_csv(EXAMPLE)
    result = predictor.predict_frame(df)
    assert result.ensemble.shape == (len(df),)
    assert np.isfinite(result.ensemble).all()
    assert result.ensemble.mean() < 0
    assert result.ensemble.min() > -30
    assert result.ensemble.max() < 30
    assert set(result.per_model) == set(predictor.members)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example input missing")
def test_missing_columns_are_imputed(predictor):
    df = pd.read_csv(EXAMPLE)
    drop = predictor.pipeline.feature_names[:50]
    reduced = df.drop(columns=[c for c in drop if c in df.columns])
    result = predictor.predict_frame(reduced)
    assert np.isfinite(result.ensemble).all()


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example input missing")
def test_evaluate_matches_labels(predictor):
    from dockpred.metrics import regression_metrics

    df = pd.read_csv(EXAMPLE)
    m = regression_metrics(df["score1"].to_numpy(), predictor.predict(df))
    assert m["Pearson"] > 0.4
