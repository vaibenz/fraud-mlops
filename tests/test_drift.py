import numpy as np
import pandas as pd

from fraud.drift import compare, demo_shift, psi
from fraud.features import engineer


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20_000), rng.normal(size=20_000)) < 0.01


def test_flags_shifted_features_only(trained_bundle):
    ref = pd.read_parquet(trained_bundle / "reference.parquet")
    report = compare(ref, demo_shift(n=10_000))
    assert {"log_amount", "V14"} <= set(report["drifted"])
    assert "V2" not in report["drifted"]
    assert report["retrain_recommended"]


def test_small_samples_are_not_judged(trained_bundle, synthetic):
    ref = pd.read_parquet(trained_bundle / "reference.parquet")
    report = compare(ref, engineer(synthetic.head(50)))
    assert report.get("insufficient_data") and not report["retrain_recommended"]
