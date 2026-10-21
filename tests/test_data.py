import pandas as pd
import pytest

from fraud.data import FEATURE_COLS, make_synthetic, time_split, validate


def test_synthetic_matches_real_schema_and_imbalance():
    df = make_synthetic(50_000, seed=1)
    assert list(df.columns) == [*FEATURE_COLS, "Class"]
    assert 0.0005 < df["Class"].mean() < 0.004


def test_split_is_chronological_and_disjoint(synthetic):
    train, val, test = time_split(synthetic)
    assert train["Time"].max() <= val["Time"].min() <= val["Time"].max() <= test["Time"].min()
    assert len(train) + len(val) + len(test) == len(synthetic)


def test_validation_rejects_bad_data(synthetic):
    with pytest.raises(ValueError):
        validate(synthetic.drop(columns=["V3"]))
    bad = synthetic.copy()
    bad.loc[0, "Amount"] = -1
    with pytest.raises(ValueError):
        validate(bad)
