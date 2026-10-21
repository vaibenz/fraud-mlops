"""Loading, validating and splitting the card transaction data.

The real data is the ULB credit card fraud dataset (Kaggle "creditcardfraud"):
284,807 European card transactions over two days, 492 of them fraud (0.172%).
Columns: Time (seconds since the first transaction), V1 to V28 (PCA components),
Amount, Class (1 = fraud).

`make_synthetic` generates data with the same schema and class balance so the
pipeline, tests and CI run without the 144 MB download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = ["Time", *V_COLS, "Amount"]
TARGET = "Class"

DEFAULT_PATH = Path("data/creditcard.csv")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in [*FEATURE_COLS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[FEATURE_COLS].isna().any().any():
        raise ValueError("Features contain missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Class must be 0 or 1")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be non negative")
    return df


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/, "
            "or run with --synthetic."
        )
    return validate(pd.read_csv(path))


def make_synthetic(n: int = 60_000, fraud_rate: float = 0.00172, seed: int = 7) -> pd.DataFrame:
    """Same schema and imbalance as the real data, with fraud shifted on a few
    components the way it is in the real dataset (V14, V12, V10, V17 low; V4, V11 high)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[: max(1, n // 2000)] = 0  # keep the first rows clean so every split has a stable reference
    V = rng.normal(0, 1, size=(n, 28))
    shift = np.zeros(28)
    for col, s in {"V14": -2.2, "V12": -1.8, "V10": -1.6, "V17": -1.9, "V4": 1.7, "V11": 1.4, "V3": -1.2}.items():
        shift[int(col[1:]) - 1] = s
    fraud = y == 1
    V[fraud] += shift * rng.uniform(0.3, 1.2, size=(fraud.sum(), 1))  # some fraud looks normal
    amount = np.round(rng.lognormal(3.2, 1.3, n), 2)
    amount[fraud] = np.round(rng.lognormal(3.6, 1.6, fraud.sum()), 2)
    time = np.sort(rng.uniform(0, 172_800, n)).round()
    df = pd.DataFrame(V, columns=V_COLS)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df[TARGET] = y
    return df


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological split: train on the past, validate and test on later transactions.
    A random split would leak future fraud patterns into training."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
