"""Feature preparation, fitted on the training split only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import V_COLS

MODEL_FEATURES = [*V_COLS, "log_amount", "hour_sin", "hour_cos"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df[V_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    hour = (df["Time"] // 3600) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[MODEL_FEATURES]


@dataclass
class Scaler:
    """Standard scaler stored as plain JSON so the serving image needs no pickle."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "Scaler":
        self.mean = X.mean().tolist()
        self.std = X.std().replace(0, 1).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((X.to_numpy(dtype=np.float32) - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        return cls(**json.loads(path.read_text()))
