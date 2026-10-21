"""A model bundle is everything serving needs, in one directory:

    model.ubj | model.pt   trained weights
    scaler.json            feature scaling fitted on the training split
    meta.json              model type, threshold, version, metrics, data fingerprint
    reference.parquet      training feature sample, used as the drift baseline
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import Scaler, engineer
from .models import MODEL_TYPES


@dataclass
class Bundle:
    model: object
    scaler: Scaler
    meta: dict

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(engineer(raw)))

    def save(self, d: Path, reference: pd.DataFrame | None = None) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d)
        self.scaler.save(d / "scaler.json")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2))
        if reference is not None:
            reference.to_parquet(d / "reference.parquet", index=False)

    @classmethod
    def load(cls, d: str | Path) -> "Bundle":
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        model = MODEL_TYPES[meta["model_type"]].load(d)
        return cls(model, Scaler.load(d / "scaler.json"), meta)
