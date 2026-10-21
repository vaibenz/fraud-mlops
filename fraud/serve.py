"""Scoring service.

    uvicorn fraud.serve:app --port 8000
    docker build -t fraud-api . && docker run -p 8000:8000 fraud-api

Loads the champion bundle from MODEL_DIR (default models/champion). Every
scored transaction's model features are appended to PREDICTION_LOG so
`python -m fraud.drift --current logs/predictions.jsonl` can monitor live traffic.
"""

from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .bundle import Bundle
from .data import FEATURE_COLS, V_COLS
from .features import engineer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/champion"))
PREDICTION_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")

Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0, description="Seconds since the first transaction in the window")),
    Amount=(float, Field(ge=0)),
    **{c: (float, ...) for c in V_COLS},
)


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def bundle() -> Bundle:
    if not (MODEL_DIR / "meta.json").exists():
        raise RuntimeError(f"No model bundle in {MODEL_DIR}. Run `python -m fraud.train` first.")
    return Bundle.load(MODEL_DIR)


_log_lock = threading.Lock()


def _log(df: pd.DataFrame, scores) -> None:
    if not PREDICTION_LOG:
        return
    path = Path(PREDICTION_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = engineer(df)
    now = time.time()
    with _log_lock, path.open("a") as f:
        for row, s in zip(feats.to_dict("records"), scores):
            f.write(json.dumps({"ts": now, "score": float(s), **row}) + "\n")


def _score(items: list) -> list[dict]:
    b = bundle()
    df = pd.DataFrame([t.model_dump() for t in items])[FEATURE_COLS]
    scores = b.predict_proba(df)
    _log(df, scores)
    thr = b.meta["threshold"]
    return [{"fraud_probability": round(float(s), 6), "is_fraud": bool(s >= thr)} for s in scores]


app = FastAPI(title="Card Fraud Scoring API", version="0.1.0")


@app.get("/health")
def health():
    try:
        bundle()
        return {"status": "ok"}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/model")
def model_info():
    m = bundle().meta
    return {
        "model_type": m["model_type"],
        "threshold": m["threshold"],
        "registry": m.get("registry"),
        "trained_at": m["trained_at"],
        "data_fingerprint": m["data"]["fingerprint"],
        "test_metrics": m["test"],
    }


@app.post("/predict")
def predict(txn: Transaction):
    return _score([txn])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {"predictions": _score(req.transactions)}
