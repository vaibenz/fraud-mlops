import importlib

import numpy as np
from fastapi.testclient import TestClient

from fraud.bundle import Bundle
from fraud.data import FEATURE_COLS


def test_bundle_round_trip(trained_bundle, synthetic):
    b = Bundle.load(trained_bundle)
    p = b.predict_proba(synthetic[FEATURE_COLS].head(100))
    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))


def _client(trained_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained_bundle))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "pred.jsonl"))
    import fraud.serve as serve
    importlib.reload(serve)
    return TestClient(serve.app), tmp_path / "pred.jsonl"


def test_api_scores_and_logs(trained_bundle, synthetic, tmp_path, monkeypatch):
    client, log = _client(trained_bundle, tmp_path, monkeypatch)
    assert client.get("/health").json() == {"status": "ok"}
    rows = synthetic[FEATURE_COLS].head(5).to_dict("records")
    single = client.post("/predict", json=rows[0]).json()
    assert set(single) == {"fraud_probability", "is_fraud"}
    batch = client.post("/predict/batch", json={"transactions": rows}).json()["predictions"]
    assert len(batch) == 5
    assert len(log.read_text().splitlines()) == 6


def test_api_rejects_invalid_input(trained_bundle, tmp_path, monkeypatch):
    client, _ = _client(trained_bundle, tmp_path, monkeypatch)
    assert client.post("/predict", json={"Time": 1, "Amount": 10}).status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
