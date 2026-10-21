import numpy as np

from fraud.data import time_split
from fraud.evaluate import metrics_at, pick_threshold
from fraud.features import Scaler, engineer
from fraud.models import TorchModel, XGBModel


def _prep(df):
    train, val, test = time_split(df)
    F = [engineer(s) for s in (train, val, test)]
    sc = Scaler().fit(F[0])
    return [sc.transform(f) for f in F], [s["Class"].to_numpy() for s in (train, val, test)]


def test_both_models_beat_random(synthetic):
    (X_tr, X_va, X_te), (y_tr, y_va, y_te) = _prep(synthetic)
    base_rate = y_te.mean()
    for model in (XGBModel(n_estimators=80), TorchModel(epochs=5)):
        model.fit(X_tr, y_tr, X_va, y_va)
        m = metrics_at(y_te, model.predict_proba(X_te), 0.5)
        assert m["pr_auc"] > 5 * base_rate, model.name  # far above the no-skill line


def test_torch_training_is_reproducible(synthetic):
    (X_tr, X_va, _), (y_tr, y_va, _) = _prep(synthetic)
    a = TorchModel(epochs=2, seed=3).fit(X_tr, y_tr, X_va, y_va).predict_proba(X_va[:50])
    b = TorchModel(epochs=2, seed=3).fit(X_tr, y_tr, X_va, y_va).predict_proba(X_va[:50])
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_threshold_respects_precision_budget():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.5, 1, 10)])
    thr = pick_threshold(y, p, min_precision=0.9)
    assert metrics_at(y, p, thr)["precision"] >= 0.9
