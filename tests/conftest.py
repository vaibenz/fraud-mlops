import pytest

from fraud.bundle import Bundle
from fraud.data import make_synthetic, time_split
from fraud.evaluate import pick_threshold
from fraud.features import Scaler, engineer
from fraud.models import XGBModel


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic(20_000, seed=11)


@pytest.fixture(scope="session")
def trained_bundle(synthetic, tmp_path_factory):
    train, val, _ = time_split(synthetic)
    F_tr, F_va = engineer(train), engineer(val)
    scaler = Scaler().fit(F_tr)
    model = XGBModel(n_estimators=80).fit(scaler.transform(F_tr), train["Class"].to_numpy(),
                                          scaler.transform(F_va), val["Class"].to_numpy())
    thr = pick_threshold(val["Class"].to_numpy(), model.predict_proba(scaler.transform(F_va)))
    meta = {"model_type": "xgboost", "threshold": thr, "trained_at": "test", "data": {"fingerprint": "test"},
            "val": {}, "test": {}}
    d = tmp_path_factory.mktemp("bundle")
    Bundle(model, scaler, meta).save(d, reference=F_tr)
    return d
