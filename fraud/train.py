"""Train both candidates, track everything in MLflow, register the winner.

    python -m fraud.train                      # real data at data/creditcard.csv
    python -m fraud.train --synthetic          # schema identical synthetic data
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Each run logs parameters, the dataset fingerprint, validation and test metrics,
the training curve and the model bundle. The model with the best validation
PR AUC is registered as `fraud-detector` and promoted with the `champion`
alias, and its bundle is exported to models/champion for serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from . import data as data_mod
from .bundle import Bundle
from .evaluate import metrics_at, pick_threshold
from .features import Scaler, engineer
from .models import TorchModel, XGBModel

REGISTERED_NAME = "fraud-detector"
EXPORT_DIR = Path("models/champion")
REPORT_DIR = Path("reports")


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """Lets the registry load any bundle the same way: mlflow.pyfunc.load_model(...)."""

    def load_context(self, context):
        self.bundle = Bundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.bundle.predict_proba(model_input)


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def run(args) -> dict:
    df = data_mod.make_synthetic(args.n_synthetic, seed=args.seed) if args.synthetic else data_mod.load(args.data)
    train, val, test = data_mod.time_split(df)
    data_info = {
        "source": "synthetic" if args.synthetic else str(args.data),
        "fingerprint": fingerprint(df),
        "rows": len(df),
        "fraud": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "split_rows": [len(train), len(val), len(test)],
        "split_fraud": [int(s["Class"].sum()) for s in (train, val, test)],
    }

    F_tr, F_va, F_te = engineer(train), engineer(val), engineer(test)
    scaler = Scaler().fit(F_tr)
    X_tr, X_va, X_te = scaler.transform(F_tr), scaler.transform(F_va), scaler.transform(F_te)
    y_tr, y_va, y_te = (s["Class"].to_numpy() for s in (train, val, test))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    candidates = [XGBModel(seed=args.seed), TorchModel(seed=args.seed, epochs=args.epochs)]
    results = []
    for model in candidates:
        with mlflow.start_run(run_name=model.name) as active:
            mlflow.set_tags({"model_type": model.name, "data_source": data_info["source"]})
            mlflow.log_params({f"model.{k}": v for k, v in model.params.items() if k != "scale_pos_weight"})
            mlflow.log_params({"data.fingerprint": data_info["fingerprint"], "data.rows": data_info["rows"],
                               "data.fraud": data_info["fraud"], "seed": args.seed, "split": "chronological 70/15/15"})

            t0 = time.time()
            model.fit(X_tr, y_tr, X_va, y_va)
            train_secs = time.time() - t0
            if isinstance(model, TorchModel):
                for h in model.history:
                    mlflow.log_metrics({"train_loss": h["train_loss"], "val_pr_auc_curve": h["val_pr_auc"]}, step=h["epoch"])

            p_va, p_te = model.predict_proba(X_va), model.predict_proba(X_te)
            threshold = pick_threshold(y_va, p_va, args.min_precision)
            val_m, test_m = metrics_at(y_va, p_va, threshold), metrics_at(y_te, p_te, threshold)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            mlflow.log_metrics({"threshold": threshold, "train_seconds": train_secs})

            meta = {
                "model_type": model.name,
                "threshold": threshold,
                "mlflow_run_id": active.info.run_id,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": data_info,
                "val": val_m,
                "test": test_m,
            }
            bundle = Bundle(model, scaler, meta)
            with tempfile.TemporaryDirectory() as tmp:
                bdir = Path(tmp) / "bundle"
                bundle.save(bdir, reference=F_tr.sample(min(len(F_tr), 20_000), random_state=args.seed))
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=FraudPyfunc(),
                    artifacts={"bundle": str(bdir)},
                    code_paths=[str(Path(__file__).parent)],
                    input_example=train[data_mod.FEATURE_COLS].head(3),
                )
                staged = Path(tmp) / "keep"
                shutil.copytree(bdir, staged)
                results.append({"model": model.name, "val": val_m, "test": test_m, "threshold": threshold,
                                "run_id": active.info.run_id, "model_uri": info.model_uri,
                                "bundle": _persist(staged, model.name)})

    best = max(results, key=lambda r: r["val"]["pr_auc"])
    version = mlflow.register_model(best["model_uri"], REGISTERED_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
    client.set_model_version_tag(REGISTERED_NAME, version.version, "val_pr_auc", f"{best['val']['pr_auc']:.4f}")

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    shutil.copytree(best["bundle"], EXPORT_DIR)
    meta = json.loads((EXPORT_DIR / "meta.json").read_text())
    meta["registry"] = {"name": REGISTERED_NAME, "version": int(version.version), "alias": "champion"}
    (EXPORT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    summary = {"data": data_info, "champion": best["model"], "registry_version": int(version.version),
               "candidates": [{k: r[k] for k in ("model", "threshold", "val", "test")} for r in results]}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    _write_model_card(summary)
    return summary


def _persist(src: Path, name: str) -> Path:
    dest = Path("models/candidates") / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def _write_model_card(s: dict) -> None:
    d = s["data"]
    rows = []
    for c in s["candidates"]:
        for split in ("val", "test"):
            m = c[split]
            rows.append(f"| {c['model']} | {split} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.3f} | "
                        f"{m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} / {m['tp'] + m['fn']} | {m['fp']} |")
    card = f"""# Model card: {REGISTERED_NAME}

**Champion:** `{s['champion']}` (registry version {s['registry_version']}, alias `champion`), chosen on validation PR AUC.

## Data

Source: `{d['source']}` (fingerprint `{d['fingerprint']}`), {d['rows']:,} transactions, {d['fraud']:,} fraud ({d['fraud_rate']:.3%}).
Chronological split 70/15/15: train {d['split_rows'][0]:,} ({d['split_fraud'][0]} fraud), validation {d['split_rows'][1]:,} ({d['split_fraud'][1]} fraud), test {d['split_rows'][2]:,} ({d['split_fraud'][2]} fraud).

## Results

Thresholds are tuned on validation (max F1) and applied unchanged to test.

| Model | Split | PR AUC | ROC AUC | Precision | Recall | F1 | Fraud caught | False alerts |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Intended use and limits

Scores card transactions for fraud review. V1 to V28 are anonymised PCA components, so the model cannot be explained in terms of raw merchant or customer attributes. Two days of data cannot capture seasonal patterns, so drift should be monitored with `python -m fraud.drift` and the model retrained when key features shift.
"""
    (REPORT_DIR / "model_card.md").write_text(card)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_mod.DEFAULT_PATH))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-precision", type=float, default=None, help="alert budget: lowest threshold with this precision")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--experiment", default="card-fraud")
    summary = run(ap.parse_args())
    for c in summary["candidates"]:
        t = c["test"]
        print(f"{c['model']:12s} test PR AUC {t['pr_auc']:.3f}  precision {t['precision']:.3f}  recall {t['recall']:.3f}")
    print(f"champion: {summary['champion']} (registry v{summary['registry_version']}), exported to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
