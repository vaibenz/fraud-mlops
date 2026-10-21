# Card Fraud Detection with MLOps

Fraud detection on 284,807 real card transactions, where only 492 (0.17%) are fraud, taken from training through to a monitored scoring service. The modelling compares a PyTorch neural network against an XGBoost baseline. The engineering is the MLOps around it: reproducible, tracked experiments, a model registry, a containerised API, CI, and drift monitoring that says when to retrain.

## Pipeline

```
creditcard.csv ─▶ validate ─▶ chronological split (70/15/15)
                                   │
                    feature prep fitted on train only
                                   │
                ┌──────────────────┴──────────────────┐
           XGBoost baseline                    PyTorch MLP
     (scale_pos_weight, early stop)    (weighted BCE, early stop)
                └──────────────────┬──────────────────┘
          threshold tuned on validation, scored once on test
                                   │
     MLflow: params, data fingerprint, metrics, curves, model
                                   │
      best validation PR AUC ─▶ registry "fraud-detector@champion"
                                   │
         bundle ─▶ FastAPI in Docker ─▶ prediction log ─▶ drift report
```

## Design decisions

- **PR AUC, not accuracy.** Predicting "never fraud" is 99.8% accurate. Models are compared on PR AUC, and reported with precision, recall, frauds caught and false alerts per 10,000 transactions.
- **Chronological split.** The model trains on earlier transactions and is tested on later ones, the way it would be used. A random split leaks future fraud patterns into training.
- **Class weighting instead of resampling.** XGBoost uses `scale_pos_weight` and the MLP uses a weighted BCE loss, so the data distribution stays real.
- **Threshold chosen on validation only.** The default maximises F1. `--min-precision 0.9` instead picks the lowest threshold that keeps 90% of alerts genuine, the knob a fraud operations team would actually turn.
- **Reproducibility.** Fixed seeds, a SHA 256 fingerprint of the training data on every run, and scaling stored as JSON with the model so serving uses exactly what training fitted.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Real data: download creditcard.csv from
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/
make train              # or: make train-synthetic (same schema, no download)

make mlflow             # compare runs at http://localhost:5000
make serve              # API at http://localhost:8000/docs
make drift              # drift report against simulated shifted traffic
make test
```

### Docker

```bash
make train              # the image packages models/champion
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
curl localhost:8000/model
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one transaction: `fraud_probability`, `is_fraud` |
| `POST /predict/batch` | Score up to 1,000 transactions |
| `GET /model` | Model type, registry version, threshold, data fingerprint, test metrics |
| `GET /health` | Readiness check used by the container healthcheck |

Every scored transaction's features are appended to `logs/predictions.jsonl` for monitoring.

## Monitoring

`python -m fraud.drift --current logs/predictions.jsonl` compares live traffic with the training reference stored in the model bundle:

- **PSI per feature:** below 0.1 is stable, 0.1 to 0.2 is worth watching, above 0.2 is drift.
- **Kolmogorov Smirnov test:** a second opinion on each feature.
- **Score drift and alert rate:** a shift in model scores is what changes the workload for fraud analysts.

The report recommends retraining when a feature or the score distribution drifts. Samples under 1,000 rows are not judged, since PSI on small samples is noise. `--fail-on-drift` returns a non zero exit code, so a scheduled job can trigger retraining.

## CI

Every push runs the unit tests and trains both models on synthetic data as a pipeline smoke test. It then runs the drift check, builds the Docker image, and scores a request against the running container.

## Results

Trained on the full real dataset with a chronological split: 199,364 transactions for training, then validation (56 fraud) and test (52 fraud) on the later transactions. The threshold for each model was tuned on validation and applied unchanged to test. Full details are in [`reports/model_card.md`](reports/model_card.md).

| Model | Test PR AUC | Test ROC AUC | Precision | Recall | Fraud caught | False alerts |
|---|---|---|---|---|---|---|
| PyTorch MLP (champion) | 0.772 | 0.987 | 1.000 | 0.654 | 34 / 52 | 0 |
| XGBoost baseline | 0.756 | 0.980 | 0.867 | 0.750 | 39 / 52 | 6 |

The test set holds only 52 frauds, so one run could be luck. Across seeds 42, 1 and 2, the MLP's test PR AUC stayed between 0.772 and 0.777 and XGBoost's between 0.756 and 0.765, so the MLP won every time.

The two models fail differently, which matters more than the small PR AUC gap. At its F1 threshold the MLP raised no false alerts but missed more fraud. XGBoost caught 5 more frauds at the cost of 6 false alerts across 42,722 transactions. Which one is better depends on what a missed fraud costs against an analyst's time, and that is what `--min-precision` is for.

Both models score lower on test than on validation (PR AUC 0.87 to 0.77). The drift report explains part of that: run against the test window, it flags strong shifts in time of day features, because the last 15% of the data falls in the evening of day two. It also flags shifts in components such as V1, V3 and V12. Two days of data cannot cover a full daily cycle, and the monitoring catches exactly that.

## Project layout

```
fraud/data.py       loading, validation, chronological split, synthetic generator
fraud/features.py   feature engineering and JSON scaler
fraud/models.py     XGBoost and PyTorch candidates
fraud/evaluate.py   PR AUC, threshold selection, confusion metrics
fraud/train.py      training, MLflow tracking, registry, model card
fraud/bundle.py     the serving bundle
fraud/serve.py      FastAPI scoring service
fraud/drift.py      PSI and KS drift monitoring
```

Dataset: Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015 (ULB Machine Learning Group).
