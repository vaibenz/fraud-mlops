# Model card: fraud-detector

**Champion:** `pytorch_mlp` (registry version 1, alias `champion`), chosen on validation PR AUC.

## Data

Source: `data/creditcard.csv` (fingerprint `89c01ac0ac68d8c9`), 284,807 transactions, 492 fraud (0.173%).
Chronological split 70/15/15: train 199,364 (384 fraud), validation 42,721 (56 fraud), test 42,722 (52 fraud).

## Results

Thresholds are tuned on validation (max F1) and applied unchanged to test.

| Model | Split | PR AUC | ROC AUC | Precision | Recall | F1 | Fraud caught | False alerts |
|---|---|---|---|---|---|---|---|---|
| xgboost | val | 0.855 | 0.982 | 0.936 | 0.786 | 0.854 | 44 / 56 | 3 |
| xgboost | test | 0.756 | 0.980 | 0.867 | 0.750 | 0.804 | 39 / 52 | 6 |
| pytorch_mlp | val | 0.872 | 0.986 | 1.000 | 0.768 | 0.869 | 43 / 56 | 0 |
| pytorch_mlp | test | 0.772 | 0.987 | 1.000 | 0.654 | 0.791 | 34 / 52 | 0 |

## Intended use and limits

Scores card transactions for fraud review. V1 to V28 are anonymised PCA components, so the model cannot be explained in terms of raw merchant or customer attributes. Two days of data cannot capture seasonal patterns, so drift should be monitored with `python -m fraud.drift` and the model retrained when key features shift.
