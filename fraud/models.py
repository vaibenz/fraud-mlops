"""The two candidate models.

XGBoost is the strong tabular baseline. The PyTorch MLP is the deep learning
candidate. Both handle the ~0.17% fraud rate by weighting the positive class
(scale_pos_weight / BCE pos_weight) instead of resampling, and both early stop
on validation PR AUC, the metric that matters when positives are this rare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score
from torch import nn


def pos_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


# ---------------------------------------------------------------- XGBoost
class XGBModel:
    name = "xgboost"

    def __init__(self, seed: int = 42, **params):
        self.params = {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 50,
            "random_state": seed,
            "n_jobs": 4,
            **params,
        }
        self.model: xgb.XGBClassifier | None = None

    def fit(self, X, y, X_val, y_val) -> "XGBModel":
        self.params["scale_pos_weight"] = pos_weight(y)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        self.best_iteration = int(self.model.best_iteration)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, d: Path) -> None:
        self.model.save_model(d / "model.ubj")

    @classmethod
    def load(cls, d: Path) -> "XGBModel":
        m = cls()
        m.model = xgb.XGBClassifier()
        m.model.load_model(d / "model.ubj")
        return m


# ---------------------------------------------------------------- PyTorch
class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchModel:
    name = "pytorch_mlp"

    def __init__(self, seed: int = 42, epochs: int = 30, batch_size: int = 2048, lr: float = 1e-3,
                 hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2, patience: int = 5):
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr
        self.hidden, self.dropout, self.patience = tuple(hidden), dropout, patience
        self.net: MLP | None = None
        self.history: list[dict] = []

    @property
    def params(self) -> dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
                "hidden": list(self.hidden), "dropout": self.dropout, "patience": self.patience}

    def fit(self, X, y, X_val, y_val) -> "TorchModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = MLP(X.shape[1], self.hidden, self.dropout)
        # Square root of the imbalance ratio: full weighting makes the MLP overpredict fraud.
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(y) ** 0.5))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            total = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            val_ap = average_precision_score(y_val, self.predict_proba(X_val)) if y_val.sum() else 0.0
            self.history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_pr_auc": val_ap})
            if val_ap > best:
                best, bad = val_ap, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmax([h["val_pr_auc"] for h in self.history]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.as_tensor(X, dtype=torch.float32))).numpy()

    def save(self, d: Path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "n_in": self.net.net[0].in_features,
                    "hidden": list(self.hidden), "dropout": self.dropout}, d / "model.pt")

    @classmethod
    def load(cls, d: Path) -> "TorchModel":
        ckpt = torch.load(d / "model.pt", weights_only=True)
        m = cls(hidden=tuple(ckpt["hidden"]), dropout=ckpt["dropout"])
        m.net = MLP(ckpt["n_in"], m.hidden, m.dropout)
        m.net.load_state_dict(ckpt["state_dict"])
        return m


MODEL_TYPES = {XGBModel.name: XGBModel, TorchModel.name: TorchModel}
