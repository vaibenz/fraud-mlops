"""Data drift monitoring.

Compares current traffic against the training reference stored in the model
bundle, feature by feature, with two statistics:

  PSI (population stability index) on reference quantile bins
      < 0.1 stable, 0.1 to 0.2 moderate shift, > 0.2 significant shift
  Kolmogorov Smirnov test, as a second opinion on distribution change

It also compares the model's score distribution, since a shift in scores is
what actually changes the alert volume an operations team sees.

    python -m fraud.drift --current logs/predictions.jsonl
    python -m fraud.drift --current new_transactions.csv
    python -m fraud.drift --demo-shift            # simulated drift for a demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .bundle import Bundle
from .data import make_synthetic
from .features import MODEL_FEATURES, engineer

PSI_WARN, PSI_ALERT = 0.1, 0.2
MIN_ROWS = 1000  # PSI on a few hundred rows is mostly noise


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(v: float) -> str:
    return "drift" if v > PSI_ALERT else "watch" if v > PSI_WARN else "stable"


def compare(reference: pd.DataFrame, current: pd.DataFrame, bundle: Bundle | None = None,
            min_rows: int = MIN_ROWS) -> dict:
    if len(current) < min_rows:
        return {"n_reference": len(reference), "n_current": len(current), "features": [], "drifted": [],
                "watch": [], "insufficient_data": True, "retrain_recommended": False}
    rows = []
    for col in MODEL_FEATURES:
        r, c = reference[col].to_numpy(), current[col].to_numpy()
        ks = ks_2samp(r, c)
        p = psi(r, c)
        rows.append({"feature": col, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": float(ks.pvalue), "ref_mean": round(float(r.mean()), 4),
                     "cur_mean": round(float(c.mean()), 4), "status": _status(p)})
    rows.sort(key=lambda x: x["psi"], reverse=True)
    out = {
        "n_reference": len(reference),
        "n_current": len(current),
        "features": rows,
        "drifted": [r["feature"] for r in rows if r["status"] == "drift"],
        "watch": [r["feature"] for r in rows if r["status"] == "watch"],
    }
    if bundle is not None:
        ref_s = bundle.model.predict_proba(bundle.scaler.transform(reference[MODEL_FEATURES]))
        cur_s = bundle.model.predict_proba(bundle.scaler.transform(current[MODEL_FEATURES]))
        thr = bundle.meta["threshold"]
        out["score"] = {
            "psi": round(psi(ref_s, cur_s), 4),
            "ref_alert_rate": float((ref_s >= thr).mean()),
            "cur_alert_rate": float((cur_s >= thr).mean()),
        }
        out["score"]["status"] = _status(out["score"]["psi"])
    out["retrain_recommended"] = bool(out["drifted"]) or out.get("score", {}).get("status") == "drift"
    return out


def to_markdown(r: dict) -> str:
    lines = [
        "# Drift report",
        "",
        f"Reference rows: {r['n_reference']:,}. Current rows: {r['n_current']:,}.",
        "",
    ]
    if r.get("insufficient_data"):
        return "\n".join(lines + [f"Not enough current data to judge drift (need at least {MIN_ROWS:,} rows)."]) + "\n"
    lines += [
        f"**Retrain recommended: {'yes' if r['retrain_recommended'] else 'no'}.** "
        f"Drifted features: {', '.join(r['drifted']) or 'none'}. Watch: {', '.join(r['watch']) or 'none'}.",
        "",
    ]
    if "score" in r:
        s = r["score"]
        lines += [f"Model score PSI {s['psi']:.3f} ({s['status']}); alert rate {s['ref_alert_rate']:.3%} on reference vs "
                  f"{s['cur_alert_rate']:.3%} now.", ""]
    lines += ["| Feature | PSI | KS stat | Ref mean | Current mean | Status |", "|---|---|---|---|---|---|"]
    lines += [f"| {x['feature']} | {x['psi']:.3f} | {x['ks_stat']:.3f} | {x['ref_mean']:.3f} | {x['cur_mean']:.3f} | {x['status']} |"
              for x in r["features"][:12]]
    lines += ["", "Top 12 features by PSI. Full table in drift_report.json."]
    return "\n".join(lines) + "\n"


def load_current(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)[MODEL_FEATURES]
    return engineer(pd.read_csv(path))


def demo_shift(n: int = 20_000, seed: int = 99) -> pd.DataFrame:
    """Later traffic where spending rises and one component shifts, as after a new merchant category launches."""
    df = make_synthetic(n, seed=seed)
    df["Amount"] = df["Amount"] * 2.5
    df["V14"] = df["V14"] - 0.6
    return engineer(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/champion")
    ap.add_argument("--current", type=Path)
    ap.add_argument("--demo-shift", action="store_true")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--fail-on-drift", action="store_true")
    args = ap.parse_args()

    bundle = Bundle.load(args.model_dir)
    reference = pd.read_parquet(Path(args.model_dir) / "reference.parquet")
    if args.demo_shift:
        current = demo_shift()
    elif args.current:
        current = load_current(args.current)
    else:
        ap.error("pass --current PATH or --demo-shift")

    report = compare(reference, current, bundle)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2))
    (out / "drift_report.md").write_text(to_markdown(report))
    print(to_markdown(report))
    if args.fail_on_drift and report["retrain_recommended"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
