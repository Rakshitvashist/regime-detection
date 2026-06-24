"""
weekly_nifty50.py
=================
SELF-CONTAINED Weekly-Expiry forecasting pipeline for NIFTY 50 only.

This is the single dedicated file for the Weekly track. Weekly expiry on NSE is
one week away, i.e. ~5 trading days, so this model predicts the market regime
{bear, sideways, bull} 5 trading days ahead from the daily NIFTY data.

It reuses the project's existing, validated data:
  * output/features_NIFTY.json          (daily feature table from the Rust core)
  * output/regime_clustered_NIFTY.json  (regime labels)
and the shared feature pipeline (regime_features.py) so the features match the
rest of the project exactly.

Honesty (no look-ahead, same as research/xgb_regime.py):
  * features use only data up to the decision bar;
  * the regime target is shifted 5 bars into the future (forward-only, scoring);
  * evaluation is walk-forward out-of-fold (TimeSeriesSplit): train on past,
    predict the next unseen block. Accuracy is reported on those OOF predictions.

Usage:
  python weekly_nifty50.py                 # 5-day (weekly) horizon, NIFTY 50
  python weekly_nifty50.py --shift 5
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from regime_features import (XGB_FEATURES, REGIME_TO_INT, LABELS,
                             add_structural_features, assign_regime)

SYMBOL = "NIFTY"            # NIFTY 50
N_SPLITS = 5
BEAR_BOOST = 3.0           # up-weight the rare bear class (as in xgb_regime.py)


def parse_args():
    p = argparse.ArgumentParser(description="Weekly-expiry NIFTY 50 regime forecasting.")
    p.add_argument("--shift", type=int, default=5,
                   help="Forward trading days = weekly expiry horizon (default 5)")
    return p.parse_args()


def load(shift):
    """Load NIFTY features + labels, derive structural features, set the 5-day target."""
    with open(os.path.join(HERE, "output", f"features_{SYMBOL}.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    df = add_structural_features(df.sort_values("date").reset_index(drop=True))
    with open(os.path.join(HERE, "output", f"regime_clustered_{SYMBOL}.json")) as f:
        lab = pd.DataFrame(json.load(f))
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab[["date", "regime"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["target"] = df["regime"].shift(-shift)
    df = df.dropna(subset=["target"] + XGB_FEATURES).reset_index(drop=True)
    df["target_int"] = df["target"].map(REGIME_TO_INT)
    return df


def sample_weights(y):
    counts = y.value_counts().to_dict(); total = len(y)
    w = {c: total / n for c, n in counts.items()}
    if REGIME_TO_INT["bear"] in w:
        w[REGIME_TO_INT["bear"]] *= BEAR_BOOST
    return y.map(w).values


def _xgb():
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=1.0,
                         objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=42, verbosity=0)


def walk_forward(X, y, sw):
    probs = np.zeros((len(y), 3)); mask = np.zeros(len(y), dtype=bool)
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        sc = StandardScaler()
        m = _xgb()
        m.fit(sc.fit_transform(X[tr]), y[tr], sample_weight=sw[tr])
        probs[va] = m.predict_proba(sc.transform(X[va])); mask[va] = True
    return probs, mask


def main():
    args = parse_args()
    df = load(args.shift)
    print(f"=== Weekly Expiry — NIFTY 50 | horizon={args.shift} trading days ===")
    print(f"Usable rows: {len(df)} | {df['date'].min().date()} -> {df['date'].max().date()}")
    mix = df["target"].value_counts(normalize=True).round(3)
    print("Class mix:", {k: float(mix.get(k, 0.0)) for k in LABELS})

    X = df[XGB_FEATURES].values
    y = df["target_int"].values
    sw = sample_weights(df["target_int"])

    probs, mask = walk_forward(X, y, sw)
    yt = y[mask]
    # served regime uses assign_regime (confidence gate -> sideways fallback)
    pred = np.array([REGIME_TO_INT[assign_regime(p)] for p in probs[mask]])
    acc = accuracy_score(yt, pred) * 100
    maj = df["target"].value_counts(normalize=True).max() * 100
    print(f"\nWalk-forward OOF accuracy : {acc:.2f}%")
    print(f"Majority-class baseline   : {maj:.2f}%   (edge = {acc - maj:+.2f} pts)")
    print(f"Macro-F1                  : {f1_score(yt, probs[mask].argmax(1), average='macro'):.3f}\n")
    print(classification_report(yt, pred, target_names=LABELS, zero_division=0))

    out = df.loc[mask, ["date", "close"]].copy()
    out["pred_regime"] = [assign_regime(p) for p in probs[mask]]
    out["actual_regime"] = df.loc[mask, "target"].values
    out[["prob_bear", "prob_sideways", "prob_bull"]] = probs[mask]
    out["confidence"] = probs[mask].max(1)
    out["correct"] = out["pred_regime"] == out["actual_regime"]
    out_path = os.path.join(HERE, "weekly_nifty50_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
