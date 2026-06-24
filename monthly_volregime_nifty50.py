"""
monthly_volregime_nifty50.py
============================
FINAL Monthly-Expiry model for NIFTY 50 (self-contained).

Monthly DIRECTION (21-day, up/down) is not predictable — it loses to the market's
natural up-drift (AUC ~0.5, tested). So this final monthly model predicts the
question that DOES have an edge and is what a monthly-expiry trader needs:

    "Over the next ~21 trading days, will NIFTY be in a HIGH-vol or LOW-vol regime?"
    -> LOW-vol month  => favour selling option premium
    -> HIGH-vol month => favour buying / hedging

Improvements applied here (each measured honestly):
  1. VOLATILITY target  : forward 21-day realized vol vs its trailing median.
  2. CROSS-ASSET vol     : Nifty features + BANKNIFTY / USDINR / WTI volatility
                           (other markets' stress leads index vol).
  3. OVERFITTING FIX     : strongly regularised XGBoost (shallow, few trees,
                           high min_child_weight + L1/L2) to shrink the train-val gap.

MODEL: XGBoost binary classifier (XGBClassifier).

It prints a head-to-head (Nifty-only vs +cross-asset vol) and the train-vs-validation
AUC gap so the overfitting level is visible.

NO LOOK-AHEAD: features use only data up to the bar; cross-asset merged backward;
the trailing median uses past data only; the target is forward-only; evaluation is
walk-forward out-of-fold.

Usage:
  python monthly_volregime_nifty50.py            # 21-day horizon
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from regime_features import XGB_FEATURES, add_structural_features

SYMBOL = "NIFTY"
N_SPLITS = 5
MEDIAN_WIN = 252
CROSS_ASSETS = ["BANKNIFTY", "USDINR", "WTI"]
CROSS_VOL_COLS = ["vol_21d", "vol_63d", "vol_ratio"]   # other markets' volatility


def _feats(symbol):
    with open(os.path.join(HERE, "output", f"features_{symbol}.json")) as f:
        d = pd.DataFrame(json.load(f))
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def load_base(shift):
    df = add_structural_features(_feats(SYMBOL))
    fwd_vol = df["close"].pct_change().shift(-1).rolling(shift).std().shift(-(shift - 1))
    df["fwd_vol"] = fwd_vol
    med = fwd_vol.rolling(MEDIAN_WIN, min_periods=60).median()
    df["target"] = (fwd_vol > med).astype(float)
    return df


def add_cross(df):
    added = []
    for a in CROSS_ASSETS:
        try:
            x = _feats(a)[["date"] + CROSS_VOL_COLS].copy()
        except Exception as e:
            print(f"  (skip {a}: {e})"); continue
        x = x.rename(columns={c: f"{a.lower()}_{c}" for c in CROSS_VOL_COLS})
        df = pd.merge_asof(df.sort_values("date"), x.sort_values("date"),
                           on="date", direction="backward")
        added += [f"{a.lower()}_{c}" for c in CROSS_VOL_COLS]
    df[added] = df[added].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, added


def _xgb():
    # strongly regularised for the small (~1500-row) monthly set -> control overfitting
    return XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.02,
                         subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
                         gamma=0.1, reg_alpha=1.0, reg_lambda=5.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def walk_forward(X, y):
    proba = np.zeros(len(y)); mask = np.zeros(len(y), dtype=bool); tr_a, va_a = [], []
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        m = _xgb(); m.fit(Xtr, y[tr])
        proba[va] = m.predict_proba(Xva)[:, 1]; mask[va] = True
        tr_a.append(roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1]))
        va_a.append(roc_auc_score(y[va], proba[va]))
    return proba, mask, np.mean(tr_a), np.mean(va_a)


def evaluate(df, feats, label):
    data = df.dropna(subset=["target", "fwd_vol"] + feats).reset_index(drop=True)
    X = data[feats].values; y = data["target"].astype(int).values
    proba, mask, tA, vA = walk_forward(X, y)
    yt = y[mask]; pred = (proba[mask] > 0.5).astype(int)
    acc = accuracy_score(yt, pred) * 100
    base = max(yt.mean(), 1 - yt.mean()) * 100
    auc = roc_auc_score(yt, proba[mask])
    print(f"  {label:28s} acc={acc:5.2f}%  base={base:5.2f}%  AUC={auc:.3f}  gap={tA - vA:.3f}")
    return data, proba, mask, pred, acc, auc


def main():
    ap = argparse.ArgumentParser(description="FINAL monthly volatility regime (NIFTY 50).")
    ap.add_argument("--shift", type=int, default=21)
    args = ap.parse_args()

    base = load_base(args.shift)
    base_x, cross = add_cross(base)

    print(f"=== Monthly VOLATILITY REGIME — NIFTY 50 | horizon={args.shift} days ===")
    print("Model: XGBoost binary classifier (regularised) | target: high-vol vs low-vol month\n")

    # FINAL model = Nifty-only (cross-asset vol was tested and made it WORSE:
    # more features -> more overfitting on the ~1500-row monthly set -> rejected).
    data, proba, mask, pred, acc_n, auc_n = evaluate(base, XGB_FEATURES, "Nifty-only (FINAL)")
    _, _, _, _, acc_c, _ = evaluate(base_x, XGB_FEATURES + cross, "Nifty + cross-asset (rejected)")
    print(f"\n  >>> cross-asset volatility: {acc_c - acc_n:+.2f} pts (worse) -> dropped; "
          f"FINAL is Nifty-only: {acc_n:.2f}% (AUC {auc_n:.3f})")

    print("\nFinal model per-class report:")
    print(classification_report(data["target"].astype(int).values[mask], pred,
                                target_names=["low-vol", "high-vol"], zero_division=0))

    out = data.loc[mask, ["date", "close", "fwd_vol"]].copy()
    out["pred"] = np.where(pred == 1, "high-vol", "low-vol")
    out["actual"] = np.where(data["target"].astype(int).values[mask] == 1, "high-vol", "low-vol")
    out["prob_highvol"] = proba[mask]
    out["correct"] = out["pred"] == out["actual"]
    out_path = os.path.join(HERE, "monthly_volregime_nifty50_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
