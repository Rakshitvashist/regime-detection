"""
monthly_v2_nifty50.py
=====================
IMPROVED Monthly-Expiry model for NIFTY 50 (self-contained).

Monthly (21-day) 3-class direction was the hard horizon (~40%, below baseline).
This v2 applies the highest-impact fixes, measured honestly:

  1. 2-CLASS up/down  : predict whether NIFTY is HIGHER or LOWER in 21 trading days
                        (drops the fuzzy "sideways" class that hurt the 3-class model).
  2. CLEAN target     : defined directly from the 21-day forward RETURN sign,
                        not from the noisy clustered regime labels.
  3. CROSS-ASSET      : Nifty features + BANKNIFTY / USDINR / WTI signals
                        (merged as-of each bar; already worth +2 pts before).
  4. TUNE FOR ACCURACY: balanced (not bear-boosted) and regularised XGBoost.

MODEL: XGBoost binary classifier (XGBClassifier).

It prints a head-to-head: Nifty-only 2-class vs Nifty + cross-asset 2-class, plus
the old 3-class number for context, all on the same honest walk-forward split.

NO LOOK-AHEAD: features use only data up to the bar; cross-asset merged backward;
target is the forward 21-day return (scoring only); evaluation is walk-forward OOF.

Usage:
  python monthly_v2_nifty50.py                 # 21-day horizon
  python monthly_v2_nifty50.py --shift 21
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
CROSS_ASSETS = ["BANKNIFTY", "USDINR", "WTI"]
CROSS_COLS = ["ret_21d", "ret_63d", "ret_126d", "vol_ratio", "rsi_14", "ema_gap", "adx"]


def _load_features(symbol):
    with open(os.path.join(HERE, "output", f"features_{symbol}.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_base(shift):
    df = add_structural_features(_load_features(SYMBOL))
    # clean 2-class target: is NIFTY higher (1) or lower (0) in `shift` days?
    df["fwd_ret"] = df["close"].shift(-shift) / df["close"] - 1.0
    df["target"] = (df["fwd_ret"] > 0).astype(float)
    return df


def add_cross(df):
    added = []
    for a in CROSS_ASSETS:
        try:
            x = _load_features(a)[["date"] + CROSS_COLS].copy()
        except Exception as e:
            print(f"  (skip {a}: {e})"); continue
        x = x.rename(columns={c: f"{a.lower()}_{c}" for c in CROSS_COLS})
        df = pd.merge_asof(df.sort_values("date"), x.sort_values("date"),
                           on="date", direction="backward")
        added += [f"{a.lower()}_{c}" for c in CROSS_COLS]
    df[added] = df[added].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, added


def _xgb():
    # balanced + regularised (tuned for accuracy, overfitting-aware)
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=3.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def walk_forward(X, y):
    proba = np.zeros(len(y)); mask = np.zeros(len(y), dtype=bool)
    tr_auc, va_auc = [], []
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        m = _xgb(); m.fit(Xtr, y[tr])
        proba[va] = m.predict_proba(Xva)[:, 1]; mask[va] = True
        tr_auc.append(roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1]))
        va_auc.append(roc_auc_score(y[va], proba[va]))
    return proba, mask, np.mean(tr_auc), np.mean(va_auc)


def evaluate(df, feats, label):
    data = df.dropna(subset=["target", "fwd_ret"] + feats).reset_index(drop=True)
    X = data[feats].values; y = data["target"].astype(int).values
    proba, mask, tA, vA = walk_forward(X, y)
    yt = y[mask]; pred = (proba[mask] > 0.5).astype(int)
    acc = accuracy_score(yt, pred) * 100
    base = max(yt.mean(), 1 - yt.mean()) * 100
    auc = roc_auc_score(yt, proba[mask])
    print(f"  {label:30s} acc={acc:5.2f}%  base={base:5.2f}%  AUC={auc:.3f}  "
          f"(train-val AUC gap={tA - vA:.3f})")
    return data, proba, mask, pred, acc, auc


def main():
    ap = argparse.ArgumentParser(description="Improved monthly (2-class) NIFTY 50.")
    ap.add_argument("--shift", type=int, default=21)
    args = ap.parse_args()

    base = load_base(args.shift)
    base_x, cross_feats = add_cross(base)

    print(f"=== Monthly Expiry v2 (2-class up/down) — NIFTY 50 | horizon={args.shift} days ===")
    print("Model: XGBoost binary classifier | target: up vs down over 21 days\n")
    up = base["target"].dropna().mean()
    print(f"Class mix: up={up:.3f}  down={1-up:.3f}  (baseline = predict 'up' always = {max(up,1-up)*100:.1f}%)\n")

    _, _, _, _, acc_n, _ = evaluate(base, XGB_FEATURES, "Nifty-only (2-class)")
    data, proba, mask, pred, acc_c, auc_c = evaluate(base_x, XGB_FEATURES + cross_feats,
                                                     "Nifty + cross-asset (2-class)")
    print(f"\n  >>> 3-class (old) was ~40.6%.  2-class now: {acc_c:.2f}%  "
          f"(cross-asset adds {acc_c - acc_n:+.2f} pts)")

    print("\nBest model per-class report:")
    print(classification_report(data["target"].astype(int).values[mask], pred,
                                target_names=["down", "up"], zero_division=0))

    out = data.loc[mask, ["date", "close", "fwd_ret"]].copy()
    out["pred"] = np.where(pred == 1, "up", "down")
    out["actual"] = np.where(data["target"].astype(int).values[mask] == 1, "up", "down")
    out["prob_up"] = proba[mask]
    out["correct"] = out["pred"] == out["actual"]
    out_path = os.path.join(HERE, "monthly_v2_nifty50_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
