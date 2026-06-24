"""
monthly_nifty50.py
==================
SELF-CONTAINED Monthly-Expiry forecasting pipeline for NIFTY 50 only.

Monthly expiry on NSE is ~21 trading days away, so this model predicts the market
regime {bear, sideways, bull} 21 trading days ahead from daily NIFTY data.

Monthly is the HARD horizon (the Nifty-only model is ~38%). This file therefore
tries to IMPROVE it using OTHER data already available in the project: cross-asset
features from BANKNIFTY, USDINR and WTI crude. The market's monthly regime is
driven by more than Nifty's own chart (bank-sector leadership, currency, oil), so
those signals can add real predictive information.

It runs TWO models head-to-head on the same honest walk-forward split:
  * BASELINE : Nifty-only features (XGB_FEATURES)              -> the current ~38%
  * IMPROVED : Nifty features + cross-asset features           -> measured gain

No look-ahead: cross-asset values are taken as-of each Nifty bar with a BACKWARD
merge (only data on/before the bar date), the regime target is forward-only, and
evaluation is walk-forward out-of-fold (train past -> predict unseen future).

Usage:
  python monthly_nifty50.py                # 21-day (monthly) horizon, NIFTY 50
  python monthly_nifty50.py --shift 21
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

SYMBOL = "NIFTY"
N_SPLITS = 5
BEAR_BOOST = 3.0

# Other instruments + the columns to pull from each (momentum / vol / trend).
CROSS_ASSETS = ["BANKNIFTY", "USDINR", "WTI"]
CROSS_COLS = ["ret_21d", "ret_63d", "vol_ratio", "rsi_14", "ema_gap", "adx"]


def parse_args():
    p = argparse.ArgumentParser(description="Monthly-expiry NIFTY 50 regime forecasting.")
    p.add_argument("--shift", type=int, default=21, help="Forward trading days (monthly ~21)")
    return p.parse_args()


def _load_features(symbol):
    with open(os.path.join(HERE, "output", f"features_{symbol}.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_base(shift):
    """NIFTY features + labels + structural derivations + 21-day target."""
    df = add_structural_features(_load_features(SYMBOL))
    with open(os.path.join(HERE, "output", f"regime_clustered_{SYMBOL}.json")) as f:
        lab = pd.DataFrame(json.load(f))
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab[["date", "regime"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["target"] = df["regime"].shift(-shift)
    return df


def add_cross_assets(df):
    """Merge cross-asset features as-of each NIFTY bar (backward = no look-ahead)."""
    added = []
    for asset in CROSS_ASSETS:
        try:
            a = _load_features(asset)[["date"] + CROSS_COLS].copy()
        except Exception as e:
            print(f"  (skipping {asset}: {e})")
            continue
        a = a.rename(columns={c: f"{asset.lower()}_{c}" for c in CROSS_COLS})
        df = pd.merge_asof(df.sort_values("date"), a.sort_values("date"),
                           on="date", direction="backward")
        added += [f"{asset.lower()}_{c}" for c in CROSS_COLS]
    df[added] = df[added].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, added


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


def evaluate(df, feats, label):
    data = df.dropna(subset=["target"] + feats).reset_index(drop=True)
    data["target_int"] = data["target"].map(REGIME_TO_INT)
    X = data[feats].values
    y = data["target_int"].values
    sw = sample_weights(data["target_int"])
    probs, mask = walk_forward(X, y, sw)
    yt = y[mask]
    pred = np.array([REGIME_TO_INT[assign_regime(p)] for p in probs[mask]])
    acc = accuracy_score(yt, pred) * 100
    f1 = f1_score(yt, probs[mask].argmax(1), average="macro")
    print(f"  {label:28s} | rows={mask.sum():4d} | acc={acc:5.2f}%  macro-F1={f1:.3f}")
    return data, probs, mask, pred, acc


def main():
    args = parse_args()
    base = load_base(args.shift)
    base_x, cross_feats = add_cross_assets(base)

    print(f"=== Monthly Expiry — NIFTY 50 | horizon={args.shift} trading days ===")
    print(f"Rows: {len(base)} | {base['date'].min().date()} -> {base['date'].max().date()}")
    maj = base["target"].dropna().value_counts(normalize=True).max() * 100
    print(f"Majority-class baseline: {maj:.2f}%\n")

    # head-to-head
    _, _, _, _, acc_base = evaluate(base, XGB_FEATURES, "BASELINE (Nifty only)")
    improved_feats = XGB_FEATURES + cross_feats
    data, probs, mask, pred, acc_impr = evaluate(base_x, improved_feats,
                                                 "IMPROVED (+cross-asset)")

    print(f"\n  >>> Cross-asset features: {acc_base:.2f}% -> {acc_impr:.2f}%  "
          f"(delta {acc_impr - acc_base:+.2f} pts)")

    best = data.loc[mask].copy()
    print("\nIMPROVED per-class report:")
    print(classification_report(data["target_int"].values[mask], pred,
                                target_names=LABELS, zero_division=0))

    out = best[["date", "close"]].copy()
    out["pred_regime"] = [assign_regime(p) for p in probs[mask]]
    out["actual_regime"] = best["target"].values
    out[["prob_bear", "prob_sideways", "prob_bull"]] = probs[mask]
    out["confidence"] = probs[mask].max(1)
    out["correct"] = out["pred_regime"] == out["actual_regime"]
    out_path = os.path.join(HERE, "monthly_nifty50_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
