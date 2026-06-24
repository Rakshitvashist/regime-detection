"""
intraday_regime.py
==================
SEPARATE intraday forecasting model for the Intraday track (Nifty50 only).

Self-contained on purpose: the daily regime_features pipeline annualises vol with
sqrt(252) and uses daily lookbacks, which are wrong for intraday bars. This module
builds its OWN intraday features and target so the intraday track stays isolated
from the daily weekly/monthly pipeline.

Task: predict the intraday DIRECTION over the next `HORIZON` bars, as a 3-class
regime {bear, sideways, bull}, using only information available at the decision bar.

Honesty (no look-ahead):
  * features use only past/current bars (rolling windows, all shifted appropriately);
  * target is a FORWARD return -> never fed as a feature;
  * evaluation is walk-forward out-of-fold (TimeSeriesSplit): train on past bars,
    predict the next unseen block. Accuracy is reported on those OOF predictions.

Usage:
  python intraday_regime.py                 # 60m bars, horizon=4 (default)
  python intraday_regime.py --interval 15m --horizon 8
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["bear", "sideways", "bull"]
REGIME_TO_INT = {"bear": 0, "sideways": 1, "bull": 2}
N_SPLITS = 5

FEATURES = [
    "ret_1", "ret_2", "ret_4", "mom_12",
    "vol_12", "vol_24", "range_pct",
    "rsi_14", "dist_sma_24", "bar_of_day",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="60m", choices=["15m", "60m"],
                   help="Intraday bar size (file NSE_NIFTY_intraday_<interval>.csv)")
    p.add_argument("--horizon", type=int, default=4,
                   help="Forward bars to predict direction over (default 4)")
    p.add_argument("--threshold", type=float, default=0.0015,
                   help="Forward-return band for 'sideways' (default 0.15%%)")
    return p.parse_args()


def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def build_features(df):
    """Intraday features from OHLC bars; all use only past/current data."""
    c = df["close"]
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_4"] = c.pct_change(4)
    df["mom_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_24"] = df["ret_1"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]) / c
    df["rsi_14"] = rsi(c, 14)
    df["dist_sma_24"] = (c - c.rolling(24).mean()) / c.rolling(24).mean()
    # bar position within the trading day (intraday seasonality)
    df["bar_of_day"] = df.groupby(df["time"].dt.date).cumcount()
    return df


def make_target(df, horizon, threshold):
    """3-class direction over the next `horizon` bars (forward; for scoring only)."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    df["fwd_ret"] = fwd
    df["target"] = np.where(fwd > threshold, "bull",
                            np.where(fwd < -threshold, "bear", "sideways"))
    return df


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=1.0,
                         objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=42, verbosity=0)


def walk_forward(X, y, sw):
    probs = np.zeros((len(y), 3)); mask = np.zeros(len(y), dtype=bool)
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        m = xgb(); m.fit(Xtr, y[tr], sample_weight=sw[tr])
        probs[va] = m.predict_proba(Xva); mask[va] = True
    return probs, mask


def main():
    args = parse_args()
    path = os.path.join(HERE, f"NSE_NIFTY_intraday_{args.interval}.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run fetch_intraday.py first.")
        return 1

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = build_features(df)
    df = make_target(df, args.horizon, args.threshold)

    data = df.dropna(subset=FEATURES + ["target", "fwd_ret"]).reset_index(drop=True)
    data["target_int"] = data["target"].map(REGIME_TO_INT)
    print(f"--- Intraday Regime (Nifty50, {args.interval}, horizon={args.horizon} bars) ---")
    print(f"Usable bars: {len(data)} | {data['time'].min()} -> {data['time'].max()}")
    mix = data["target"].value_counts(normalize=True).round(3)
    print("Class mix:", {k: float(mix.get(k, 0.0)) for k in LABELS})

    X = data[FEATURES].values
    y = data["target_int"].values
    # balance classes so the model doesn't just predict the majority
    counts = pd.Series(y).value_counts().to_dict()
    w = {c: len(y) / n for c, n in counts.items()}
    sw = np.array([w[v] for v in y])

    probs, mask = walk_forward(X, y, sw)
    yt = y[mask]; pred = probs[mask].argmax(1)
    acc = accuracy_score(yt, pred) * 100
    f1 = f1_score(yt, pred, average="macro")
    maj = max(counts.values()) / len(y) * 100
    print(f"\nHonest walk-forward OOF accuracy : {acc:.2f}%")
    print(f"Majority-class baseline          : {maj:.2f}%")
    print(f"Macro-F1                         : {f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(yt, pred, target_names=LABELS, zero_division=0))

    out = data.loc[mask, ["time", "close", "fwd_ret"]].copy()
    out["pred_regime"] = [LABELS[i] for i in pred]
    out["actual_regime"] = data.loc[mask, "target"].values
    out[["prob_bear", "prob_sideways", "prob_bull"]] = probs[mask]
    out["confidence"] = probs[mask].max(1)
    out["correct"] = out["pred_regime"] == out["actual_regime"]
    out_path = os.path.join(HERE, f"intraday_{args.interval}_NIFTY_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} OOF predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
