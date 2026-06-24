"""
intraday_nifty50.py
===================
SELF-CONTAINED intraday forecasting pipeline for NIFTY 50 only.

This is the single dedicated file for the Intraday track. It is fully isolated
from the daily weekly/monthly pipeline: it fetches its own intraday bars, builds
its own intraday features, trains its own model, and writes its own output.

Pipeline (all in this one file):
  1. fetch   : pull real NIFTY 50 intraday bars from yfinance (15m or 60m)
  2. features: intraday-scale technical features (only past/current data)
  3. target  : 3-class direction {bear, sideways, bull} over the next H bars
  4. model   : XGBoost, honest walk-forward (TimeSeriesSplit) out-of-fold
  5. output  : accuracy report + per-bar predictions CSV

No look-ahead: features never see the future; the target is forward-only and used
solely for scoring; evaluation trains on past bars and predicts the next unseen block.

Usage:
  python intraday_nifty50.py                       # 60m bars, horizon=4, fetches if missing
  python intraday_nifty50.py --fetch               # force re-download intraday data
  python intraday_nifty50.py --interval 15m --horizon 8
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
TICKER = "^NSEI"                         # NIFTY 50
LABELS = ["bear", "sideways", "bull"]
REGIME_TO_INT = {"bear": 0, "sideways": 1, "bull": 2}
N_SPLITS = 5
PERIOD = {"15m": "60d", "60m": "730d"}   # yfinance intraday history limits

FEATURES = [
    "ret_1", "ret_2", "ret_4", "mom_12",
    "vol_12", "vol_24", "range_pct",
    "rsi_14", "dist_sma_24", "bar_of_day",
]


# ----------------------------------------------------------------------------- 1. FETCH
def fetch_intraday(interval):
    """Download NIFTY 50 intraday bars and save them; returns the CSV path."""
    import yfinance as yf
    print(f"Fetching {TICKER} {interval} bars (period={PERIOD[interval]})...")
    df = yf.download(TICKER, period=PERIOD[interval], interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no intraday data returned for {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "Volume"})
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["time", "open", "high", "low", "close", "Volume"]].dropna(
        subset=["open", "high", "low", "close"]).reset_index(drop=True)
    path = os.path.join(HERE, f"NSE_NIFTY_intraday_{interval}.csv")
    df.to_csv(path, index=False)
    print(f"  saved {os.path.basename(path)}: {len(df)} bars "
          f"| {df['time'].min()} -> {df['time'].max()}")
    return path


# ----------------------------------------------------------------------------- 2. FEATURES
def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + up / (down + 1e-12))


def build_features(df):
    c = df["close"]
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_4"] = c.pct_change(4)
    df["mom_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_24"] = df["ret_1"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]) / c
    df["rsi_14"] = _rsi(c, 14)
    df["dist_sma_24"] = (c - c.rolling(24).mean()) / c.rolling(24).mean()
    df["bar_of_day"] = df.groupby(df["time"].dt.date).cumcount()
    return df


# ----------------------------------------------------------------------------- 3. TARGET
def make_target(df, horizon, threshold):
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    df["fwd_ret"] = fwd
    df["target"] = np.where(fwd > threshold, "bull",
                            np.where(fwd < -threshold, "bear", "sideways"))
    return df


# ----------------------------------------------------------------------------- 4. MODEL
def _xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
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


# ----------------------------------------------------------------------------- DRIVER
def parse_args():
    p = argparse.ArgumentParser(description="Intraday NIFTY 50 forecasting (self-contained).")
    p.add_argument("--interval", default="60m", choices=["15m", "60m"])
    p.add_argument("--horizon", type=int, default=4, help="forward bars to predict")
    p.add_argument("--threshold", type=float, default=0.0015, help="sideways band (0.15%%)")
    p.add_argument("--fetch", action="store_true", help="force re-download intraday data")
    return p.parse_args()


def main():
    args = parse_args()
    path = os.path.join(HERE, f"NSE_NIFTY_intraday_{args.interval}.csv")
    if args.fetch or not os.path.exists(path):
        path = fetch_intraday(args.interval)

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = build_features(df.sort_values("time").reset_index(drop=True))
    df = make_target(df, args.horizon, args.threshold)

    data = df.dropna(subset=FEATURES + ["target", "fwd_ret"]).reset_index(drop=True)
    data["target_int"] = data["target"].map(REGIME_TO_INT)

    print(f"\n=== Intraday NIFTY 50 | {args.interval} | horizon={args.horizon} bars ===")
    print(f"Usable bars: {len(data)} | {data['time'].min()} -> {data['time'].max()}")
    mix = data["target"].value_counts(normalize=True).round(3)
    print("Class mix:", {k: float(mix.get(k, 0.0)) for k in LABELS})

    X = data[FEATURES].values
    y = data["target_int"].values
    counts = pd.Series(y).value_counts().to_dict()
    w = {c: len(y) / n for c, n in counts.items()}
    sw = np.array([w[v] for v in y])

    probs, mask = walk_forward(X, y, sw)
    yt = y[mask]; pred = probs[mask].argmax(1)
    acc = accuracy_score(yt, pred) * 100
    maj = max(counts.values()) / len(y) * 100
    print(f"\nWalk-forward OOF accuracy : {acc:.2f}%")
    print(f"Majority-class baseline   : {maj:.2f}%   (edge = {acc - maj:+.2f} pts)")
    print(f"Macro-F1                  : {f1_score(yt, pred, average='macro'):.3f}\n")
    print(classification_report(yt, pred, target_names=LABELS, zero_division=0))

    out = data.loc[mask, ["time", "close", "fwd_ret"]].copy()
    out["pred_regime"] = [LABELS[i] for i in pred]
    out["actual_regime"] = data.loc[mask, "target"].values
    out[["prob_bear", "prob_sideways", "prob_bull"]] = probs[mask]
    out["confidence"] = probs[mask].max(1)
    out["correct"] = out["pred_regime"] == out["actual_regime"]
    out_path = os.path.join(HERE, f"intraday_nifty50_{args.interval}_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
