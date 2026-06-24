"""
intraday_trendchop_nifty50.py
=============================
SELF-CONTAINED intraday "Trend vs Chop" model for NIFTY 50, with India VIX.

WHY THIS EXISTS
Predicting intraday DIRECTION from price is near-random (our direction model sat at
~38%, no edge). So this file changes the QUESTION to one that IS predictable and is
exactly what an expiry-options trader needs:

    "Over the next few hours, will NIFTY TREND (strong directional move) or CHOP
     (stay range-bound)?"

This works because volatility CLUSTERS and trend/chop states PERSIST — unlike raw
direction. India VIX (the market's fear/vol gauge) adds the single best external
signal for this regime.

MODEL: XGBoost binary classifier (XGBClassifier) — gradient-boosted decision trees.

TARGET (forward-only, for scoring): Kaufman Efficiency Ratio (ER) over the next H
bars. ER = |net move| / |path length| in [0,1]; ~1 = clean trend, ~0 = chop.
    label = 1 (TREND) if forward ER > --er-threshold, else 0 (CHOP).

NO LOOK-AHEAD: features use only past/current bars (incl. PAST efficiency ratio and
realized vol); VIX is merged bar-aligned; ER target is forward-only; evaluation is
walk-forward out-of-fold (train past -> predict unseen future).

Usage:
  python intraday_trendchop_nifty50.py                  # 60m, H=4, fetches if missing
  python intraday_trendchop_nifty50.py --fetch
  python intraday_trendchop_nifty50.py --horizon 6 --er-threshold 0.5
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
NIFTY, VIX = "^NSEI", "^INDIAVIX"
N_SPLITS = 5
PERIOD = {"15m": "60d", "60m": "730d"}

FEATURES = [
    "ret_1", "ret_2", "ret_4", "mom_12",
    "vol_12", "vol_24", "atr_pct", "range_pct",
    "rsi_14", "dist_sma_24", "bar_of_day",
    "er_past_12",                       # trend persistence
    "vix", "vix_chg", "vix_z",          # India VIX level / change / z-score
]


# ----------------------------------------------------------- fetch
def _fetch(ticker, interval, out_name):
    import yfinance as yf
    df = yf.download(ticker, period=PERIOD[interval], interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {ticker} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "Volume"})
    keep = [c for c in ["time", "open", "high", "low", "close"] if c in df.columns]
    df = df[keep].dropna().reset_index(drop=True)
    path = os.path.join(HERE, out_name)
    df.to_csv(path, index=False)
    print(f"  saved {out_name}: {len(df)} bars")
    return path


def load_or_fetch(ticker, interval, out_name, force):
    path = os.path.join(HERE, out_name)
    if force or not os.path.exists(path):
        print(f"Fetching {ticker} {interval}...")
        _fetch(ticker, interval, out_name)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


# ----------------------------------------------------------- features / target
def _rsi(close, p=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def efficiency_ratio(close, window):
    """Kaufman ER over a trailing `window`: |net change| / sum(|bar changes|)."""
    net = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return (net / (path + 1e-12)).clip(0, 1)


def forward_er(close, horizon):
    """ER over the NEXT `horizon` bars (forward; target only)."""
    net = (close.shift(-horizon) - close).abs()
    path = close.diff().abs().shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return (net / (path + 1e-12)).clip(0, 1)


def build(df_nifty, df_vix, horizon, er_th):
    df = df_nifty.copy()
    c = df["close"]
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_4"] = c.pct_change(4)
    df["mom_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_24"] = df["ret_1"].rolling(24).std()
    tr = (df["high"] - df["low"]).abs()
    df["atr_pct"] = tr.rolling(14).mean() / c
    df["range_pct"] = (df["high"] - df["low"]) / c
    df["rsi_14"] = _rsi(c, 14)
    df["dist_sma_24"] = (c - c.rolling(24).mean()) / c.rolling(24).mean()
    df["bar_of_day"] = df.groupby(df["time"].dt.date).cumcount()
    df["er_past_12"] = efficiency_ratio(c, 12)

    # India VIX, bar-aligned (as-of backward = no look-ahead)
    v = df_vix[["time", "close"]].rename(columns={"close": "vix"})
    df = pd.merge_asof(df.sort_values("time"), v.sort_values("time"),
                       on="time", direction="backward")
    df["vix_chg"] = df["vix"].pct_change()
    df["vix_z"] = (df["vix"] - df["vix"].rolling(48).mean()) / (df["vix"].rolling(48).std() + 1e-9)

    # forward target: TREND (1) vs CHOP (0)
    df["fwd_er"] = forward_er(c, horizon)
    df["target"] = (df["fwd_er"] > er_th).astype(float)
    return df


def _xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=1.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def walk_forward(X, y):
    proba = np.zeros(len(y)); mask = np.zeros(len(y), dtype=bool)
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        sc = StandardScaler()
        m = _xgb()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        proba[va] = m.predict_proba(sc.transform(X[va]))[:, 1]; mask[va] = True
    return proba, mask


def main():
    ap = argparse.ArgumentParser(description="Intraday Trend-vs-Chop for NIFTY 50 + India VIX.")
    ap.add_argument("--interval", default="60m", choices=["15m", "60m"])
    ap.add_argument("--horizon", type=int, default=4, help="forward bars for the ER target")
    ap.add_argument("--er-threshold", type=float, default=0.5, help="ER>th = TREND else CHOP")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    nifty = load_or_fetch(NIFTY, args.interval, f"NSE_NIFTY_intraday_{args.interval}.csv", args.fetch)
    vix = load_or_fetch(VIX, args.interval, f"NSE_INDIAVIX_intraday_{args.interval}.csv", args.fetch)
    df = build(nifty, vix, args.horizon, args.er_threshold)

    data = df.dropna(subset=FEATURES + ["target", "fwd_er"]).reset_index(drop=True)
    print(f"\n=== Intraday TREND vs CHOP — NIFTY 50 | {args.interval} | "
          f"H={args.horizon} bars | ER>{args.er_threshold} ===")
    print(f"Model: XGBoost binary classifier (gradient-boosted trees)")
    print(f"Usable bars: {len(data)} | {data['time'].min()} -> {data['time'].max()}")
    bal = data["target"].mean()
    print(f"Class mix: trend={bal:.3f}  chop={1-bal:.3f}")

    X = data[FEATURES].values
    y = data["target"].values.astype(int)
    proba, mask = walk_forward(X, y)
    yt = y[mask]; pred = (proba[mask] > 0.5).astype(int)
    acc = accuracy_score(yt, pred) * 100
    maj = max(yt.mean(), 1 - yt.mean()) * 100
    auc = roc_auc_score(yt, proba[mask])
    print(f"\nWalk-forward OOF accuracy : {acc:.2f}%")
    print(f"Majority-class baseline   : {maj:.2f}%   (edge = {acc - maj:+.2f} pts)")
    print(f"ROC-AUC                   : {auc:.3f}   (0.5 = no skill)")
    print(f"F1                        : {f1_score(yt, pred):.3f}\n")
    print(classification_report(yt, pred, target_names=["chop", "trend"], zero_division=0))

    out = data.loc[mask, ["time", "close", "vix", "fwd_er"]].copy()
    out["pred"] = np.where(pred == 1, "trend", "chop")
    out["actual"] = np.where(yt == 1, "trend", "chop")
    out["prob_trend"] = proba[mask]
    out["correct"] = out["pred"] == out["actual"]
    out_path = os.path.join(HERE, f"intraday_trendchop_{args.interval}_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
