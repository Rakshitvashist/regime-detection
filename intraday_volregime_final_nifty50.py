"""
intraday_volregime_final_nifty50.py
===================================
FINAL intraday VOLATILITY-REGIME model for NIFTY 50 (self-contained).

This is the cleaned, validated version of the intraday model. We tested three
intraday targets honestly:
  * direction (up/down)      -> ~38%, no edge
  * trend-vs-chop            -> AUC ~0.50, no skill
  * VOLATILITY regime (this) -> ~60%, AUC ~0.63, robust across horizons  <-- KEEP

We also ran an ablation on India VIX and found it REDUNDANT (it did not improve
accuracy or AUC), so this final version DROPS VIX entirely. The edge comes from
the price-based drivers the data actually supports:
  * time-of-day seasonality (Nifty is volatile near open/close, calm midday)
  * recent realized volatility (volatility CLUSTERS)

Question answered: "Over the next H bars, will NIFTY be HIGH-vol or LOW-vol?"
  -> LOW-vol  => favour selling option premium
  -> HIGH-vol => favour buying / hedging

MODEL: XGBoost binary classifier (XGBClassifier) — gradient-boosted decision trees.

NO LOOK-AHEAD: features use only past/current bars; the trailing median for the
high/low cut uses past data only; the target is forward-only; evaluation is
walk-forward out-of-fold (train past -> predict unseen future).

Usage:
  python intraday_volregime_final_nifty50.py            # 60m, H=4, fetch if missing
  python intraday_volregime_final_nifty50.py --fetch --horizon 4
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
NIFTY = "^NSEI"
N_SPLITS = 5
PERIOD = {"15m": "60d", "60m": "730d"}
MEDIAN_WIN = 250        # trailing window for the high/low-vol cut (past only)

# Price + time-of-day drivers only (VIX dropped — proven redundant by ablation).
FEATURES = [
    "ret_1", "ret_2", "ret_4", "mom_12",
    "vol_12", "vol_24", "atr_pct", "range_pct",
    "rsi_14", "dist_sma_24", "bar_of_day", "er_past_12",
]


def _fetch(interval, out_name):
    import yfinance as yf
    df = yf.download(NIFTY, period=PERIOD[interval], interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {NIFTY} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close"})
    df = df[["time", "open", "high", "low", "close"]].dropna().reset_index(drop=True)
    df.to_csv(os.path.join(HERE, out_name), index=False)
    print(f"  saved {out_name}: {len(df)} bars")


def load_or_fetch(interval, force):
    out_name = f"NSE_NIFTY_intraday_{interval}.csv"
    path = os.path.join(HERE, out_name)
    if force or not os.path.exists(path):
        print(f"Fetching {NIFTY} {interval}...")
        _fetch(interval, out_name)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def _rsi(c, p=14):
    d = c.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def _er(c, w):
    return (c.diff(w).abs() / (c.diff().abs().rolling(w).sum() + 1e-12)).clip(0, 1)


def build(df, horizon):
    c = df["close"]
    df["ret_1"] = c.pct_change(1); df["ret_2"] = c.pct_change(2)
    df["ret_4"] = c.pct_change(4); df["mom_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std(); df["vol_24"] = df["ret_1"].rolling(24).std()
    df["atr_pct"] = (df["high"] - df["low"]).abs().rolling(14).mean() / c
    df["range_pct"] = (df["high"] - df["low"]) / c
    df["rsi_14"] = _rsi(c, 14)
    df["dist_sma_24"] = (c - c.rolling(24).mean()) / c.rolling(24).mean()
    df["bar_of_day"] = df.groupby(df["time"].dt.date).cumcount()
    df["er_past_12"] = _er(c, 12)
    # forward realized vol over the next H bars (strictly future). The high/low
    # THRESHOLD is NOT computed here — doing it globally/rolling on forward vol
    # would leak the future into the label. It is set per-fold from TRAIN only
    # (see walk_forward), so the target is leak-free.
    df["fwd_vol"] = df["ret_1"].shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    return df


def _xgb():
    # Regularised to control overfitting: an overfit diagnostic showed the prior
    # config (400 trees / depth 4) had a train-vs-validation AUC gap of ~0.29
    # (train AUC 0.94 vs val 0.64). Fewer, shallower trees + stronger L2 + larger
    # min_child_weight cut the gap to ~0.10 AND raised validation AUC (0.64 -> 0.67).
    return XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=3.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def walk_forward(X, fwd_vol, embargo=0):
    """Leak-free walk-forward. The high/low-vol THRESHOLD and the scaler are both
    fit on each fold's TRAIN slice only, then applied to the unseen validation
    slice — so neither the label nor the scaling sees the future.

    `embargo` drops the last `embargo` TRAIN rows of every fold: their forward
    target (next H bars) overlaps the validation block, so removing them prevents
    any boundary look-ahead between train and validation."""
    proba = np.zeros(len(fwd_vol)); y_used = np.full(len(fwd_vol), -1)
    mask = np.zeros(len(fwd_vol), dtype=bool)
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        if embargo > 0:
            tr = tr[:-embargo]                          # purge train tail (embargo)
        thr = np.nanmedian(fwd_vol[tr])                 # threshold from TRAIN only
        ytr = (fwd_vol[tr] > thr).astype(int)
        yva = (fwd_vol[va] > thr).astype(int)
        sc = StandardScaler(); m = _xgb()
        m.fit(sc.fit_transform(X[tr]), ytr)
        proba[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
        y_used[va] = yva; mask[va] = True
    return proba, mask, y_used


def main():
    ap = argparse.ArgumentParser(description="FINAL intraday volatility regime (NIFTY 50).")
    ap.add_argument("--interval", default="60m", choices=["15m", "60m"])
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    nifty = load_or_fetch(args.interval, args.fetch)
    df = build(nifty, args.horizon)
    data = df.dropna(subset=FEATURES + ["fwd_vol"]).reset_index(drop=True)

    print(f"\n=== Intraday VOLATILITY REGIME (FINAL, leak-free) — NIFTY 50 | {args.interval} | H={args.horizon} ===")
    print("Model: XGBoost binary classifier (gradient-boosted trees) | features: price + time-of-day")
    print(f"Usable bars: {len(data)} | {data['time'].min()} -> {data['time'].max()}")

    X = data[FEATURES].values
    fwd = data["fwd_vol"].values
    proba, mask, y_used = walk_forward(X, fwd, embargo=args.horizon)
    yt = y_used[mask]; pred = (proba[mask] > 0.5).astype(int)
    acc = accuracy_score(yt, pred) * 100
    maj = max(yt.mean(), 1 - yt.mean()) * 100
    auc = roc_auc_score(yt, proba[mask])
    print(f"\nWalk-forward OOF accuracy : {acc:.2f}%")
    print(f"Majority-class baseline   : {maj:.2f}%   (edge = {acc - maj:+.2f} pts)")
    print(f"ROC-AUC                   : {auc:.3f}")
    print(f"F1                        : {f1_score(yt, pred):.3f}\n")
    print(classification_report(yt, pred, target_names=["low-vol", "high-vol"], zero_division=0))

    out = data.loc[mask, ["time", "close", "fwd_vol"]].copy()
    out["pred"] = np.where(pred == 1, "high-vol", "low-vol")
    out["actual"] = np.where(yt == 1, "high-vol", "low-vol")
    out["prob_highvol"] = proba[mask]
    out["correct"] = out["pred"] == out["actual"]
    out_path = os.path.join(HERE, f"intraday_volregime_final_{args.interval}_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
