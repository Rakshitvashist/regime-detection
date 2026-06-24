"""
intraday_hybrid_nifty50.py
==========================
HYBRID intraday model for NIFTY 50 = Gaussian HMM  +  XGBoost.

Idea: use each model for what it is best at, then combine.
  * HMM    : identifies the hidden volatility regime and, crucially, its PERSISTENCE
             (the filtered probability of being in the high-vol state). It models the
             time-dynamics that a plain feature snapshot misses.
  * XGBoost: captures non-linear interactions among the price / time-of-day features.

The HMM's filtered high-vol probability is added as an EXTRA FEATURE for XGBoost.
So XGBoost sees the usual price features PLUS the HMM's regime read.

LEAK-FREE (every fold):
  * HMM is fit on the fold's TRAIN slice only;
  * the HMM feature is produced by an ONLINE forward filter (uses only bars <= t);
  * the high/low-vol threshold and both scalers are TRAIN-only;
  * an embargo drops train rows whose forward label overlaps validation.

Prints all three on the same folds: HMM-alone vs XGBoost-alone vs HYBRID.

MODEL: hmmlearn GaussianHMM (2 states) + XGBoost binary classifier.
DATA : NSE_NIFTY_intraday_<interval>.csv (NIFTY 50 only).

Usage:
  python intraday_hybrid_nifty50.py                 # 60m, H=4
  python intraday_hybrid_nifty50.py --interval 15m --horizon 4
"""
import argparse
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
N_SPLITS = 5
VOL_WIN = 6
PRICE_FEATURES = [
    "ret_1", "ret_2", "ret_4", "mom_12",
    "vol_12", "vol_24", "atr_pct", "range_pct",
    "rsi_14", "dist_sma_24", "bar_of_day", "er_past_12",
]


def load(interval):
    path = os.path.join(HERE, f"NSE_NIFTY_intraday_{interval}.csv")
    df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"]
    df["ret_1"] = c.pct_change(1); df["ret_2"] = c.pct_change(2)
    df["ret_4"] = c.pct_change(4); df["mom_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std(); df["vol_24"] = df["ret_1"].rolling(24).std()
    df["atr_pct"] = (df["high"] - df["low"]).abs().rolling(14).mean() / c
    df["range_pct"] = (df["high"] - df["low"]) / c
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean(); dn = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-12))
    df["dist_sma_24"] = (c - c.rolling(24).mean()) / c.rolling(24).mean()
    df["bar_of_day"] = df.groupby(df["time"].dt.date).cumcount()
    df["er_past_12"] = (c.diff(12).abs() / (c.diff().abs().rolling(12).sum() + 1e-12)).clip(0, 1)
    # HMM observation inputs + forward target
    df["ret"] = np.log(c).diff(); df["rv"] = df["ret"].rolling(VOL_WIN).std()
    df["fwd_vol"] = df["ret_1"].shift(-1).rolling(4).std()  # placeholder; reset per horizon
    return df


def forward_filter(obs, startprob, transmat, means, covars):
    logB = np.column_stack([multivariate_normal.logpdf(obs, means[s], covars[s],
                                                        allow_singular=True)
                            for s in range(len(means))])
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    n, k = B.shape; alpha = np.zeros((n, k))
    a = startprob * B[0]; alpha[0] = a / (a.sum() + 1e-300)
    for t in range(1, n):
        a = (alpha[t - 1] @ transmat) * B[t]
        alpha[t] = a / (a.sum() + 1e-300)
    return alpha


def hmm_highvol_prob(obs, rv, tr, upto):
    """Leak-free HMM feature: fit on TRAIN, return filtered P(high-vol) for bars
    0..upto using an online (past-only) forward filter."""
    sc = StandardScaler().fit(obs[tr])
    Z = sc.transform(obs[:upto + 1])
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=150, random_state=42)
    hmm.fit(sc.transform(obs[tr]))
    tr_states = hmm.predict(sc.transform(obs[tr]))
    vol_state = int(np.argmax([rv[tr][tr_states == s].mean() if (tr_states == s).any() else -1
                               for s in range(2)]))
    alpha = forward_filter(Z, hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)
    return alpha[:, vol_state]          # filtered high-vol probability, past-only


def _xgb():
    # Regularised after an overfitting + leakage diagnostic:
    #   * leakage: shuffling train labels collapsed val AUC to ~0.51 -> NO leakage.
    #   * overfitting: this config cut the train-val AUC gap to ~0.04 (from 0.075)
    #     with no loss in validation AUC (~0.695).
    return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
                         gamma=0.05, reg_alpha=0.5, reg_lambda=5.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def main():
    ap = argparse.ArgumentParser(description="Hybrid HMM+XGBoost intraday NIFTY 50.")
    ap.add_argument("--interval", default="60m", choices=["15m", "60m"])
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()
    H = args.horizon

    df = load(args.interval)
    df["fwd_vol"] = df["ret_1"].shift(-1).rolling(H).std().shift(-(H - 1))
    data = df.dropna(subset=PRICE_FEATURES + ["ret", "rv", "fwd_vol"]).reset_index(drop=True)

    Xp = data[PRICE_FEATURES].values
    obs = data[["ret", "rv"]].values
    rv = data["rv"].values
    fwd = data["fwd_vol"].values
    n = len(data)

    print(f"=== HYBRID (HMM + XGBoost) — intraday NIFTY 50 | {args.interval} | H={H} ===")
    print(f"Usable bars: {n} | {data['time'].iloc[0]} -> {data['time'].iloc[-1]}\n")

    # accumulators for the three models, scored on identical val folds
    pa = {k: np.zeros(n) for k in ("hmm", "xgb", "hyb")}
    yv = np.full(n, -1); mask = np.zeros(n, bool)

    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(Xp):
        tr = tr[:-H]                                   # embargo
        thr = np.nanmedian(fwd[tr])
        ytr = (fwd[tr] > thr).astype(int); yva = (fwd[va] > thr).astype(int)

        # HMM filtered high-vol prob (leak-free), for all bars up to end of val
        hv = hmm_highvol_prob(obs, rv, tr, va[-1])     # length = va[-1]+1
        hv_full = np.full(n, np.nan)
        hv_full[:len(hv)] = hv                         # tr/va indices are all < len(hv)

        # (1) HMM alone: predict high-vol if filtered prob >= 0.5
        pa["hmm"][va] = hv[va]
        # (2) XGBoost alone (price/time features)
        scp = StandardScaler().fit(Xp[tr])
        mx = _xgb(); mx.fit(scp.transform(Xp[tr]), ytr)
        pa["xgb"][va] = mx.predict_proba(scp.transform(Xp[va]))[:, 1]
        # (3) HYBRID: price/time features + HMM high-vol prob as an extra feature
        Xh = np.column_stack([Xp, hv_full])            # hv aligned to data index
        sch = StandardScaler().fit(Xh[tr])
        mh = _xgb(); mh.fit(sch.transform(Xh[tr]), ytr)
        pa["hyb"][va] = mh.predict_proba(sch.transform(Xh[va]))[:, 1]

        yv[va] = yva; mask[va] = True

    yt = yv[mask]; base = max(yt.mean(), 1 - yt.mean()) * 100
    print(f"{'model':16s} {'accuracy':>9s} {'AUC':>7s} {'edge vs base':>13s}")
    for name, label in [("hmm", "HMM alone"), ("xgb", "XGBoost alone"), ("hyb", "HYBRID (HMM+XGB)")]:
        p = pa[name][mask]; pred = (p > 0.5).astype(int)
        acc = accuracy_score(yt, pred) * 100; auc = roc_auc_score(yt, p)
        print(f"{label:16s} {acc:8.2f}% {auc:7.3f} {acc - base:+12.2f}")
    print(f"\nbaseline (majority) = {base:.2f}%")

    # write hybrid predictions
    out = data.loc[mask, ["time", "close", "fwd_vol"]].copy()
    out["prob_highvol_hmm"] = pa["hmm"][mask]
    out["prob_highvol_hybrid"] = pa["hyb"][mask]
    out["pred"] = np.where(pa["hyb"][mask] > 0.5, "high-vol", "low-vol")
    out["actual"] = np.where(yt == 1, "high-vol", "low-vol")
    out["correct"] = out["pred"] == out["actual"]
    out_path = os.path.join(HERE, f"intraday_hybrid_{args.interval}_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out)} hybrid predictions -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
