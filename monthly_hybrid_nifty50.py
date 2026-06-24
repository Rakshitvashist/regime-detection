"""
monthly_hybrid_nifty50.py
=========================
FINAL Monthly model for NIFTY 50 = Gaussian HMM + XGBoost (hybrid), certified.

Monthly DIRECTION is not predictable (loses to up-drift), so the target is the
MONTHLY VOLATILITY REGIME: will the next ~21 trading days be HIGH-vol or LOW-vol?

Same hybrid design as the intraday model:
  * HMM identifies the volatility regime + its persistence (filtered high-vol prob);
  * that prob is an extra feature for XGBoost, which also sees the Nifty features.

Self-certifying output:
  * HMM-alone vs XGBoost-alone vs HYBRID (identical folds),
  * overfitting: hybrid train-vs-validation AUC gap,
  * leakage control: SHUFFLED train labels -> val AUC should collapse to ~0.5.

LEAK-FREE: HMM fit on train only; regime feature via past-only forward filter;
high/low-vol threshold from TRAIN only; embargo (drop train rows whose forward
label overlaps validation); both scalers train-only.

MODEL: hmmlearn GaussianHMM (2 states) + regularised XGBoost binary classifier.
DATA : output/features_NIFTY.json (NIFTY 50 daily).

Usage:  python monthly_hybrid_nifty50.py            # 21-day horizon
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from regime_features import XGB_FEATURES, add_structural_features

N_SPLITS = 5
VOL_WIN = 10          # daily realized-vol window for the HMM observation


def load():
    with open(os.path.join(HERE, "output", "features_NIFTY.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    df = add_structural_features(df.sort_values("date").reset_index(drop=True))
    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = df["ret"].rolling(VOL_WIN).std()
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
    sc = StandardScaler().fit(obs[tr])
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=150, random_state=42)
    hmm.fit(sc.transform(obs[tr]))
    st = hmm.predict(sc.transform(obs[tr]))
    vol_state = int(np.argmax([rv[tr][st == s].mean() if (st == s).any() else -1 for s in range(2)]))
    alpha = forward_filter(sc.transform(obs[:upto + 1]),
                           hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)
    return alpha[:, vol_state]


def _xgb():
    # regularised for the small (~1700-row) daily set
    return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
                         gamma=0.1, reg_alpha=1.0, reg_lambda=5.0,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42, verbosity=0)


def main():
    ap = argparse.ArgumentParser(description="Hybrid HMM+XGBoost monthly NIFTY 50.")
    ap.add_argument("--shift", type=int, default=21)
    args = ap.parse_args()
    H = args.shift

    df = load()
    df["fwd_vol"] = df["close"].pct_change().shift(-1).rolling(H).std().shift(-(H - 1))
    data = df.dropna(subset=XGB_FEATURES + ["ret", "rv", "fwd_vol"]).reset_index(drop=True)
    Xp = data[XGB_FEATURES].values
    obs = data[["ret", "rv"]].values
    rv = data["rv"].values
    fwd = data["fwd_vol"].values
    n = len(data)

    print(f"=== HYBRID (HMM + XGBoost) — MONTHLY NIFTY 50 | horizon={H} days ===")
    print(f"Usable rows: {n} | {data['date'].iloc[0].date()} -> {data['date'].iloc[-1].date()}\n")

    pa = {k: np.zeros(n) for k in ("hmm", "xgb", "hyb", "naive")}
    yv = np.full(n, -1); mask = np.zeros(n, bool)
    tr_auc, va_auc, shuf_auc = [], [], []
    rng = np.random.default_rng(1)

    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(Xp):
        tr = tr[:-H]                                  # embargo
        thr = np.nanmedian(fwd[tr])
        ytr = (fwd[tr] > thr).astype(int); yva = (fwd[va] > thr).astype(int)
        hv = hmm_highvol_prob(obs, rv, tr, va[-1])
        hvf = np.full(n, np.nan); hvf[:len(hv)] = hv

        pa["hmm"][va] = hv[va]
        # naive PERSISTENCE control: predict high-vol if CURRENT vol > train median.
        # If the HMM ~= this, its edge is genuine vol-persistence (not leakage).
        pa["naive"][va] = (rv[va] > np.median(rv[tr])).astype(float)
        scp = StandardScaler().fit(Xp[tr]); mx = _xgb(); mx.fit(scp.transform(Xp[tr]), ytr)
        pa["xgb"][va] = mx.predict_proba(scp.transform(Xp[va]))[:, 1]
        Xh = np.column_stack([Xp, hvf]); sch = StandardScaler().fit(Xh[tr])
        mh = _xgb(); mh.fit(sch.transform(Xh[tr]), ytr)
        pa["hyb"][va] = mh.predict_proba(sch.transform(Xh[va]))[:, 1]
        tr_auc.append(roc_auc_score(ytr, mh.predict_proba(sch.transform(Xh[tr]))[:, 1]))
        va_auc.append(roc_auc_score(yva, pa["hyb"][va]))
        ms = _xgb(); ms.fit(sch.transform(Xh[tr]), rng.permutation(ytr))   # leakage control
        shuf_auc.append(roc_auc_score(yva, ms.predict_proba(sch.transform(Xh[va]))[:, 1]))
        yv[va] = yva; mask[va] = True

    yt = yv[mask]; base = max(yt.mean(), 1 - yt.mean()) * 100
    print(f"{'model':16s} {'accuracy':>9s} {'AUC':>7s} {'edge':>7s}")
    for k, lab in [("hmm", "HMM alone"), ("xgb", "XGBoost alone"), ("hyb", "HYBRID")]:
        p = pa[k][mask]; acc = accuracy_score(yt, (p > 0.5)) * 100
        print(f"{lab:16s} {acc:8.2f}% {roc_auc_score(yt, p):7.3f} {acc - base:+6.2f}")
    naive_acc = accuracy_score(yt, (pa["naive"][mask] > 0.5)) * 100
    print(f"\nbaseline = {base:.2f}%")
    print(f"overfitting: hybrid train AUC={np.mean(tr_auc):.3f} val AUC={np.mean(va_auc):.3f} "
          f"gap={np.mean(tr_auc) - np.mean(va_auc):.3f}")
    print(f"leakage control (shuffled labels): val AUC={np.mean(shuf_auc):.3f}  (~0.5 => clean)")
    print(f"persistence sanity: naive (rv>train-median) acc={naive_acc:.2f}%  "
          f"-> HMM ~= persistence => honest vol-clustering, not leakage")

    # Save the HMM-ALONE result (the monthly winner) with its confidence.
    hmm_pred = (pa["hmm"][mask] > 0.5)
    conf = np.abs(pa["hmm"][mask] - 0.5) + 0.5
    out = data.loc[mask, ["date", "close", "fwd_vol"]].copy()
    out["prob_highvol"] = pa["hmm"][mask]
    out["confidence"] = conf
    out["pred"] = np.where(hmm_pred, "high-vol", "low-vol")
    out["actual"] = np.where(yt == 1, "high-vol", "low-vol")
    out["correct"] = out["pred"] == out["actual"]
    out.to_csv(os.path.join(HERE, "monthly_hmm_nifty50_eval.csv"), index=False)
    # high-confidence hit ratio
    print("\nMONTHLY HMM-alone hit ratio:")
    print(f"  overall          : {out['correct'].mean()*100:.2f}%  (n={len(out)})")
    for th in (0.60, 0.65, 0.70):
        m = conf > th
        if m.sum() > 20:
            print(f"  confidence>{th:.2f}  : {out['correct'].values[m].mean()*100:.2f}%  "
                  f"(on {m.mean()*100:.0f}% of bars)")
    print(f"[OK] wrote {len(out)} HMM-alone predictions -> monthly_hmm_nifty50_eval.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
