"""
intraday_hmm_nifty50.py
=======================
Gaussian HMM regime model for INTRADAY NIFTY 50 (self-contained).

A Hidden Markov Model treats the market as moving between a few hidden "states"
(regimes), each with its own Gaussian behaviour for (return, volatility), plus a
transition matrix giving the probability of moving between states. This is a
classic regime-detection technique and naturally captures regime PERSISTENCE.

This file does BOTH things, clearly separated, because an HMM is mainly an
IDENTIFIER and only secondarily a predictor:

  (A) IDENTIFY  : fit a GaussianHMM on intraday (return, realized-vol) and decode
                  the hidden state at each bar -> a calm vs volatile regime label.
                  Reports each state's character + transition matrix + persistence.

  (B) PREDICT   : honest forward test. Fit the HMM on the PAST (train split), then
                  walk the TEST bars with an ONLINE forward filter (past-only) and
                  use the transition matrix to predict the NEXT bar's regime.
                  Reports prediction accuracy vs baseline -> directly comparable to
                  the XGBoost volatility-regime model.

MODEL: hmmlearn GaussianHMM (2 hidden states, full covariance).
DATA : NSE_NIFTY_intraday_<interval>.csv (NIFTY 50 only).

NO LOOK-AHEAD (prediction): HMM parameters learned on the train split only; the
test-time filter uses only bars up to t; the next-bar target is forward-only.

Usage:
  python intraday_hmm_nifty50.py                 # 60m bars, 2 states
  python intraday_hmm_nifty50.py --interval 15m --states 3
"""
import argparse
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
VOL_WIN = 6          # bars for realized-vol observation
TRAIN_FRAC = 0.70    # chronological split: fit on past, test on future


def load(interval):
    path = os.path.join(HERE, f"NSE_NIFTY_intraday_{interval}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run an intraday fetch first.")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = df["ret"].rolling(VOL_WIN).std()
    return df.dropna(subset=["ret", "rv"]).reset_index(drop=True)


def emission_logprob(obs, means, covars):
    """log P(obs | state) for each state -> (n_obs, n_states)."""
    n, k = len(obs), len(means)
    out = np.zeros((n, k))
    for s in range(k):
        out[:, s] = multivariate_normal.logpdf(obs, mean=means[s], cov=covars[s],
                                                allow_singular=True)
    return out


def forward_filter(obs, startprob, transmat, means, covars):
    """Online forward filter: filtered state posterior at each t using bars <= t
    only. Returns alpha (n, k) normalised per row."""
    logB = emission_logprob(obs, means, covars)
    B = np.exp(logB - logB.max(axis=1, keepdims=True))   # stabilised emissions
    n, k = B.shape
    alpha = np.zeros((n, k))
    a = startprob * B[0]; alpha[0] = a / (a.sum() + 1e-300)
    for t in range(1, n):
        a = (alpha[t - 1] @ transmat) * B[t]
        alpha[t] = a / (a.sum() + 1e-300)
    return alpha


def main():
    ap = argparse.ArgumentParser(description="Intraday Gaussian HMM for NIFTY 50.")
    ap.add_argument("--interval", default="60m", choices=["15m", "60m"])
    ap.add_argument("--states", type=int, default=2, help="hidden states (2 = calm/volatile)")
    ap.add_argument("--horizon", type=int, default=4,
                    help="forward bars for the prediction target (strictly future; matches XGBoost)")
    args = ap.parse_args()

    df = load(args.interval)
    obs_all = df[["ret", "rv"]].values
    n = len(obs_all)
    cut = int(n * TRAIN_FRAC)

    # scale on TRAIN only (no look-ahead)
    scaler = StandardScaler().fit(obs_all[:cut])
    Z = scaler.transform(obs_all)
    Ztr, Zte = Z[:cut], Z[cut:]

    print(f"=== Intraday Gaussian HMM — NIFTY 50 | {args.interval} | {args.states} states ===")
    print(f"Bars: {n} (train {cut} / test {n - cut}) | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    hmm = GaussianHMM(n_components=args.states, covariance_type="full",
                      n_iter=200, random_state=42)
    hmm.fit(Ztr)

    # --- (A) IDENTIFY: characterise states from TRAIN decode ------------------
    tr_states = hmm.predict(Ztr)
    rv_tr = df["rv"].values[:cut]
    state_vol = {s: rv_tr[tr_states == s].mean() for s in range(args.states)}
    vol_state = max(state_vol, key=state_vol.get)          # highest-vol hidden state
    ret_tr = df["ret"].values[:cut]
    print("\n(A) IDENTIFY — hidden-state character (train):")
    for s in range(args.states):
        m = tr_states == s
        dur = 1.0 / (1.0 - hmm.transmat_[s, s] + 1e-9)
        tag = "VOLATILE" if s == vol_state else "calm"
        print(f"  state {s} [{tag:8s}] share={m.mean():.2f}  "
              f"mean_ret={ret_tr[m].mean():+.5f}  mean_rv={rv_tr[m].mean():.5f}  "
              f"avg_duration={dur:.1f} bars")

    # --- (B) PREDICT: honest forward filter on TEST ---------------------------
    H = args.horizon
    alpha = forward_filter(Zte, hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)
    # propagate each filtered state H steps ahead; predict "volatile next H bars"
    # = average probability of the high-vol state over steps t+1..t+H >= 0.5.
    T = hmm.transmat_
    vol_prob = np.zeros(len(alpha))
    d = alpha.copy()
    for _ in range(H):
        d = d @ T
        vol_prob += d[:, vol_state]
    vol_prob /= H
    pred_volatile = (vol_prob >= 0.5).astype(int)

    # STRICTLY-FORWARD target (no overlap with the rv feature): realized vol over
    # the next H bars (ret[t+1..t+H]) vs the TRAIN median. Same shape as XGBoost.
    fwd_vol_all = df["ret"].shift(-1).rolling(H).std().shift(-(H - 1)).values
    thr = np.nanmedian(fwd_vol_all[:cut])
    fwd_te = fwd_vol_all[cut:]
    ok = ~np.isnan(fwd_te)
    pred = pred_volatile[ok]
    truth = (fwd_te[ok] > thr).astype(int)
    acc = (pred == truth).mean() * 100
    base = max(truth.mean(), 1 - truth.mean()) * 100
    print(f"\n(B) PREDICT volatility regime over next {H} bars (honest, strictly-forward):")
    print(f"  accuracy = {acc:.2f}%   baseline = {base:.2f}%   (edge = {acc - base:+.2f} pts)")
    print(f"  (compare: XGBoost volatility-regime model ~61% at H={H})")

    # --- output -------------------------------------------------------------
    test_df = df.iloc[cut:].reset_index(drop=True).iloc[ok].copy()
    test_df["hmm_state"] = alpha[ok].argmax(axis=1)
    test_df["identified_regime"] = np.where(test_df["hmm_state"] == vol_state, "volatile", "calm")
    test_df["pred_next"] = np.where(pred == 1, "volatile", "calm")
    test_df["actual_next"] = np.where(truth == 1, "volatile", "calm")
    test_df["correct"] = test_df["pred_next"] == test_df["actual_next"]
    out = test_df[["time", "close", "ret", "rv", "hmm_state", "identified_regime",
                   "pred_next", "actual_next", "correct"]]
    out_path = os.path.join(HERE, f"intraday_hmm_{args.interval}_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[OK] wrote {len(out)} rows -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
