"""
forecast_horizons.py
====================
Two honest, evidence-backed signals per instrument:

  DETECTION  -- nowcast the CURRENT regime (bull / sideways / bear) from current
                features. The present state is observable, so walk-forward
                accuracy is high (~90%). This drives the *direction* of the
                deployment.

  RISK LADDER -- for h in {5,10,15,21}, forecast whether the NEXT h days will be
                CALM or STORMY (forward realised vol above/below its trailing
                median). Volatility persists, so this has genuine skill
                (AUC ~0.6-0.68). Graded by AUC, not raw accuracy (the calm class
                dominates, so accuracy-vs-majority is misleading). This drives a
                *haircut* on the deployment.

Why not forecast direction? Measured corr(features, forward return) ~ 0.03 --
index direction at 5-21d is unforecastable, so it is deliberately excluded.

DEPLOY % = base(detected regime, scaled by detection confidence) reduced by the
trusted storm probability. Calm bull -> high; stormy anything -> defensive.

Writes forecast_<SYMBOL>.json; does not touch the existing served files.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from xgboost import XGBClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "regime_features.py")) else os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regime_features import XGB_FEATURES, REGIME_TO_INT, LABELS, add_structural_features  # noqa: E402
from xgb_regime import model_config, load_data  # noqa: E402

# ----------------------------- TUNABLE KNOBS --------------------------------
HORIZONS = [5, 10, 15, 21]
N_SPLITS = 5
MIN_VOL_HISTORY = 252        # bars before trailing-median vol threshold is trusted

# Risk trust gate, graded on AUC (ranking skill), not accuracy.
AUC_GREEN = 0.62
AUC_AMBER = 0.55

# Deploy mapping.
NEUTRAL_DEPLOY = 65.0
BASE_DEPLOY = {"bull": 100.0, "sideways": 65.0, "bear": 30.0}
DETECT_CONF_LO, DETECT_CONF_HI = 0.50, 0.90   # detection confidence -> 0..1 conviction
STORM_HAIRCUT = 0.50          # a fully-stormy outlook removes up to 50% of deploy
STORM_TH = 0.50               # P(stormy) above this = "stormy" label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="features_<SYMBOL>.json from the Rust core")
    p.add_argument("--labels", default=None, help="regime_clustered_<SYMBOL>.json (current-regime labels)")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--json-out", default=None, help="defaults to frontend/.../forecast_<SYMBOL>.json")
    return p.parse_args()


def _binary_cfg():
    cfg = model_config()
    cfg.update(objective="binary:logistic", eval_metric="logloss")
    cfg.pop("num_class", None)
    return cfg


# --------------------------- DETECTION (nowcast) ----------------------------
def run_detection(df, last_row):
    """Train a no-shift current-regime classifier; return its OOF accuracy and
    the latest-bar nowcast (regime + confidence)."""
    d = df.dropna(subset=["regime"] + XGB_FEATURES).reset_index(drop=True)
    X = d[XGB_FEATURES].values
    y = d["regime"].map(REGIME_TO_INT).values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), -1)
    for tr, va in tscv.split(X):
        sc = StandardScaler()
        m = XGBClassifier(**model_config())
        m.fit(sc.fit_transform(X[tr]), y[tr], verbose=False)
        oof[va] = m.predict(sc.transform(X[va]))
    mask = oof != -1
    acc = float((oof[mask] == y[mask]).mean())
    _, cnts = np.unique(y[mask], return_counts=True)
    baseline = float(cnts.max() / cnts.sum())

    # Final model on all labelled rows -> nowcast the latest bar.
    sc = StandardScaler()
    m = XGBClassifier(**model_config())
    m.fit(sc.fit_transform(X), y, verbose=False)
    probs = m.predict_proba(sc.transform(last_row[XGB_FEATURES].values.reshape(1, -1)))[0]
    idx = int(np.argmax(probs))
    return {
        "regime": LABELS[idx],
        "confidence": round(float(probs[idx]), 4),
        "prob_bear": round(float(probs[0]), 4),
        "prob_sideways": round(float(probs[1]), 4),
        "prob_bull": round(float(probs[2]), 4),
        "oof_accuracy": round(acc, 4),
        "baseline": round(baseline, 4),
        "n_eval": int(mask.sum()),
    }


# --------------------------- RISK (forward vol) -----------------------------
def run_risk(df, last_row, horizon):
    """Binary calm/stormy forecast for the next `horizon` days. Returns OOF AUC /
    balanced-accuracy / storm-recall and the latest-bar storm probability."""
    r1 = df["close"].pct_change()
    hist_med = df["vol_21d"].expanding(MIN_VOL_HISTORY).median()
    fwd_vol = r1.shift(-horizon).rolling(horizon).std() * np.sqrt(252)
    y_full = pd.Series(np.where(fwd_vol >= hist_med, 1, 0), index=df.index).astype("float")
    y_full[fwd_vol.isna() | hist_med.isna()] = np.nan

    d = df.assign(_y=y_full).dropna(subset=["_y"] + XGB_FEATURES).reset_index(drop=True)
    X = d[XGB_FEATURES].values
    y = d["_y"].astype(int).values
    if len(np.unique(y)) < 2 or len(d) < 300:
        return None

    cfg = _binary_cfg()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), -1)
    proba = np.full(len(y), np.nan)
    for tr, va in tscv.split(X):
        sc = StandardScaler()
        m = XGBClassifier(**cfg)
        m.fit(sc.fit_transform(X[tr]), y[tr], verbose=False)
        Xva = sc.transform(X[va])
        oof[va] = m.predict(Xva)
        proba[va] = m.predict_proba(Xva)[:, 1]
    mask = oof != -1
    yt, yp, pp = y[mask], oof[mask], proba[mask]
    auc = float(roc_auc_score(yt, pp)) if len(np.unique(yt)) > 1 else 0.5
    balacc = float(balanced_accuracy_score(yt, yp))
    storm_recall = float(recall_score(yt, yp, pos_label=1, zero_division=0))

    # Final model -> latest-bar storm probability.
    sc = StandardScaler()
    m = XGBClassifier(**cfg)
    m.fit(sc.fit_transform(X), y, verbose=False)
    storm_prob = float(m.predict_proba(sc.transform(last_row[XGB_FEATURES].values.reshape(1, -1)))[0, 1])

    trust = "GREEN" if auc >= AUC_GREEN else ("AMBER" if auc >= AUC_AMBER else "RED")
    return {
        "horizon": horizon,
        "storm_prob": round(storm_prob, 4),
        "outlook": "stormy" if storm_prob >= STORM_TH else "calm",
        "auc": round(auc, 4),
        "balanced_acc": round(balacc, 4),
        "storm_recall": round(storm_recall, 4),
        "trust": trust,
        "n_eval": int(mask.sum()),
    }


# ------------------------------- DEPLOY % -----------------------------------
def compute_deploy(detection, risk_ladder):
    regime = detection["regime"]
    cf = (detection["confidence"] - DETECT_CONF_LO) / (DETECT_CONF_HI - DETECT_CONF_LO)
    cf = float(np.clip(cf, 0.0, 1.0))
    base = NEUTRAL_DEPLOY + (BASE_DEPLOY[regime] - NEUTRAL_DEPLOY) * cf

    trusted = [r for r in risk_ladder if r["trust"] != "RED"]
    if trusted:
        headline_storm = float(np.mean([r["storm_prob"] for r in trusted]))
        deploy = base * (1.0 - STORM_HAIRCUT * headline_storm)
    else:
        headline_storm = None
        deploy = base
    return round(deploy / 5.0) * 5.0, headline_storm


def main():
    args = parse_args()
    labels_path = args.labels or f"output/regime_clustered_{args.symbol}.json"
    json_out = args.json_out or f"frontend/frontend/public/data/forecast_{args.symbol}.json"
    print(f"--- Detection + risk forecaster: {args.symbol} ---")

    df = add_structural_features(load_data(args.input))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # current-regime labels (for the detection target)
    with open(labels_path) as f:
        lab = pd.DataFrame(json.load(f))
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab[["date", "regime"]], on="date", how="left")

    df_feat = df.dropna(subset=XGB_FEATURES).reset_index(drop=True)
    last = df_feat.iloc[-1]

    detection = run_detection(df, last)
    print(f"  DETECTION: now = {detection['regime']} ({detection['confidence']*100:.0f}%) | "
          f"OOF acc {detection['oof_accuracy']*100:.1f}% vs base {detection['baseline']*100:.1f}%")

    risk_ladder = []
    for h in HORIZONS:
        r = run_risk(df, last, h)
        if r is None:
            continue
        risk_ladder.append(r)
        print(f"  RISK h={h:>2}: {r['outlook']:<6} P(storm)={r['storm_prob']:.2f} | "
              f"AUC {r['auc']:.3f} -> {r['trust']}")

    deploy, headline_storm = compute_deploy(detection, risk_ladder)
    trusted_h = [r["horizon"] for r in risk_ladder if r["trust"] != "RED"]
    max_trusted = max(trusted_h) if trusted_h else 0

    out = {
        "symbol": args.symbol,
        "as_of": str(last["date"].date()),
        "close": float(last["close"]),
        "detection": detection,
        "risk": risk_ladder,
        "headline_storm_prob": round(headline_storm, 4) if headline_storm is not None else None,
        "max_trusted_horizon": max_trusted,
        "deploy_pct": deploy,
    }
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] {args.symbol}: regime {detection['regime']} + risk trusted to {max_trusted}d "
          f"-> deploy {deploy:.0f}% -> {json_out}")


if __name__ == "__main__":
    main()
