"""
forecast_horizons.py
====================
Direct multi-horizon regime FORECASTER with an accuracy gate that drives a
deployment percentage.

Idea (option 2 from the design discussion):
  Instead of recursively chaining a 5-day model out to 21 days (which feeds the
  model its own guesses and compounds error), we train FOUR INDEPENDENT models,
  one per horizon h in {5, 10, 15, 21}. Each predicts the regime `h` trading
  days ahead directly from TODAY's real features. Each is validated by the same
  honest walk-forward (out-of-fold) backtest as xgb_regime.py.

  The backtested skill of each horizon becomes a TRUST GATE:
      GREEN  -> trust the call, full conviction
      AMBER  -> reduced conviction
      RED    -> no edge over the naive baseline; stand aside
  The trust + forecast then map to a DEPLOY %.

  The dashboard shows the forecast ladder out to whatever horizon is still
  trusted (e.g. 5d & 10d GREEN, 15d AMBER, 21d RED -> only act on 5/10d).

This script does NOT modify the existing served files; it writes a new
forecast_<SYMBOL>.json so the current frontend keeps working.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "regime_features.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from regime_features import (  # noqa: E402
    XGB_FEATURES, REGIME_TO_INT, LABELS, INT_TO_REGIME,
    add_structural_features, assign_regime,
)
# Reuse the EXACT training/eval used by the in-sample dashboard model so the
# forecaster and the rest of the system can never drift apart.
from xgb_regime import train_and_evaluate, load_data  # noqa: E402

# -----------------------------------------------------------------------------
# TUNABLE KNOBS
# -----------------------------------------------------------------------------
HORIZONS = [5, 10, 15, 21]

# Forward-return label thresholds (causal expanding quantiles of the *future*
# h-day return). Convention matches kmeans_regime.py: bear = bottom 20%,
# bull = top 40%, sideways = the middle.
BEAR_Q = 0.20
BULL_Q = 0.60
MIN_HISTORY = 252   # ~1 trading year before we trust the quantile cutoffs

# Trust gate: skill = OOF accuracy - majority-class baseline accuracy.
# A 3-class problem with class imbalance has a baseline well above 1/3, so we
# grade on edge OVER that baseline, not raw accuracy.
GREEN_SKILL = 0.08   # >= +8 pts over baseline
AMBER_SKILL = 0.03   # >= +3 pts over baseline

# Deploy mapping.
NEUTRAL_DEPLOY = 65.0
BASE_DEPLOY = {"bull": 100.0, "sideways": 65.0, "bear": 30.0}
TRUST_FACTOR = {"GREEN": 1.0, "AMBER": 0.6, "RED": 0.0}
CONF_LO, CONF_HI = 0.50, 0.85   # confidence below LO -> 0 conviction, above HI -> full


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="features_<SYMBOL>.json from the Rust core")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--json-out", default=None, help="defaults to frontend/.../forecast_<SYMBOL>.json")
    return p.parse_args()


def trust_tier(skill: float) -> str:
    if skill >= GREEN_SKILL:
        return "GREEN"
    if skill >= AMBER_SKILL:
        return "AMBER"
    return "RED"


def conviction(trust: str, confidence: float) -> float:
    cf = (confidence - CONF_LO) / (CONF_HI - CONF_LO)
    cf = float(np.clip(cf, 0.0, 1.0))
    return TRUST_FACTOR[trust] * cf


def deploy_pct(regime: str, trust: str, confidence: float) -> float:
    conv = conviction(trust, confidence)
    raw = NEUTRAL_DEPLOY + (BASE_DEPLOY[regime] - NEUTRAL_DEPLOY) * conv
    return round(raw / 5.0) * 5.0   # round to nearest 5%


def build_labeled(df: pd.DataFrame, horizon: int):
    """Attach a TRUE forward-return target: bucket the realised return over the
    NEXT `horizon` trading days into bear/sideways/bull.

    The bucket cutoffs are causal expanding quantiles, additionally lagged by
    `horizon` so the threshold at row t only uses forward-return windows that
    have already CLOSED by t (no look-ahead leaks into the cutoffs). The label
    value itself is the future return — that's the thing we are forecasting.
    """
    d = df.copy()
    fwd = d["close"].shift(-horizon) / d["close"] - 1.0   # return over (t, t+h]

    bear_th = fwd.expanding(min_periods=MIN_HISTORY).quantile(BEAR_Q).shift(horizon)
    bull_th = fwd.expanding(min_periods=MIN_HISTORY).quantile(BULL_Q).shift(horizon)

    def label(i, r):
        bt, bl = bear_th.iloc[i], bull_th.iloc[i]
        if pd.isna(r) or pd.isna(bt) or pd.isna(bl):
            return None
        if r <= bt:
            return "bear"
        if r >= bl:
            return "bull"
        return "sideways"

    d["target"] = [label(i, r) for i, r in enumerate(fwd)]
    d = d.dropna(subset=["target"] + XGB_FEATURES).reset_index(drop=True)
    d["target_int"] = d["target"].map(REGIME_TO_INT)
    return d


def evaluate_horizon(df_feat: pd.DataFrame, horizon: int):
    """Walk-forward OOF for one horizon. Returns a summary dict + the final
    model/scaler so we can score the latest bar."""
    d = build_labeled(df_feat, horizon)
    if len(d) < 300:
        return None

    final_model, final_scaler, oof_preds, _ = train_and_evaluate(d)

    mask = oof_preds != -1
    y = d["target_int"].values
    oof_acc = float((oof_preds[mask] == y[mask]).mean())

    # Majority-class baseline on the same evaluated rows.
    vals, counts = np.unique(y[mask], return_counts=True)
    baseline = float(counts.max() / counts.sum())
    skill = oof_acc - baseline
    trust = trust_tier(skill)

    return {
        "horizon": horizon,
        "oof_accuracy": round(oof_acc, 4),
        "baseline": round(baseline, 4),
        "skill": round(skill, 4),
        "trust": trust,
        "n_eval": int(mask.sum()),
        "_model": final_model,
        "_scaler": final_scaler,
    }


def main():
    args = parse_args()
    json_out = args.json_out or f"frontend/frontend/public/data/forecast_{args.symbol}.json"
    print(f"--- Multi-horizon forecaster: {args.symbol} ---")

    # 1. Features + structural derivations (forward targets are built per-horizon
    #    from the close price inside build_labeled, so no external label file).
    df = load_data(args.input)
    df = add_structural_features(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Latest fully-featured bar = the point we forecast FROM.
    df_feat = df.dropna(subset=XGB_FEATURES).reset_index(drop=True)
    last = df_feat.iloc[-1]
    x_last = None  # filled per-horizon (scaler differs)

    # 4. Per-horizon train + evaluate + forecast.
    ladder = []
    for h in HORIZONS:
        res = evaluate_horizon(df, h)
        if res is None:
            print(f"  h={h:>2}: insufficient data, skipped")
            continue

        x_last = res["_scaler"].transform(last[XGB_FEATURES].values.reshape(1, -1))
        probs = res["_model"].predict_proba(x_last)[0]
        regime = assign_regime(probs)
        confidence = float(np.max(probs))
        dep = deploy_pct(regime, res["trust"], confidence)

        entry = {
            "horizon": h,
            "forecast_regime": regime,
            "confidence": round(confidence, 4),
            "prob_bear": round(float(probs[0]), 4),
            "prob_sideways": round(float(probs[1]), 4),
            "prob_bull": round(float(probs[2]), 4),
            "oof_accuracy": res["oof_accuracy"],
            "baseline": res["baseline"],
            "skill": res["skill"],
            "trust": res["trust"],
            "deploy_pct": dep,
            "n_eval": res["n_eval"],
        }
        ladder.append(entry)
        print(f"  h={h:>2}: {regime:<8} conf={confidence:.2f} | "
              f"OOF acc={res['oof_accuracy']:.3f} vs base {res['baseline']:.3f} "
              f"(skill {res['skill']:+.3f}) -> {res['trust']:<5} deploy {dep:.0f}%")

    # 5. Headline deploy = trust-weighted blend across non-RED horizons.
    weighted = [(TRUST_FACTOR[e["trust"]], e["deploy_pct"]) for e in ladder
                if e["trust"] != "RED"]
    if weighted:
        wsum = sum(w for w, _ in weighted)
        headline = round(sum(w * d for w, d in weighted) / wsum / 5.0) * 5.0
    else:
        headline = NEUTRAL_DEPLOY

    # Furthest horizon we still trust at all (AMBER or GREEN).
    trusted = [e["horizon"] for e in ladder if e["trust"] != "RED"]
    max_trusted = max(trusted) if trusted else 0

    out = {
        "symbol": args.symbol,
        "as_of": str(last["date"].date()),
        "close": float(last["close"]),
        "headline_deploy_pct": headline,
        "max_trusted_horizon": max_trusted,
        "horizons": ladder,
    }

    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] {args.symbol}: headline deploy {headline:.0f}% "
          f"(trusted out to {max_trusted}d) -> {json_out}")


if __name__ == "__main__":
    main()
