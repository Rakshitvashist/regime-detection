"""
export_our_results.py
=====================
Bundle the three NEW model results (intraday / weekly / monthly) into a single
JSON the dashboard can plot:  frontend/frontend/public/data/our_models.json

Reads the *_eval.csv files written by the model scripts and produces, per track:
  * headline hit ratio + baseline + edge,
  * a confidence curve (hit ratio vs confidence threshold, with coverage),
  * a down-sampled prediction timeline (pred vs actual, prob, correct).

Run after the model scripts. No look-ahead concern — this only summarises
already-computed out-of-fold predictions.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frontend", "frontend", "public", "data", "our_models.json")
MAX_POINTS = 400
THRESHOLDS = np.round(np.arange(0.50, 0.91, 0.05), 2)


def conf_curve(correct, conf):
    out = []
    for th in THRESHOLDS:
        m = conf >= th
        if m.sum() >= 20:
            out.append({"threshold": float(th),
                        "hit_ratio": round(float(correct[m].mean()) * 100, 2),
                        "coverage": round(float(m.mean()) * 100, 1)})
    return out


def build(key, label, model, horizon, path, xcol, predcol, actualcol,
          confcol, raw_prob, baseline):
    df = pd.read_csv(os.path.join(HERE, path))
    correct = df["correct"].astype(bool).to_numpy()
    raw = df[confcol].to_numpy(dtype=float)
    conf = (np.abs(raw - 0.5) + 0.5) if raw_prob else raw     # 0.5..1.0 confidence

    d = df.iloc[:: max(1, len(df) // MAX_POINTS)] if len(df) > MAX_POINTS else df
    timeline = [{"x": str(r[xcol]), "close": float(r["close"]),
                 "prob": round(float(r[confcol]), 4),
                 "pred": str(r[predcol]), "actual": str(r[actualcol]),
                 "correct": bool(r["correct"])}
                for _, r in d.iterrows()]

    hit = float(correct.mean()) * 100
    return {"key": key, "label": label, "model": model, "horizon": horizon,
            "overall_hit_ratio": round(hit, 2), "baseline": round(baseline, 2),
            "edge": round(hit - baseline, 2), "n": int(len(df)),
            "confidence_curve": conf_curve(correct, conf),
            "timeline": timeline}


def main():
    tracks = [
        build("intraday", "Intraday — Volatility Regime", "HMM + XGBoost hybrid",
              "next ~4 hours (60m bars)", "intraday_hybrid_60m_eval.csv",
              "time", "pred", "actual", "prob_highvol_hybrid", True, 53.63),
        build("weekly", "Weekly Expiry — Direction", "HMM + XGBoost hybrid",
              "5 trading days", "weekly_hybrid_nifty50_eval.csv",
              "date", "pred_regime", "actual_regime", "confidence", False, 47.53),
        build("monthly", "Monthly Expiry — Volatility Regime", "Volatility-persistence rule",
              "21 trading days", "monthly_persistence_nifty50_eval.csv",
              "date", "pred", "actual", "confidence", False, 56.39),
    ]
    payload = {"title": "New NIFTY 50 Regime Models", "tracks": tracks}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] wrote {OUT}")
    for t in tracks:
        print(f"  {t['label']:42s} hit={t['overall_hit_ratio']}%  edge={t['edge']:+}  "
              f"n={t['n']}  curve={len(t['confidence_curve'])}pts  tl={len(t['timeline'])}")


if __name__ == "__main__":
    main()
