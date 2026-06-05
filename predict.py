"""
predict.py
==========
Standalone IN-SAMPLE / live inference utility: load a trained model and score a
features file with it.

NOTE: the predictions this produces for already-labelled rows are IN-SAMPLE
(the deployed model was fit on those rows). The dashboard's walk-forward
backtest is produced by xgb_regime.py (out-of-fold). Use this script for
ad-hoc inspection or to score the newest bars — not as the source of the
"backtest" hit-rate. To avoid clobbering the honest file it writes to
output/regime_insample_<SYMBOL>.json by default.

Two bugs were fixed versus the previous version:
  * ground-truth labels are now read from the per-symbol clustered file
    (was the stale, symbol-less output/regime_clustered.json);
  * predictions are built directly from the de-NaN'd frame, removing the
    fragile dual-index loop that could attach probabilities to the wrong rows
    after the ground-truth merge re-indexed the frame.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "regime_features.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from regime_features import (  # noqa: E402
    XGB_FEATURES, add_structural_features, assign_regime,
    calculate_checklist, dashboard_metrics,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/features.json")
    parser.add_argument("--symbol", default="NIFTY_500")
    parser.add_argument("--output", default=None,
                        help="Defaults to output/regime_insample_<SYMBOL>.json")
    parser.add_argument("--model", default="output/xgb_model.pkl")
    parser.add_argument("--labels", default=None,
                        help="Per-symbol clustered labels (defaults to output/regime_clustered_<SYMBOL>.json)")
    parser.add_argument("--horizon", type=int, default=21)
    return parser.parse_args()


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def main():
    args = get_args()
    labels_path = args.labels or f"output/regime_clustered_{args.symbol}.json"
    out_path = args.output or f"output/regime_insample_{args.symbol}.json"
    print(f"--- Regime Inference (in-sample): {args.symbol} ---")

    # 1. Features
    try:
        df = load_data(args.input)
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    df = add_structural_features(df)
    df["date"] = pd.to_datetime(df["date"])

    # 2. Model + scaler
    try:
        with open(args.model, "rb") as f:
            model, scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Predict on rows with complete features (single index, no realignment)
    df_pred = df.dropna(subset=XGB_FEATURES).reset_index(drop=True)
    if df_pred.empty:
        print("No rows with complete features for prediction.")
        return
    probs = model.predict_proba(scaler.transform(df_pred[XGB_FEATURES].values))

    # 4. Ground truth = this symbol's regime, `horizon` days ahead
    truth_by_date = {}
    try:
        with open(labels_path) as f:
            df_truth = pd.DataFrame(json.load(f))
        df_truth["date"] = pd.to_datetime(df_truth["date"])
        df_truth = df_truth.sort_values("date").reset_index(drop=True)
        df_truth["true_regime"] = df_truth["regime"].shift(-args.horizon)
        truth_by_date = dict(zip(df_truth["date"], df_truth["true_regime"]))
    except Exception as e:
        print(f"Warning: could not load ground truth ({labels_path}): {e}")

    res = []
    correct = total = 0
    for i, row in df_pred.iterrows():
        p = probs[i]
        regime = assign_regime(p)
        true_regime = truth_by_date.get(row["date"])
        true_regime = str(true_regime) if pd.notna(true_regime) else None

        rec = {
            "date": str(row["date"].date()),
            "symbol": str(args.symbol),
            "close": float(row["close"]),
            "regime": regime,
            "regime_actual": true_regime,
            "confidence": float(np.max(p)),
            "prob_bear": float(p[0]),
            "prob_sideways": float(p[1]),
            "prob_bull": float(p[2]),
            "checklist": calculate_checklist(row, p),
            "accuracy": None,
        }
        rec.update(dashboard_metrics(row))

        if true_regime is not None:
            total += 1
            if regime == true_regime:
                correct += 1
            rec["accuracy"] = correct / total
        res.append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    acc = (correct / total) if total else 0.0
    print(f"[OK] {args.symbol}: {len(res)} rows, in-sample hit-rate {acc:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
