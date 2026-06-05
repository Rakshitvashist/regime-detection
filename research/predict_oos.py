"""
predict_oos.py
==============
Cross-instrument out-of-sample generalization test.

Takes a model trained on instrument A and applies it to instrument B's
features. This is TRUE out-of-sample — the model has never seen instrument B.

Fixed versus the previous version:
  * ground truth is read from the TARGET symbol's own per-symbol clustered
    labels (was the stale, symbol-less output/regime_clustered.json, so OOS
    "accuracy" was measured against the wrong instrument);
  * shared feature pipeline + unified confidence threshold;
  * ground truth is joined by a date map rather than a merge, so the
    probability array can never desync from the rows.

Usage:
  python research/predict_oos.py \
    --model output/xgb_model_NIFTY.pkl \
    --input output/features_NIFTY_500.json \
    --symbol NIFTY_500 --trained-on NIFTY \
    --json-out frontend/frontend/public/data/regime_NIFTY_on_NIFTY500.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "regime_features.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from regime_features import (  # noqa: E402
    XGB_FEATURES, LABELS, add_structural_features, assign_regime, dashboard_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained .pkl model")
    p.add_argument("--input", required=True, help="Features JSON for target instrument")
    p.add_argument("--symbol", required=True, help="Target instrument name (e.g. NIFTY_500)")
    p.add_argument("--trained-on", required=True, help="Source instrument name (e.g. NIFTY)")
    p.add_argument("--json-out", required=True, help="Output JSON path")
    p.add_argument("--labels", default=None,
                   help="Target symbol's clustered labels (defaults to output/regime_clustered_<SYMBOL>.json)")
    p.add_argument("--horizon", type=int, default=21)
    return p.parse_args()


def main():
    args = parse_args()
    labels_path = args.labels or f"output/regime_clustered_{args.symbol}.json"

    print(f"\n{'='*60}")
    print(f"  Cross-Instrument OOS Generalization Test")
    print(f"  Model trained on : {args.trained_on}")
    print(f"  Applied to       : {args.symbol}")
    print(f"{'='*60}\n")

    with open(args.model, "rb") as f:
        model, scaler = pickle.load(f)
    print(f"[OK] Loaded model from {args.model}")

    with open(args.input) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")

    df = add_structural_features(df)
    df["date"] = pd.to_datetime(df["date"])

    df_pred = df.dropna(subset=XGB_FEATURES).reset_index(drop=True)
    print(f"[OK] {len(df_pred)} rows with complete features")

    # Scale with the SOURCE instrument's scaler (same scaler the model trained with)
    probs = model.predict_proba(scaler.transform(df_pred[XGB_FEATURES].values))

    # Ground truth = TARGET symbol's own regime, `horizon` days ahead
    truth_by_date = {}
    try:
        with open(labels_path) as f:
            truth_df = pd.DataFrame(json.load(f))
        truth_df["date"] = pd.to_datetime(truth_df["date"])
        truth_df = truth_df.sort_values("date").reset_index(drop=True)
        truth_df["true_regime"] = truth_df["regime"].shift(-args.horizon)
        truth_by_date = dict(zip(truth_df["date"], truth_df["true_regime"]))
    except Exception as e:
        print(f"[WARN] No ground truth ({labels_path}): {e}")

    results = []
    correct = total = 0
    for i, row in df_pred.iterrows():
        p = probs[i]
        pred_label = assign_regime(p)
        true_regime = truth_by_date.get(row["date"])
        is_valid = pd.notna(true_regime) and true_regime in LABELS
        if is_valid:
            total += 1
            if pred_label == true_regime:
                correct += 1
        rolling_acc = (correct / total) if total else None

        rec = {
            "date": str(row["date"].date()),
            "close": float(row["close"]),
            "symbol": args.symbol,
            "trained_on": args.trained_on,
            "is_oos": True,
            "regime": pred_label,
            "regime_actual": str(true_regime) if is_valid else None,
            "confidence": float(np.max(p)),
            "prob_bear": float(p[0]),
            "prob_sideways": float(p[1]),
            "prob_bull": float(p[2]),
            "accuracy": rolling_acc,
        }
        rec.update(dashboard_metrics(row))
        results.append(rec)

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if r["regime_actual"] is not None]
    if valid:
        y_true = [r["regime_actual"] for r in valid]
        y_pred = [r["regime"] for r in valid]
        oos_acc = accuracy_score(y_true, y_pred)
        print(f"\n  OOS ACCURACY ({args.trained_on} -> {args.symbol}): {oos_acc:.4f} ({oos_acc*100:.1f}%)")
        print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    print(f"[OK] Saved {len(results)} rows -> {args.json_out}")


if __name__ == "__main__":
    main()
