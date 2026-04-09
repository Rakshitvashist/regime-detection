"""
predict_oos.py
==============
Cross-instrument out-of-sample generalization test.

Takes a model trained on instrument A and applies it to instrument B's features.
This is TRUE out-of-sample performance — the model has never seen instrument B's data.

Usage:
  python research/predict_oos.py \
    --model output/xgb_model_NIFTY.pkl \
    --input output/features_NIFTY_500.json \
    --symbol NIFTY_500 \
    --trained-on NIFTY \
    --json-out frontend/frontend/public/data/regime_NIFTY_on_NIFTY500.json
"""

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from predict import StatefulRegimeFilter, calculate_checklist

# Must match xgb_regime.py exactly
XGB_FEATURES = [
    "ret_10d",
    "vol_21d", "vol_ratio",
    "skew_21d", "kurt_21d",
    "rsi_14", "rsi_21",
    "macd_h", "bb_pct", "ema_gap", "adx",
    "drawdown_21d",
    "mom_divergence",
    "vol_spike",
    "dist_from_252h",
    "tii_21",
    "vol_of_vol_21",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, help="Path to trained .pkl model")
    p.add_argument("--input",      required=True, help="Features JSON for target instrument")
    p.add_argument("--symbol",     required=True, help="Target instrument name (e.g. NIFTY_500)")
    p.add_argument("--trained-on", required=True, help="Source instrument name (e.g. NIFTY)")
    p.add_argument("--json-out",   required=True, help="Output JSON path")
    return p.parse_args()


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "vol_21d" not in df.columns:
        df["vol_21d"] = df["close"].pct_change().rolling(21).std() * np.sqrt(252)

    rolling_max_21  = df["close"].rolling(21, min_periods=1).max()
    rolling_max_252 = df["close"].rolling(252, min_periods=1).max()

    df["drawdown_21d"]   = (df["close"] - rolling_max_21) / rolling_max_21
    df["dist_from_252h"] = (df["close"] - rolling_max_252) / rolling_max_252

    if "ret_21d" not in df.columns:
        df["ret_21d"] = df["close"].pct_change(21)

    df["mom_divergence"] = df["ret_10d"] - df["ret_21d"]
    df["vol_spike"]      = df["vol_21d"] / df["vol_21d"].rolling(63, min_periods=1).mean()

    sma21 = df["close"].rolling(21, min_periods=1).mean()
    df["tii_21"] = df["close"].rolling(21).apply(
        lambda x: (x > sma21.loc[x.index[-1]]).sum() / 21.0, raw=False
    )
    df["vol_of_vol_21"] = df["vol_21d"].rolling(21, min_periods=1).std()

    fix_cols = ["drawdown_21d", "dist_from_252h", "mom_divergence",
                "vol_spike", "tii_21", "vol_of_vol_21", "vol_21d", "ret_21d"]
    df[fix_cols] = df[fix_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Cross-Instrument OOS Generalization Test")
    print(f"  Model trained on : {args.trained_on}")
    print(f"  Applied to       : {args.symbol}")
    print(f"{'='*60}\n")

    # Load model
    with open(args.model, "rb") as f:
        model, scaler = pickle.load(f)
    print(f"[OK] Loaded model from {args.model}")

    # Load target features
    with open(args.input) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")

    # Build structural features
    df = add_structural_features(df)

    # Filter rows with all features
    df_pred = df.dropna(subset=XGB_FEATURES).copy().reset_index(drop=True)
    print(f"[OK] {len(df_pred)} rows with complete features")

    # Scale with the SOURCE instrument's scaler (critical — same scaler as training)
    X = scaler.transform(df_pred[XGB_FEATURES].values)
    probs = model.predict_proba(X)

    LABEL_MAP = {0: "bear", 1: "sideways", 2: "bull"}

    # Load regime_clustered.json for ground truth accuracy calculation
    accuracy_data = None
    try:
        with open("output/regime_clustered.json") as f:
            truth_df = pd.DataFrame(json.load(f))
        truth_df["date"] = pd.to_datetime(truth_df["date"])
        truth_df["true_regime_21"] = truth_df["regime"].shift(-21)
        df_pred = df_pred.merge(truth_df[["date", "true_regime_21"]], on="date", how="left")
        accuracy_data = True
    except Exception as e:
        print(f"[WARN] Could not load ground truth: {e}")
        df_pred["true_regime_21"] = None

    results = []
    correct = 0
    total_valid = 0

    regime_filter = StatefulRegimeFilter(lock_days=5, checklist_threshold=4)

    for i, row in df_pred.iterrows():
        p = probs[i]
        confidence = float(np.max(p))
        
        pred_label, is_locked, trigger = regime_filter.process(row, p)

        true_regime = row.get("true_regime_21")
        is_valid = pd.notna(true_regime) and true_regime in ["bear", "sideways", "bull"]
        if is_valid:
            total_valid += 1
            if pred_label == true_regime:
                correct += 1

        rolling_acc = float(correct / total_valid) if total_valid > 0 else None

        results.append({
            "date":         str(row["date"].date()),
            "close":        float(row["close"]),
            "symbol":       args.symbol,
            "trained_on":   args.trained_on,
            "is_oos":       True,
            "regime":       pred_label,
            "regime_actual": str(true_regime) if is_valid else None,
            "confidence":   confidence,
            "prob_bear":    float(p[0]),
            "prob_sideways": float(p[1]),
            "prob_bull":    float(p[2]),
            "accuracy":     rolling_acc,
            "is_locked":    is_locked,
            "trigger":      trigger,
            "checklist":    calculate_checklist(row, p),
            # Dashboard metrics
            "ret_21d":    float(row.get("ret_21d", 0)) if pd.notna(row.get("ret_21d")) else 0,
            "ret_63d":    float(row.get("ret_63d", 0)) if pd.notna(row.get("ret_63d")) else 0,
            "vol_21d":    float(row.get("vol_21d", 0)) if pd.notna(row.get("vol_21d")) else 0,
            "vol_63d":    float(row.get("vol_63d", 0)) if pd.notna(row.get("vol_63d")) else 0,
            "vol_ratio":  float(row.get("vol_ratio", 1)) if pd.notna(row.get("vol_ratio")) else 1,
            "skew_21d":   float(row.get("skew_21d", 0)) if pd.notna(row.get("skew_21d")) else 0,
            "rsi_14":     float(row.get("rsi_14", 50)) if pd.notna(row.get("rsi_14")) else 50,
            "macd_h":     float(row.get("macd_h", 0)) if pd.notna(row.get("macd_h")) else 0,
            "ema_gap":    float(row.get("ema_gap", 0)) if pd.notna(row.get("ema_gap")) else 0,
            "tii_21":     float(row.get("tii_21", 0)),
            "vol_of_vol_21": float(row.get("vol_of_vol_21", 0)),
            "dist_from_252h": float(row.get("dist_from_252h", 0)),
        })

    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)

    # Print OOS accuracy report
    valid_results = [r for r in results if r["regime_actual"] is not None]
    if valid_results:
        y_true = [r["regime_actual"] for r in valid_results]
        y_pred = [r["regime"] for r in valid_results]
        oos_acc = accuracy_score(y_true, y_pred)
        print(f"\n{'='*60}")
        print(f"  OOS ACCURACY ({args.trained_on} -> {args.symbol}): {oos_acc:.4f} ({oos_acc*100:.1f}%)")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=["bear", "sideways", "bull"], zero_division=0))

    print(f"\n[OK] Saved {len(results)} rows -> {args.json_out}")


if __name__ == "__main__":
    main()
