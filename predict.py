"""
predict.py
==========
Main inference script to generate regime predictions using a trained XGBoost model.
Uses output/features.json (from Rust core) and output/xgb_model.pkl (trained model).
Outputs output/regime_ml.json compatible with the frontend.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import argparse
import sys
import os
import pickle
import json

# --- Config ---
MODEL_PATH    = "output/xgb_model.pkl"
OUTPUT_PATH   = "output/regime_ml.json"
CONFIDENCE_TH = 0.65

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/features.json")
    parser.add_argument("--symbol", default="NIFTY_500")
    parser.add_argument("--output", default="output/regime_ml.json")
    parser.add_argument("--horizon", type=int, default=21)
    return parser.parse_args()

XGB_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d",
    "vol_10d", "vol_21d", "vol_ratio",
    "skew_21d", "kurt_21d",
    "rsi_14", "rsi_21",
    "macd_h", "bb_pct", "ema_gap", "adx",
    "drawdown_21d",
    "drawdown_63d",
    "mom_divergence",
    "vol_spike",
]

def calculate_checklist(row, probs):
    """Calculates the confirmation checklist for Bull and Bear regimes."""
    p_bear, p_side, p_bull = probs
    
    bull_checks = {
        "momentum_21d": bool(row.get("ret_21d", 0) > 0.05),
        "trend_63d": bool(row.get("ret_63d", 0) > 0.08),
        "vol_stability": bool(row.get("vol_21d", 0) < row.get("vol_63d", 0)),
        "risk_cooling": bool(row.get("vol_ratio", 0) < 0.90),
        "tail_risk": bool(row.get("skew_21d", 0) > -0.5),
        "ema_cross": bool(row.get("ema_gap", 0) > 0),
        "macd_hist": bool(row.get("macd_h", 0) > 0),
        "rsi_healthy": bool(50 <= row.get("rsi_14", 0) <= 70),
        "ml_confidence": bool(p_bull >= 0.65)
    }
    
    bear_checks = {
        "momentum_21d": bool(row.get("ret_21d", 0) < -0.05),
        "trend_63d": bool(row.get("ret_63d", 0) < -0.08),
        "panic_vol": bool(row.get("vol_21d", 0) > (row.get("vol_63d", 0) * 1.3)),
        "tail_crash": bool(row.get("skew_21d", 0) < -1.0),
        "ema_death_cross": bool(row.get("ema_gap", 0) < 0),
        "macd_hist_neg": bool(row.get("macd_h", 0) < 0),
        "rsi_weak": bool(row.get("rsi_14", 0) < 40),
        "ml_confidence": bool(p_bear >= 0.65)
    }
    
    return {"bull": bull_checks, "bear": bear_checks}

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_structural_features(df):
    """Derived features that are key for bear detection structure."""
    df = df.copy()
    rolling_max_21 = df["close"].rolling(21, min_periods=1).max()
    rolling_max_63 = df["close"].rolling(63, min_periods=1).max()

    df["drawdown_21d"]   = (df["close"] - rolling_max_21) / rolling_max_21
    df["drawdown_63d"]   = (df["close"] - rolling_max_63) / rolling_max_63
    df["mom_divergence"] = df["ret_5d"] - df["ret_21d"]
    df["vol_spike"]      = df["vol_10d"] / df["vol_63d"]

    # Handle numeric issues
    cols = ["drawdown_21d", "drawdown_63d", "mom_divergence", "vol_spike"]
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df

def assign_regime(probs, threshold=CONFIDENCE_TH):
    """Assign regime based on max probability and confidence threshold."""
    max_p = max(probs)
    idx = int(np.argmax(probs))
    if max_p < threshold:
        return "sideways"
    return ["bear", "sideways", "bull"][idx]

def main():
    args = get_args()
    print(f"--- Regime Inference: {args.symbol} ---")
    
    # 1. Load data
    try:
        df = load_data(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    # 2. Add structural features
    df = add_structural_features(df)

    # 3. Load model and scaler
    try:
        with open(MODEL_PATH, "rb") as f:
            model, scaler = pickle.load(f)
        print(f"Loaded model and scaler from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Filter rows with valid features
    df_pred = df.dropna(subset=XGB_FEATURES).copy()
    if df_pred.empty:
        print("No rows with complete features for prediction.")
        return

    # 5. Inference
    X = scaler.transform(df_pred[XGB_FEATURES].values)
    probs = model.predict_proba(X)

    # 6. Map predictions and calculate accuracy/hit rate
    # Load original clustered labels for ground truth calculation
    try:
        with open("output/regime_clustered.json", "r") as f:
            df_truth = pd.DataFrame(json.load(f))
            df_truth["date"] = pd.to_datetime(df_truth["date"])
            # The "truth" at time t is the regime N days ahead
            df_truth["true_regime"] = df_truth["regime"].shift(-args.horizon)
            df = df.merge(df_truth[["date", "true_regime"]], on="date", how="left")
    except Exception as e:
        print(f"Warning: Could not load ground truth (regime_clustered.json): {e}")
        df["true_regime"] = None

    res = []
    prob_idx = 0
    correct_count = 0
    total_valid_for_acc = 0
    
    for i, row in df.iterrows():
        date_str = str(row["date"])
        if " " in date_str:
            date_str = date_str.split(" ")[0]
            
        out_row = {
            "date": date_str,
            "symbol": str(args.symbol),
            "close": float(row["close"]),
            "ret_21d": float(row["ret_21d"]) if pd.notna(row["ret_21d"]) else 0.0,
            "ret_63d": float(row["ret_63d"]) if pd.notna(row["ret_63d"]) else 0.0,
            "vol_21d": float(row["vol_21d"]) if pd.notna(row["vol_21d"]) else 0.0,
            "vol_63d": float(row["vol_63d"]) if pd.notna(row["vol_63d"]) else 0.0,
            "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else 1.0,
            "skew_21d": float(row["skew_21d"]) if pd.notna(row["skew_21d"]) else 0.0,
            "rsi_14": float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else 50.0,
            "macd_h": float(row["macd_h"]) if pd.notna(row["macd_h"]) else 0.0,
            "ema_gap": float(row["ema_gap"]) if pd.notna(row["ema_gap"]) else 0.0,
            "regime": None,
            "regime_actual": str(row["true_regime"]) if pd.notna(row["true_regime"]) else None,
            "prob_bear": 0.0,
            "prob_sideways": 0.0,
            "prob_bull": 0.0,
            "confidence": 0.0,
            "accuracy": None,
            "checklist": None
        }
        
        if i in df_pred.index:
            p = probs[prob_idx]
            pred_regime = assign_regime(p)
            out_row["prob_bear"] = float(p[0])
            out_row["prob_sideways"] = float(p[1])
            out_row["prob_bull"] = float(p[2])
            out_row["confidence"] = float(np.max(p))
            out_row["regime"] = pred_regime
            out_row["checklist"] = calculate_checklist(row, p)
            prob_idx += 1
            
            if out_row["regime"] is not None and out_row["regime_actual"] is not None:
                total_valid_for_acc += 1
                if out_row["regime"] == out_row["regime_actual"]:
                    correct_count += 1
                out_row["accuracy"] = float(correct_count / total_valid_for_acc) if total_valid_for_acc > 0 else 0.0
            
        res.append(out_row)

    save_path = f"frontend/frontend/public/data/regime_{args.symbol}.json"
    with open(save_path, "w") as f:
        json.dump(res, f, indent=4)
        
    print(f"✅ Prediction Complete for {args.symbol} -> Saved to {save_path}")

if __name__ == "__main__":
    main()
