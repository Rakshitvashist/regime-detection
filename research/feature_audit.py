import argparse
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os

# Paths are derived from this file's location (repo root = parent of research/)
# instead of a machine-specific absolute path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_p = argparse.ArgumentParser(description="Feature redundancy / importance audit")
_p.add_argument("--symbol", default="NIFTY", help="Instrument to audit")
_p.add_argument("--input", default=None, help="Features JSON (defaults to per-symbol)")
_p.add_argument("--labels", default=None, help="Clustered labels JSON (defaults to per-symbol)")
_args, _ = _p.parse_known_args()

INPUT_PATH = _args.input or os.path.join(BASE_DIR, "output", f"features_{_args.symbol}.json")
LABEL_PATH = _args.labels or os.path.join(BASE_DIR, "output", f"regime_clustered_{_args.symbol}.json")
LABEL_SHIFT = 21

XGB_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d",
    "vol_10d", "vol_21d", "vol_ratio",
    "skew_21d", "kurt_21d",
    "rsi_14", "rsi_21",
    "macd_h", "bb_pct", "ema_gap", "adx",
    "drawdown_21d", "drawdown_63d",
    "mom_divergence", "vol_spike",
]

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Load data
    with open(INPUT_PATH) as f:
        df = pd.DataFrame(json.load(f))
    
    # Calculate structural features (matching xgb_regime.py)
    rolling_max_21 = df["close"].rolling(21, min_periods=1).max()
    rolling_max_63 = df["close"].rolling(63, min_periods=1).max()
    df["drawdown_21d"]   = (df["close"] - rolling_max_21) / rolling_max_21
    df["drawdown_63d"]   = (df["close"] - rolling_max_63) / rolling_max_63
    df["mom_divergence"] = df["ret_5d"] - df["ret_21d"]
    df["vol_spike"]      = df["vol_10d"] / df["vol_63d"]
    
    with open(LABEL_PATH) as f:
        df_regime = pd.DataFrame(json.load(f))
    
    df["date"] = pd.to_datetime(df["date"])
    df_regime["date"] = pd.to_datetime(df_regime["date"])
    
    df = df.merge(df_regime[["date", "regime"]], on="date", how="left")
    df["fwd_ret_21d"] = df["close"].shift(-LABEL_SHIFT) / df["close"] - 1
    
    label_map = {"bear": 0, "sideways": 1, "bull": 2}
    df["target_int"] = df["regime"].shift(-LABEL_SHIFT).map(label_map)
    df_clean = df.dropna(subset=XGB_FEATURES + ["target_int", "fwd_ret_21d"]).copy()
    
    print(f"Analyzing {len(df_clean)} samples...")

    # 1. Feature Correlation (Redundancy)
    X = df_clean[XGB_FEATURES]
    corr_matrix = X.corr()
    
    print("\n--- REDUNDANCY CHECK (Highly Correlated Pairs > 0.85) ---")
    redundant = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.85:
                redundant.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    for f1, f2, val in sorted(redundant, key=lambda x: abs(x[2]), reverse=True):
        print(f"{f1:20} | {f2:20} | Corr: {val:.4f}")

    # 2. Predictive Power (Correlation with Forward Return)
    print("\n--- PREDICTIVE POWER (Correlation with 21d Forward Return) ---")
    ret_corr = df_clean[XGB_FEATURES + ["fwd_ret_21d"]].corr()["fwd_ret_21d"].abs().sort_values(ascending=False)
    print(ret_corr.drop("fwd_ret_21d").head(10))

    # 3. Model Importance
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, df_clean["target_int"])
    
    importances = pd.Series(model.feature_importances_, index=XGB_FEATURES).sort_values(ascending=False)
    print("\n--- XGBOOST FEATURE IMPORTANCE ---")
    print(importances)

    # 4. Recommendation
    print("\n--- RECOMMENDATIONS ---")
    low_imp = importances[importances < 0.03].index.tolist()
    print(f"Candidate for Removal (Low Importance): {low_imp}")
    
    high_corr_to_prune = []
    for f1, f2, val in redundant:
        # Prune the one with lower importance
        p = f2 if importances[f1] > importances[f2] else f1
        if p not in high_corr_to_prune: high_corr_to_prune.append(p)
    print(f"Candidate for Removal (Redundancy): {high_corr_to_prune}")

if __name__ == "__main__":
    main()
