import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/features.json", help="Path to input features JSON")
    parser.add_argument("--shift", type=int, default=21, help="Regime lookahead shift (5 or 21)")
    parser.add_argument("--model-out", type=str, default="output/xgb_model.pkl", help="Path to save the model")
    parser.add_argument("--json-out", type=str, default="output/regime_ml.json", help="Path to save the result")
    parser.add_argument("--symbol", type=str, default="NIFTY_500", help="Symbol name")
    return parser.parse_args()

# Removed hardcoded paths - using args instead
N_SPLITS      = 5
BEAR_BOOST    = 3.0

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

def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")
    return df

def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Base indicators for recovery
    if "vol_21d" not in df.columns and "close" in df.columns:
        df["vol_21d"] = df["close"].pct_change().rolling(21).std() * np.sqrt(252)
        
    rolling_max_21  = df["close"].rolling(21, min_periods=1).max()
    rolling_max_252 = df["close"].rolling(252, min_periods=1).max()
    
    df["drawdown_21d"]   = (df["close"] - rolling_max_21) / rolling_max_21
    df["dist_from_252h"] = (df["close"] - rolling_max_252) / rolling_max_252
    
    # Recover ret_21d if missing
    if "ret_21d" not in df.columns:
        df["ret_21d"] = df["close"].pct_change(21)
        
    df["mom_divergence"] = df["ret_10d"] - df["ret_21d"]
    df["vol_spike"]      = df["vol_21d"] / df["vol_21d"].rolling(63, min_periods=1).mean()
    
    # TII (Trend Intensity): Percentage of bars above SMA21 in last 21 candles
    sma21 = df["close"].rolling(21, min_periods=1).mean()
    df["tii_21"] = df["close"].rolling(21).apply(lambda x: (x > sma21.iloc[x.index[-1]]).sum() / 21.0, raw=False)
    
    # Vol of Vol: captures regime instability
    df["vol_of_vol_21"] = df["vol_21d"].rolling(21, min_periods=1).std()

    cols_to_fix = ["drawdown_21d", "dist_from_252h", "mom_divergence", "vol_spike", "tii_21", "vol_of_vol_21", "vol_21d", "ret_21d"]
    df[cols_to_fix] = df[cols_to_fix].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def build_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total  = len(y)
    base_w = {cls: total / count for cls, count in counts.items()}
    if 0 in base_w:
        base_w[0] *= BEAR_BOOST
    return y.map(base_w).values

def train_and_evaluate(df_clean: pd.DataFrame):
    X_raw = df_clean[XGB_FEATURES].values
    y     = df_clean["target_int"].values
    sw    = build_sample_weights(df_clean["target_int"])
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    model_cfg = dict(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        gamma=0.05, reg_alpha=0.5, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42,
    )
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_probs = np.zeros((len(y), 3))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        X_tr_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sw[train_idx]
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_tr_raw)
        X_val = scaler_fold.transform(X_val_raw)
        fold_model = XGBClassifier(**model_cfg)
        fold_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
        fold_probs = fold_model.predict_proba(X_val)
        fold_preds = fold_probs.argmax(axis=1)
        oof_preds[val_idx] = fold_preds
        oof_probs[val_idx] = fold_probs
    
    mask = oof_preds != -1
    print("\nOOF Classification Report:")
    print(classification_report(y[mask], oof_preds[mask], target_names=["bear", "sideways", "bull"], zero_division=0))
    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X_raw)
    final_model = XGBClassifier(**model_cfg)
    final_model.fit(X_all, y, sample_weight=sw, verbose=False)
    return final_model, final_scaler

def assign_regime(row, threshold=0.5):
    probs = [row["prob_bear"], row["prob_sideways"], row["prob_bull"]]
    idx = int(np.argmax(probs))
    max_p = probs[idx]
    
    # If the strongest signal is Bull or Bear and >= 0.5, take it.
    # This prevents the previous 0.65 threshold from defaulting everything to sideways.
    if idx == 1: # Sideways
        return "sideways"
    
    if max_p >= threshold:
        return ["bear", "sideways", "bull"][idx]
    
    return "sideways"

def main():
    args = parse_args()
    print(f"--- Training Regime Model for {args.symbol} ---")
    
    # 1. Load Features
    df = load_data(args.input)
    
    # 2. Add structural features
    df = add_structural_features(df)
    
    # 3. Load Labels
    label_path = "output/regime_clustered.json"
    with open(label_path) as f:
        labels = json.load(f)
    df_labels = pd.DataFrame(labels)
    df_labels["date"] = pd.to_datetime(df_labels["date"])
    
    # Merge targets
    df = df.merge(df_labels[["date", "regime"]], on="date", how="inner")
    df["target"] = df["regime"].shift(-args.shift)
    df = df.dropna(subset=["target"] + XGB_FEATURES)
    
    map_regime = {"bear": 0, "sideways": 1, "bull": 2}
    df["target_int"] = df["target"].map(map_regime)
    
    # 4. Train
    model, scaler = train_and_evaluate(df)
    
    # 5. Save results
    with open(args.model_out, "wb") as f:
        pickle.dump((model, scaler), f)
    # Generate full history with predictions
    X_all_raw = df[XGB_FEATURES].values
    X_all = scaler.transform(X_all_raw)
    probs = model.predict_proba(X_all)
    
    res_list = []
    # Reset index to ensure alignment with probs
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        p = probs[i]
        pred_idx = np.argmax(p)
        pred_label = ["bear", "sideways", "bull"][pred_idx]
        
        # Override with 0.5 confidence threshold for decisive signals
        confidence = np.max(p)
        if confidence < 0.5:
            pred_label = "sideways"
            
        res_list.append({
            "date": str(row["date"].date()),
            "close": float(row["close"]),
            "symbol": args.symbol,
            "regime": pred_label,
            "regime_actual": str(row["target"]),
            "confidence": float(confidence),
            "prob_bear": float(p[0]),
            "prob_sideways": float(p[1]),
            "prob_bull": float(p[2]),
            # Support metrics for dashboard plots
            "ret_21d": float(row.get("ret_21d", 0)) if pd.notna(row.get("ret_21d")) else 0,
            "ret_63d": float(row.get("ret_63d", 0)) if pd.notna(row.get("ret_63d")) else 0,
            "vol_21d": float(row.get("vol_21d", 0)) if pd.notna(row.get("vol_21d")) else 0,
            "vol_63d": float(row.get("vol_63d", 0)) if pd.notna(row.get("vol_63d")) else 0,
            "vol_ratio": float(row.get("vol_ratio", 1)) if pd.notna(row.get("vol_ratio")) else 1,
            "skew_21d": float(row.get("skew_21d", 0)) if pd.notna(row.get("skew_21d")) else 0,
            "rsi_14": float(row.get("rsi_14", 50)) if pd.notna(row.get("rsi_14")) else 50,
            "macd_h": float(row.get("macd_h", 0)) if pd.notna(row.get("macd_h")) else 0,
            "ema_gap": float(row.get("ema_gap", 0)) if pd.notna(row.get("ema_gap")) else 0,
            "drawdown_21d": float(row.get("drawdown_21d", 0)) if pd.notna(row.get("drawdown_21d")) else 0,
            # Niche metrics
            "tii_21": float(row.get("tii_21", 0)),
            "vol_of_vol_21": float(row.get("vol_of_vol_21", 0)),
            "dist_from_252h": float(row.get("dist_from_252h", 0))
        })
        
    with open(args.json_out, "w") as f:
        json.dump(res_list, f, indent=4)
        
    print(f"✅ Training Complete for {args.symbol} -> {args.json_out}")

if __name__ == "__main__":
    main()