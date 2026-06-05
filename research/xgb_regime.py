"""
xgb_regime.py
=============
Trains the in-sample XGBoost regime classifier for one instrument and writes
an HONEST walk-forward history for the dashboard.

Two things were fixed here:

  1. Label path (was a critical bug): the script previously hardcoded
     `output/regime_clustered.json` (no symbol suffix) while the pipeline
     generated `output/regime_clustered_<SYMBOL>.json`. Every model therefore
     trained against a single stale file (NIFTY's, last written months earlier)
     merged by date onto the wrong instrument's features. The label file is now
     an explicit `--labels` argument defaulting to the per-symbol file.

  2. In-sample inflation: the served regime_<SYMBOL>.json used to be produced
     by refitting on ALL rows and predicting those same rows. The dashboard
     then advertised that as a "backtest". We now serve OUT-OF-FOLD
     (walk-forward) predictions for the labelled span, plus genuine
     final-model forecasts for the unlabelled tail (the most recent ~`shift`
     bars whose forward regime is not yet known). The numbers are lower but
     real.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- shared feature pipeline (single source of truth) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "regime_features.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from regime_features import (  # noqa: E402
    XGB_FEATURES, REGIME_TO_INT, LABELS,
    add_structural_features, assign_regime, calculate_checklist, dashboard_metrics,
)

N_SPLITS = 5
BEAR_BOOST = 3.0  # up-weight the (rare) bear class during training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/features.json", help="Path to input features JSON")
    parser.add_argument("--shift", type=int, default=21, help="Regime lookahead shift (5 or 21)")
    parser.add_argument("--model-out", type=str, default="output/xgb_model.pkl", help="Path to save the model")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Served regime JSON (defaults to frontend/.../regime_<SYMBOL>.json)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Per-symbol clustered label file (defaults to output/regime_clustered_<SYMBOL>.json)")
    parser.add_argument("--symbol", type=str, default="NIFTY_500", help="Symbol name")
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    base_w = {cls: total / count for cls, count in counts.items()}
    if REGIME_TO_INT["bear"] in base_w:
        base_w[REGIME_TO_INT["bear"]] *= BEAR_BOOST
    return y.map(base_w).values


def model_config():
    return dict(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        gamma=0.05, reg_alpha=0.5, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42,
    )


def train_and_evaluate(df_labeled: pd.DataFrame):
    """Walk-forward CV for honest out-of-fold predictions, then a final model
    fit on the whole labelled set (used only to forecast genuinely unseen rows
    — the unlabelled tail and other instruments via predict_oos.py).

    Returns: final_model, final_scaler, oof_preds, oof_probs
    (oof arrays are aligned row-for-row with df_labeled.)
    """
    X_raw = df_labeled[XGB_FEATURES].values
    y = df_labeled["target_int"].values
    sw = build_sample_weights(df_labeled["target_int"])
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cfg = model_config()

    oof_preds = np.full(len(y), -1, dtype=int)
    oof_probs = np.zeros((len(y), 3))
    for train_idx, val_idx in tscv.split(X_raw):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_raw[train_idx])
        X_val = scaler_fold.transform(X_raw[val_idx])
        fold_model = XGBClassifier(**cfg)
        fold_model.fit(X_tr, y[train_idx],
                       sample_weight=sw[train_idx], eval_set=[(X_val, y[val_idx])], verbose=False)
        fold_probs = fold_model.predict_proba(X_val)
        oof_probs[val_idx] = fold_probs
        oof_preds[val_idx] = fold_probs.argmax(axis=1)

    mask = oof_preds != -1
    print("\nOOF (walk-forward) Classification Report:")
    print(classification_report(y[mask], oof_preds[mask], target_names=LABELS, zero_division=0))

    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X_raw)
    final_model = XGBClassifier(**cfg)
    final_model.fit(X_all, y, sample_weight=sw, verbose=False)
    return final_model, final_scaler, oof_preds, oof_probs


def _row_record(row, probs, symbol, regime_actual):
    regime = assign_regime(probs)
    rec = {
        "date": str(row["date"].date()),
        "close": float(row["close"]),
        "symbol": symbol,
        "regime": regime,
        "regime_actual": regime_actual,
        "confidence": float(np.max(probs)),
        "prob_bear": float(probs[0]),
        "prob_sideways": float(probs[1]),
        "prob_bull": float(probs[2]),
        "checklist": calculate_checklist(row, probs),
    }
    rec.update(dashboard_metrics(row))
    return rec


def main():
    args = parse_args()
    labels_path = args.labels or f"output/regime_clustered_{args.symbol}.json"
    json_out = args.json_out or f"frontend/frontend/public/data/regime_{args.symbol}.json"
    print(f"--- Training Regime Model for {args.symbol} ---")
    print(f"    labels : {labels_path}")
    print(f"    output : {json_out}")

    # 1. Features + structural derivations
    df = load_data(args.input)
    df = add_structural_features(df)
    df["date"] = pd.to_datetime(df["date"])

    # 2. Labels (per-symbol)
    with open(labels_path) as f:
        labels = json.load(f)
    df_labels = pd.DataFrame(labels)
    df_labels["date"] = pd.to_datetime(df_labels["date"])

    df = df.merge(df_labels[["date", "regime"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["target"] = df["regime"].shift(-args.shift)

    # 3. Labelled training set (rows with a known forward regime + all features)
    df_labeled = df.dropna(subset=["target"] + XGB_FEATURES).reset_index(drop=True)
    df_labeled["target_int"] = df_labeled["target"].map(REGIME_TO_INT)
    print(f"Training rows: {len(df_labeled)}")

    # 4. Train (walk-forward OOF + final model)
    final_model, final_scaler, oof_preds, oof_probs = train_and_evaluate(df_labeled)
    with open(args.model_out, "wb") as f:
        pickle.dump((final_model, final_scaler), f)

    # 5. Build served history:
    #    - OOF predictions for the labelled span (honest backtest rows)
    #    - final-model forecasts for the unlabelled tail (live, no ground truth)
    oof_by_date = {}
    actual_by_date = {}
    for i in range(len(df_labeled)):
        if oof_preds[i] == -1:
            continue
        d = df_labeled.loc[i, "date"]
        oof_by_date[d] = oof_probs[i]
        actual_by_date[d] = str(df_labeled.loc[i, "target"])

    last_labeled_date = df_labeled["date"].max()

    df_feat = df.dropna(subset=XGB_FEATURES).sort_values("date").reset_index(drop=True)
    tail_mask = df_feat["date"] > last_labeled_date
    tail_probs = {}
    if tail_mask.any():
        X_tail = final_scaler.transform(df_feat.loc[tail_mask, XGB_FEATURES].values)
        probs_tail = final_model.predict_proba(X_tail)
        for j, idx in enumerate(df_feat.index[tail_mask]):
            tail_probs[df_feat.loc[idx, "date"]] = probs_tail[j]

    records = []
    for _, row in df_feat.iterrows():
        d = row["date"]
        if d in oof_by_date:
            records.append(_row_record(row, oof_by_date[d], args.symbol, actual_by_date[d]))
        elif d in tail_probs:
            records.append(_row_record(row, tail_probs[d], args.symbol, None))
        # else: pre-OOF warm-up rows have no honest signal — omitted.

    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2)

    backtest_rows = sum(1 for r in records if r["regime_actual"] not in (None, "None"))
    print(f"[OK] {args.symbol}: {len(records)} served rows "
          f"({backtest_rows} walk-forward backtest + {len(records) - backtest_rows} live forecast) -> {json_out}")


if __name__ == "__main__":
    main()
