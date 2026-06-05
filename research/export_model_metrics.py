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
from regime_features import XGB_FEATURES, add_structural_features  # noqa: E402

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--json-out", required=True)
    p.add_argument("--predictions", required=False, help="Path to regime predictions json for actual labels")
    return p.parse_args()

def main():
    args = parse_args()
    
    with open(args.model, "rb") as f:
        model, scaler = pickle.load(f)
        
    booster = model.get_booster()
    
    # 1. Hyperparams
    params = booster.save_config()
    parsed_params = json.loads(params)
    learner = parsed_params.get("learner", {})
    tree_param = learner.get("gradient_booster", {}).get("tree_train_param", {})
    
    importances_gain = booster.get_score(importance_type='gain')
    importances_weight = booster.get_score(importance_type='weight')
    
    feat_data = []
    # Booster uses f0, f1, f2 internally if we didn't name them. Let's map back.
    for i, fname in enumerate(XGB_FEATURES):
        fkey = f"f{i}"
        gain = importances_gain.get(fkey, importances_gain.get(fname, 0))
        weight = importances_weight.get(fkey, importances_weight.get(fname, 0))
        feat_data.append({"feature": fname, "gain": gain, "weight": weight})
        
    feat_data = sorted(feat_data, key=lambda x: x["gain"], reverse=True)
    max_gain = max([f["gain"] for f in feat_data]) if feat_data else 1
    for f in feat_data:
        f["normalized_gain"] = f["gain"] / max_gain
        
    # Get total nodes (approximate)
    trees = booster.get_dump()
    total_nodes = sum(tree.count("\n") for tree in trees)
    
    # Analyze data
    with open(args.input) as f: raw = json.load(f)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = add_structural_features(df).dropna(subset=XGB_FEATURES)
    
    # Scatter representation for Beeswarm substitute
    # For top 5 features, take a random sample of 300 points
    scatter_data = []
    if not df.empty:
        sample_df = df.sample(n=min(300, len(df)), random_state=42)
        X_samp = scaler.transform(sample_df[XGB_FEATURES].values)
        probs = model.predict_proba(X_samp)
        
        top5 = [f["feature"] for f in feat_data[:5]]
        for i, row in sample_df.iterrows():
            idx = df.index.get_loc(i) if i in df.index else 0
            p_probs = probs[sample_df.index.get_loc(i)]
            regime_idx = np.argmax(p_probs)
            confidence = float(np.max(p_probs))
            
            for feat in top5:
                val = float(row[feat])
                scatter_data.append({
                    "feature": feat,
                    "value": val,
                    "prob_bear": float(p_probs[0]),
                    "prob_bull": float(p_probs[2]),
                    "confidence": confidence,
                    "regime": "bull" if regime_idx == 2 else ("bear" if regime_idx == 0 else "sideways")
                })

    # Predictions Scatter Data (Actual vs Predicted probabilities)
    # Expected value: Bear=0.0, Sideways=0.5, Bull=1.0
    predictions_data = []
    if args.predictions:
        with open(args.predictions) as f:
            pred_raw = json.load(f)
        pred_df = pd.DataFrame(pred_raw)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        
        # Merge regime columns into df based on date
        # Assuming df already has 'date' normalized
        merged_df = df.merge(pred_df[["date", "regime", "regime_actual"]], on="date", how="inner")
        
        # Use full sample for prediction scatter
        predict_df = merged_df.dropna(subset=["regime"]).sample(n=min(500, len(merged_df.dropna(subset=["regime"]))), random_state=42)
        if not predict_df.empty:
            X_all = scaler.transform(predict_df[XGB_FEATURES].values)
            probs_all = model.predict_proba(X_all)
            for i, row in predict_df.iterrows():
                actual_regime = row["regime_actual"] if "regime_actual" in row and row["regime_actual"] not in [None, "None"] else row.get("regime", "sideways")
                actual_val = 1.0 if actual_regime == 'bull' else (0.0 if actual_regime == 'bear' else 0.5)
                # Calculate expected value based on probabilities
                # Classes in model: 0=Bear, 1=Sideways, 2=Bull
                idx = predict_df.index.get_loc(i)
                p = probs_all[idx]
                predicted_val = float(0.0 * p[0] + 0.5 * p[1] + 1.0 * p[2])
                residual = float(actual_val - predicted_val)
                predictions_data.append({
                    "actual": actual_val,
                    "predicted": predicted_val,
                    "residual": residual,
                    "regime": actual_regime
                })

    # Parse Tree #0
    import re
    def parse_xgboost_tree(tree_text):
        nodes = {}
        for line in tree_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            try:
                node_idx = int(line.split(':')[0])
                if 'leaf=' in line:
                    val = float(line.split('leaf=')[1].split(',')[0])
                    nodes[node_idx] = {"name": f"Leaf: {val:.3f}", "is_leaf": True, "value": val}
                else:
                    condition = line.split('[')[1].split(']')[0]
                    # Map feature names
                    for fi, fname in reversed(list(enumerate(XGB_FEATURES))):
                        condition = condition.replace(f"f{fi}<", f"{fname} < ")
                    yes_match = re.search(r'yes=(\d+)', line)
                    no_match = re.search(r'no=(\d+)', line)
                    yes_node = int(yes_match.group(1)) if yes_match else -1
                    no_node = int(no_match.group(1)) if no_match else -1
                    nodes[node_idx] = {"name": condition, "is_leaf": False, "yes": yes_node, "no": no_node}
            except:
                pass
        
        def build_nested(idx):
            if idx not in nodes: return {"name": "Unknown"}
            node = nodes[idx]
            if node.get("is_leaf"): return {"name": node["name"]}
            return {
                "name": node["name"],
                "children": [
                     build_nested(node["yes"]),
                     build_nested(node["no"])
                ]
            }
        
        return build_nested(0) if nodes else {}

    tree_graph = parse_xgboost_tree(trees[0]) if trees else {}

    out = {
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "max_depth": tree_param.get("max_depth", 4),
            "learning_rate": tree_param.get("learning_rate", 0.02),
            "subsample": tree_param.get("subsample", 0.8),
            "colsample_bytree": tree_param.get("colsample_bytree", 0.8),
            "min_child_weight": tree_param.get("min_child_weight", 1),
        },
        "complexity": {
            "total_trees": len(trees),
            "est_nodes": total_nodes,
            "objective": learner.get("learner_model_param", {}).get("objective", "multi:softprob")
        },
        "feature_importance": feat_data,
        "scatter_analysis": scatter_data,
        "predictions_analysis": predictions_data,
        "tree_graph": tree_graph
    }
    
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Generated deep metrics -> {args.json_out}")

if __name__ == "__main__":
    main()
