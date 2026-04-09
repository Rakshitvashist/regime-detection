import json
import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/features.json", help="Path to input features JSON")
    parser.add_argument("--output", type=str, default="output/regime_clustered.json", help="Path to save clustered regimes")
    parser.add_argument("--symbol", type=str, default="NIFTY", help="Symbol name for the output metadata")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    with open(args.input, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"--- Clustering Regime: {args.symbol} ---")
    print(f"Loaded {len(df)} rows from {args.input}")

    # -----------------------------
    # REQUIRED FEATURES
    # -----------------------------
    required_cols = [
        "ret_21d",
        "ret_63d",
        "vol_21d",
        "vol_63d",
        "vol_ratio",
        "skew_21d",
        "ret_z21"
    ]

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df_clean = df.dropna(subset=required_cols).copy()
    if df_clean.empty:
        print("❌ Error: No clean data for clustering.")
        return
        
    print(f"After cleaning: {len(df_clean)} rows")

    # -----------------------------
    # FEATURE MATRIX
    # -----------------------------
    X = df_clean[required_cols].values

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # KMEANS
    # -----------------------------
    kmeans = KMeans(
        n_clusters=3,
        n_init=50,
        max_iter=1000,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)
    df_clean["cluster"] = clusters

    # -----------------------------
    # INITIAL CLUSTER → REGIME (REFERENCE ONLY)
    # -----------------------------
    cluster_means = (
        df_clean
        .groupby("cluster")["ret_21d"]
        .mean()
        .sort_values()
    )

    sorted_clusters = cluster_means.index.tolist()

    mapping = {
        sorted_clusters[0]: "bear",
        sorted_clusters[1]: "sideways",
        sorted_clusters[2]: "bull"
    }

    df_clean["regime_kmeans"] = df_clean["cluster"].map(mapping)

    # -----------------------------
    # ✅ METHOD 1: PERCENTILE OVERRIDE
    # -----------------------------
    bear_threshold = df_clean["ret_21d"].quantile(0.20)   # bottom 20%
    bull_threshold = df_clean["ret_21d"].quantile(0.60)   # top 40%

    def refined_regime(row):
        if row["ret_21d"] <= bear_threshold:
            return "bear"
        elif row["ret_21d"] >= bull_threshold:
            return "bull"
        else:
            return "sideways"

    df_clean["regime_label"] = df_clean.apply(refined_regime, axis=1)

    # -----------------------------
    # DEBUG DISTRIBUTION
    # -----------------------------
    print("\nNew regime distribution:")
    print(df_clean["regime_label"].value_counts())

    print("\nNew regime distribution (%):")
    print(df_clean["regime_label"].value_counts(normalize=True))

    # -----------------------------
    # MERGE BACK
    # -----------------------------
    df["cluster"] = np.nan
    df["regime_label"] = None

    df.loc[df_clean.index, "cluster"] = df_clean["cluster"]
    df.loc[df_clean.index, "regime_label"] = df_clean["regime_label"]

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    output = []

    for _, row in df.iterrows():
        output.append({
            "date": str(row["date"]),
            "close": row["close"],
            "ret_21d": row["ret_21d"],
            "vol_ratio": row["vol_ratio"],
            "cluster": int(row["cluster"]) if not pd.isna(row["cluster"]) else None,
            "regime": row["regime_label"]
        })

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Saved clustered regimes to {args.output} [DONE]")

    # -----------------------------
    # DEBUG INFO
    # -----------------------------
    print("\nCluster Means (ret_21d):")
    print(cluster_means)

    print("\nCluster Mapping (reference only):")
    for k, v in mapping.items():
        print(f"Cluster {k} -> {v}")

if __name__ == "__main__":
    main()