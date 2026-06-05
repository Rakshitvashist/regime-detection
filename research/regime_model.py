import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_json("../output/features.json")

# Drop nulls (VERY IMPORTANT)
df = df.dropna().reset_index(drop=True)

# ----------------------------
# FEATURE MATRIX
# ----------------------------
features = ["ret_21d", "ret_63d", "vol_21d", "vol_ratio"]
X = df[features].values

# ----------------------------
# SCALE FEATURES
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# KMEANS (Regime Discovery)
# ----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["regime_cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------
# CLUSTER ANALYSIS
# ----------------------------
cluster_stats = df.groupby("regime_cluster")[features].mean()
print("\nCluster Stats:\n", cluster_stats)

# ----------------------------
# MAP CLUSTERS → LABELS (CORRECT WAY)
# ----------------------------
# Sort clusters by return (descending)
cluster_order = cluster_stats["ret_21d"].sort_values(ascending=False).index.tolist()

cluster_map = {
    cluster_order[0]: "bull",
    cluster_order[1]: "sideways",
    cluster_order[2]: "bear"
}

df["regime_label"] = df["regime_cluster"].map(cluster_map)

print("\nCluster Mapping:", cluster_map)

# ----------------------------
# ENCODE LABELS
# ----------------------------
le = LabelEncoder()
y = le.fit_transform(df["regime_label"])

label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel Encoding:", label_map)

# ----------------------------
# XGBOOST CLASSIFIER
# ----------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_scaled, y)

# ----------------------------
# EVALUATION
# ----------------------------
preds = model.predict(X_scaled)
decoded_preds = le.inverse_transform(preds)

accuracy = (decoded_preds == df["regime_label"]).mean()
print(f"\nTraining Accuracy: {accuracy:.4f}")

# ----------------------------
# PROBABILITY (VERY IMPORTANT)
# ----------------------------
proba = model.predict_proba(X_scaled)

df["prob_bear"] = proba[:, 0]
df["prob_bull"] = proba[:, 1]
df["prob_sideways"] = proba[:, 2]

# confidence = max probability
df["confidence"] = np.max(proba, axis=1)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
df["regime"] = df["regime_label"]

df.to_json("../output/regime_labeled.json", orient="records")

print("\nSaved regime_labeled.json")