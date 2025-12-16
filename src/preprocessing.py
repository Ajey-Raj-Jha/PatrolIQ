import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = "data/processed/crimes_500k_features.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 200_000
RANDOM_STATE = 42

# ----------------------------
# Load data
# ----------------------------
print("Loading feature dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Full dataset shape: {df.shape}")

df_sample = df.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Subsample shape: {df_sample.shape}")

# ----------------------------
# Features for clustering
# ----------------------------
cluster_features = ["Latitude", "Longitude", "hour", "day_of_week", "month"]
X = df_sample[cluster_features]

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# 1️⃣ MINI-BATCH K-MEANS (FAST)
# ======================================================
print("\nRunning MiniBatch K-Means...")

kmeans = MiniBatchKMeans(
    n_clusters=7,
    batch_size=10_000,
    random_state=RANDOM_STATE
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"MiniBatch K-Means Silhouette: {kmeans_sil:.3f}")
print(f"MiniBatch K-Means Davies-Bouldin: {kmeans_db:.3f}")

df_sample["kmeans_cluster"] = kmeans_labels

# ======================================================
# 2️⃣ DBSCAN (SMALLER SUBSET)
# ======================================================
print("\nRunning DBSCAN...")

dbscan_sample = df_sample.sample(n=50_000, random_state=RANDOM_STATE)
X_db = scaler.fit_transform(dbscan_sample[cluster_features])

dbscan = DBSCAN(eps=0.7, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_db)

mask = dbscan_labels != -1

if mask.sum() > 0 and len(set(dbscan_labels)) > 1:
    dbscan_sil = silhouette_score(X_db[mask], dbscan_labels[mask])
    dbscan_db = davies_bouldin_score(X_db[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette: {dbscan_sil:.3f}")
    print(f"DBSCAN Davies-Bouldin: {dbscan_db:.3f}")
else:
    print("DBSCAN produced mostly noise.")

dbscan_sample["dbscan_cluster"] = dbscan_labels

# ======================================================
# 3️⃣ HIERARCHICAL (VERY SMALL SUBSET)
# ======================================================
print("\nRunning Hierarchical Clustering...")

hier_sample = df_sample.sample(n=20_000, random_state=RANDOM_STATE)
X_hier = scaler.fit_transform(hier_sample[cluster_features])

hier = AgglomerativeClustering(n_clusters=7)
hier_labels = hier.fit_predict(X_hier)

hier_sil = silhouette_score(X_hier, hier_labels)
hier_db = davies_bouldin_score(X_hier, hier_labels)

print(f"Hierarchical Silhouette: {hier_sil:.3f}")
print(f"Hierarchical Davies-Bouldin: {hier_db:.3f}")

hier_sample["hierarchical_cluster"] = hier_labels

# ----------------------------
# Save results
# ----------------------------
df_sample.to_csv(os.path.join(OUTPUT_DIR, "crimes_cluster_sample.csv"), index=False)
dbscan_sample.to_csv(os.path.join(OUTPUT_DIR, "crimes_dbscan_sample.csv"), index=False)
hier_sample.to_csv(os.path.join(OUTPUT_DIR, "crimes_hierarchical_sample.csv"), index=False)

print("\nSTEP 4 COMPLETE (OPTIMIZED)")
