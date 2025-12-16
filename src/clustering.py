import pandas as pd
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

RANDOM_STATE = 42
KMEANS_SAMPLE_SIZE = 200_000
DBSCAN_SAMPLE_SIZE = 30_000
HIER_SAMPLE_SIZE = 20_000

# ----------------------------
# Load data
# ----------------------------
print("Loading feature dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"Full dataset shape: {df.shape}")

cluster_features = [
    "Latitude",
    "Longitude",
    "hour",
    "day_of_week",
    "month"
]

# ======================================================
# 1️⃣ MINI-BATCH K-MEANS (on 200k)
# ======================================================
print("\nRunning MiniBatch K-Means...")

df_kmeans = df.sample(n=KMEANS_SAMPLE_SIZE, random_state=RANDOM_STATE)
X_km = df_kmeans[cluster_features]

scaler = StandardScaler()
X_km_scaled = scaler.fit_transform(X_km)

kmeans = MiniBatchKMeans(
    n_clusters=7,
    batch_size=10_000,
    random_state=RANDOM_STATE
)

kmeans_labels = kmeans.fit_predict(X_km_scaled)

kmeans_sil = silhouette_score(X_km_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_km_scaled, kmeans_labels)

print(f"MiniBatch K-Means Silhouette Score: {kmeans_sil:.3f}")
print(f"MiniBatch K-Means Davies-Bouldin Index: {kmeans_db:.3f}")

df_kmeans["kmeans_cluster"] = kmeans_labels

# ======================================================
# 2️⃣ DBSCAN (on 30k)
# ======================================================
print("\nRunning DBSCAN...")

df_dbscan = df.sample(n=DBSCAN_SAMPLE_SIZE, random_state=RANDOM_STATE)
X_db = df_dbscan[cluster_features]

X_db_scaled = scaler.fit_transform(X_db)

dbscan = DBSCAN(eps=0.7, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_db_scaled)

mask = dbscan_labels != -1

if mask.sum() > 0 and len(set(dbscan_labels)) > 1:
    dbscan_sil = silhouette_score(X_db_scaled[mask], dbscan_labels[mask])
    dbscan_db = davies_bouldin_score(X_db_scaled[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette Score: {dbscan_sil:.3f}")
    print(f"DBSCAN Davies-Bouldin Index: {dbscan_db:.3f}")
else:
    print("DBSCAN produced mostly noise (expected for sparse regions).")

df_dbscan["dbscan_cluster"] = dbscan_labels

# ======================================================
# 3️⃣ HIERARCHICAL (on 20k)
# ======================================================
print("\nRunning Hierarchical Clustering...")

df_hier = df.sample(n=HIER_SAMPLE_SIZE, random_state=RANDOM_STATE)
X_hier = df_hier[cluster_features]

X_hier_scaled = scaler.fit_transform(X_hier)

hierarchical = AgglomerativeClustering(n_clusters=7)
hier_labels = hierarchical.fit_predict(X_hier_scaled)

hier_sil = silhouette_score(X_hier_scaled, hier_labels)
hier_db = davies_bouldin_score(X_hier_scaled, hier_labels)

print(f"Hierarchical Silhouette Score: {hier_sil:.3f}")
print(f"Hierarchical Davies-Bouldin Index: {hier_db:.3f}")

df_hier["hierarchical_cluster"] = hier_labels

# ----------------------------
# Save results
# ----------------------------
df_kmeans.to_csv(
    os.path.join(OUTPUT_DIR, "crimes_kmeans_sample.csv"),
    index=False
)

df_dbscan.to_csv(
    os.path.join(OUTPUT_DIR, "crimes_dbscan_sample.csv"),
    index=False
)

df_hier.to_csv(
    os.path.join(OUTPUT_DIR, "crimes_hierarchical_sample.csv"),
    index=False
)

print("\nSTEP 4 COMPLETE (OPTIMIZED & STABLE)")
