import mlflow 
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/processed/crimes_500k_features.csv"
SAMPLE_SIZE = 200_000
N_CLUSTERS = 7
RANDOM_STATE = 42

cluster_features = [
    "Latitude",
    "Longitude",
    "hour",
    "day_of_week",
    "month"
]

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

X = df_sample[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# MLflow experiment
# ----------------------------
mlflow.set_experiment("Crime Hotspot Clustering")

with mlflow.start_run():

    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=10_000,
        random_state=RANDOM_STATE
    )

    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    # Log params
    mlflow.log_param("algorithm", "MiniBatchKMeans")
    mlflow.log_param("n_clusters", N_CLUSTERS)
    mlflow.log_param("sample_size", SAMPLE_SIZE)

    # Log metrics
    mlflow.log_metric("silhouette_score", sil)
    mlflow.log_metric("davies_bouldin_index", db)

    # Save clustered output
    df_sample["cluster"] = labels
    output_path = "data/processed/mlflow_kmeans_output.csv"
    df_sample.to_csv(output_path, index=False)

    # Log artifact
    mlflow.log_artifact(output_path)

    print("MLflow run logged successfully")
