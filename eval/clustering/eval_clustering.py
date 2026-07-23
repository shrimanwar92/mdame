import sys
import os
import pandas as pd
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import (
    GOLD_LABELLED_DATASET_PATH,
    TRAIN_AUDIT_REPORT,
    CLUSTER_BASELINE_PATH
)

def generate_cluster_baseline(cluster_col="Cluster"):
    """
    Generate baseline statistics for inference drift detection.
    """

    print("Loading labelled dataset...")
    df = pd.read_parquet(GOLD_LABELLED_DATASET_PATH)

    print("Loading training audit...")
    with open(TRAIN_AUDIT_REPORT, "r") as f:
        audit = json.load(f)

    training_features = audit["training_features"]
    algorithm = audit["recommended_algorithm"]

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    if cluster_col not in df.columns:
        raise ValueError(f"{cluster_col} not found.")

    missing = [c for c in training_features if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing training features:\n{missing}"
        )

    print(f"Clusters : {df[cluster_col].nunique()}")
    print(f"Samples  : {len(df)}")

    baseline = {
        "algorithm": algorithm,
        "training_features": training_features,
        "cluster_count": int(df[cluster_col].nunique()),
        "n_samples": int(len(df)),
    }

    # -------------------------------------------------------
    # Global statistics
    # -------------------------------------------------------

    all_distances = []

    centroids = {}
    cluster_distribution = {}

    for cluster_id, group in df.groupby(cluster_col):

        X = group[training_features].values

        centroid = X.mean(axis=0)

        centroids[str(cluster_id)] = centroid.tolist()

        distances = np.linalg.norm(
            X - centroid,
            axis=1
        )

        all_distances.extend(distances.tolist())

        cluster_distribution[str(cluster_id)] = (
            len(group) / len(df)
        )

    all_distances = np.asarray(all_distances)

    baseline["centroids"] = centroids

    baseline["global_distance"] = {
        "mean": float(np.mean(all_distances)),
        "p90": float(np.percentile(all_distances, 90)),
        "p95": float(np.percentile(all_distances, 95)),
    }

    baseline["cluster_distribution"] = cluster_distribution

    with open(CLUSTER_BASELINE_PATH, "w") as f:
        json.dump(
            baseline,
            f,
            indent=4,
        )

    print(f"\nBaseline saved to:{CLUSTER_BASELINE_PATH}")

    return baseline