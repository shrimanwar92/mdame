import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import (
    CLEANED_DATASET_PATH, GOLD_DATASET_PATH, MODEL_METRICS_REPORT, 
    BEST_MODEL_PATH, CLUSTERING_IMAGE_PATH, GOLD_LABELLED_DATASET_PATH, 
    TRAIN_AUDIT_REPORT, AUTO_INC_ID
)
from utils import Utils

# ==============================================================================
# --- CORE ENGINE UTILITIES ---
# ==============================================================================

def load_gold_dataset_context():
    """Unified entrypoint for pipeline configuration and data loading with robust index locking."""
    with open(TRAIN_AUDIT_REPORT, 'r') as f:
        train_audit = json.load(f)

    subject_id = train_audit.get("subject_id")
    use_parquet = train_audit.get("use_parquet", False)
    target_algo = train_audit.get("recommended_algorithm", "GMM") 
    training_features = train_audit.get("training_features", [])
    omitted_features = train_audit.get("omitted_features", [])

    # 1. Load the engineered gold dataset (Parquet natively retains AUTO_INC_ID as the index)
    df = pd.read_parquet(GOLD_DATASET_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # 2. Load the raw text table for pulling profile features back
    df_raw = pd.read_csv(CLEANED_DATASET_PATH)
        
    # Explicitly lock the index of df_raw to use the actual generated column IDs
    if AUTO_INC_ID in df_raw.columns:
        df_raw = df_raw.set_index(AUTO_INC_ID)
    elif subject_id and subject_id in df_raw.columns:
        df_raw = df_raw.set_index(subject_id)
        df_raw.index.name = AUTO_INC_ID

    return df, df_raw, training_features, omitted_features, target_algo, use_parquet


def generate_cluster_plot(X, labels, algo_name):
    """Generates a compressed 2D scatter visualization of the feature matrix."""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=pca_df, s=80, alpha=0.6)
    plt.title(f'Cluster Distribution ({algo_name}) - Spatial Boundary Analysis', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(CLUSTERING_IMAGE_PATH)
    plt.close()
    print(f"🎨 Generated {algo_name} visualization.")


# ==============================================================================
# 🚀 WORKFLOW 1: MODEL TRAINING 
# ==============================================================================
def run_training_pipeline():
    print("🧠 Phase 6: Model Audit & Clustering Initialization [TRAINING MODE]")
    df_gold, df_raw, training_features, omitted_features, target_algo, use_parquet = load_gold_dataset_context()
    
    # DECOUPLED ROUTING: Slice matrix using features dictated by the Train Audit JSON
    X_train = df_gold[training_features].to_numpy()
    
    n_samples, n_features = X_train.shape
    print(f"📊 Active Matrix Dimensions: {n_samples} rows, {n_features} features")
    print(f"🧬 Features routed for training: {training_features}")

    print(f"🔬 Executing Hyperparameter Optimization for: {target_algo}...")
    k_range = range(2, min(11, n_samples))
    
    if target_algo == "GMM":
        bics = [GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5).fit(X_train).bic(X_train) for k in k_range]
        suggested_k = k_range[np.argmin(bics)]
        print(f"💡 Optimal K determined by BIC minimization: {suggested_k}")
        final_model = GaussianMixture(n_components=suggested_k, random_state=42, reg_covar=1e-5)
        final_labels = final_model.fit_predict(X_train)
        probs = final_model.predict_proba(X_train).max(axis=1)
        algo_metrics = {"minimized_bic_score": float(final_model.bic(X_train)), "converged": bool(final_model.converged_)}
    else:
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_train).inertia_ for k in k_range]
        suggested_k = 3 
        final_model = KMeans(n_clusters=suggested_k, random_state=42, n_init=10)
        final_labels = final_model.fit_predict(X_train)
        probs = np.ones(len(df_gold))
        algo_metrics = {"final_inertia": float(final_model.inertia_)}

    # Compute Clustering Validation Metrics
    valid_mask = final_labels != -1
    unique_labels = set(final_labels) - {-1}
    silhouette = float(silhouette_score(X_train[valid_mask], final_labels[valid_mask])) if len(unique_labels) > 1 else -1.0
    ch_idx = float(calinski_harabasz_score(X_train[valid_mask], final_labels[valid_mask])) if len(unique_labels) > 1 else -1.0

    metrics_report = {
        "algorithm": target_algo, "n_samples": n_samples, "n_features": n_features,
        "estimated_clusters": len(unique_labels), "silhouette_score": silhouette, "calinski_harabasz_index": ch_idx,
        "algo_specific_metrics": algo_metrics
    }
    with open(MODEL_METRICS_REPORT, 'w') as f:
        json.dump(metrics_report, f, indent=4)

    joblib.dump(final_model, BEST_MODEL_PATH)
    
    # Append labels back to the raw structural dataframe asset (retains omitted profiling columns)
    df_gold['Cluster'] = final_labels
    df_gold['Certainty'] = probs
    
    cols_to_join = [
        c for c in df_gold.columns
        if c not in df_raw.columns
    ]
    df_labelled_output = df_raw.join(df_gold[cols_to_join], how='inner')
    
    # save to parquet
    Utils.save_to_parquet(df_labelled_output, GOLD_LABELLED_DATASET_PATH)
    generate_cluster_plot(X_train, final_labels, target_algo)
    print("🏆 Training execution success!")


if __name__ == "__main__":
    run_training_pipeline()