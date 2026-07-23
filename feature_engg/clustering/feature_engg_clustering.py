import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import hashlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from umap import UMAP
from sklearn.preprocessing import normalize

# 1. Establish absolute baseline repository context
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from constants import (
    CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, FEATURE_PIPELINE_PATH,
    COMPACT_DIM, get_onnx_model, mean_pooling, CACHE_DIR,
    AUTO_INC_ID, BOOLEAN_MAPPING
)
from utils import Utils

# Force future pandas options to suppress downcasting noise
pd.set_option('future.no_silent_downcasting', True)

tokenizer, ort_session = get_onnx_model()

def cast_binary_features(X, columns):
    df = X.copy()
    for col in columns:
        normalized = df[col].astype(str).str.lower().str.strip()
        # True -> 1, False -> 0, unknown values -> 0
        df[col] = normalized.map(BOOLEAN_MAPPING).fillna(False).astype(np.int8)
            
    return df


def calculate_umap_params(no_of_rows: int, embedding_dims: int,):
    """
    Determine UMAP parameters from dataset size and embedding size.
    """

    # Compression heuristic
    n_components = int(
        np.clip(np.sqrt(embedding_dims) * 0.6, 8, 20)
    )

    # Neighborhood heuristic
    n_neighbors = int(
        np.clip(np.log2(max(no_of_rows, 2)) * 3, 10, 50)
    )

    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
    }


def get_column_hash(series: pd.Series) -> str:
    """Generates a fast, unique MD5 signature based on column strings and structure."""
    data_string = "".join(series.fillna("").astype(str).tolist()) + str(series.shape)
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def generate_sentiment(X, sentiment_col):
    df = X.copy()

    if sentiment_col not in df.columns:
        print(f"⚠ Sentiment column '{sentiment_col}' not found.")
        return df

    ratings = pd.to_numeric(df[sentiment_col], errors="coerce")
    valid = ratings.notna()

    if valid.sum() == 0:
        print("⚠ No valid ratings found.")
        return df

    min_rating = ratings[valid].min()
    max_rating = ratings[valid].max()

    if min_rating == max_rating:
        score = pd.Series(0.5, index=df.index)
    else:
        score = (ratings - min_rating) / (max_rating - min_rating)

    score = score.fillna(0.5)

    df["Feature_Sentiment_Score"] = score

    print(
        f"✅ Sentiment generated from '{sentiment_col}' "
        f"(min={min_rating}, max={max_rating})"
    )

    return df

def load_cached_embeddings(cache_file: str, index: pd.Index) -> pd.DataFrame:
    """
    Loads cached embeddings from disk.

    If cache dimension > COMPACT_DIM,
    slices and re-normalizes the vectors.
    """

    cached_data = pd.read_parquet(cache_file)

    if not cached_data.index.equals(index):
        raise ValueError("Cached embedding index does not match input dataframe.")

    emb_cols = sorted(
        [c for c in cached_data.columns if "_onnx_" in c],
        key=lambda x: int(x.split("_")[-1])
    )

    matrix = cached_data[emb_cols].to_numpy()

    if matrix.shape[1] > COMPACT_DIM:
        print(
            f"✂️ Retroactively slicing cache "
            f"from {matrix.shape[1]} → {COMPACT_DIM}"
        )

        matrix = matrix[:, :COMPACT_DIM]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-9, None)

    emb_df = pd.DataFrame(matrix, index=index)

    prefix = emb_cols[0].rsplit("_", 1)[0]

    emb_df.columns = [
        f"{prefix}_{i}"
        for i in range(COMPACT_DIM)
    ]

    return emb_df

def compute_embeddings(series: pd.Series) -> pd.DataFrame:
    """
    Computes compact ONNX sentence embeddings
    for a text column.
    """

    BATCH_SIZE = 2048

    text_list = (
        series
        .fillna("")
        .astype(str)
        .tolist()
    )

    all_embeddings = []

    for start in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64)
        }

        model_output = ort_session.run(None, ort_inputs)
        pooled = mean_pooling(model_output,encoded["attention_mask"])
        pooled = pooled[:, :COMPACT_DIM]

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-9, None)
        all_embeddings.append(pooled)

    matrix = np.vstack(all_embeddings)

    emb_df = pd.DataFrame(
        matrix,
        index=series.index
    )

    emb_df.columns = [
        f"{series.name}_onnx_{i}"
        for i in range(COMPACT_DIM)
    ]

    return emb_df

def save_embeddings_to_cache(
    cache_file: str,
    text_series: pd.Series,
    emb_df: pd.DataFrame
) -> None:
    """
    Saves original text and computed embeddings
    to a parquet cache.
    """

    cache_df = pd.concat(
        [text_series.to_frame(), emb_df],
        axis=1
    )

    cache_df.to_parquet(
        cache_file,
        compression="snappy"
    )

    print(
        f"💾 Saved {emb_df.shape[1]} embedding dimensions "
        f"to cache."
    )


def apply_onnx_embeddings(X, text_columns):
    """
    Transforms text variables into normalized Matryoshka embeddings.
    Loads column metadata dynamically from the audit plan and uses local disk caching.
    """
    df = X.copy()
    
    # --- 2. Core Embedding Processing Loop ---
    for col in text_columns:
        cache_file = os.path.join(
            CACHE_DIR,
            f"dual_axis_{col}_{get_column_hash(df[col])}.parquet"
        )

        if os.path.exists(cache_file):
            print(f"♻️ Cache Hit: {col}")
            emb_df = load_cached_embeddings(cache_file, df.index)
        else:
            print(f"⚡ Cache Miss: {col}")
            emb_df = compute_embeddings(df[col])
            save_embeddings_to_cache(cache_file, df[col], emb_df)

        df.drop(columns=emb_df.columns, errors="ignore", inplace=True)
        df = pd.concat([df, emb_df], axis=1)

    df.drop(columns=text_columns, errors="ignore", inplace=True)
    return df


def run_gold_engineering():
    print("🚀 Phase 5: Executing feature engineering pipeline")

    with open(GOLD_AUDIT_REPORT, 'r') as f:
        audit_plan = json.load(f)

    subject_id = audit_plan.get("subject_id")
    use_parquet = audit_plan.get("use_parquet", False)
    
    df = pd.read_csv(CLEANED_DATASET_PATH)

    encoding_cfg = audit_plan.get("encoding", {})
    trans_cfg = audit_plan.get("transformations", {})
    scaling_cfg = audit_plan.get("scaling", {})
    embed_cfg = audit_plan.get("text_embeddings", {})
    sentiment = audit_plan.get("sentiment", {})
    
    longtext_cols = embed_cfg.get("embed_columns", [])

    # Accumulate and strictly cast all planned categorical targets to string objects
    all_categorical_targets = set()
    if encoding_cfg.get("rare_label"):
        all_categorical_targets.update(encoding_cfg["rare_label"])
    if encoding_cfg.get("one_hot"):
        all_categorical_targets.update(encoding_cfg["one_hot"])
        
    valid_cat_targets = [c for c in all_categorical_targets if c in df.columns]
    if valid_cat_targets:
        print(f"🔤 Casting potential mixed continuous types to string objects: {valid_cat_targets}")
        df[valid_cat_targets] = df[valid_cat_targets].fillna("Missing").astype(str)

    steps = []

    if valid_cat_targets:
        steps.append(('gold_categorical_imputer', CategoricalImputer(
            imputation_method='missing', fill_value='Missing', variables=valid_cat_targets
        )))

    if encoding_cfg.get("rare_label"):
        valid_rare = [c for c in encoding_cfg["rare_label"] if c in df.columns]
        if valid_rare:
            steps.append(('rare_encoder', RareLabelEncoder(tol=0.05, n_categories=1, variables=valid_rare)))
            
    if encoding_cfg.get("one_hot"):
        valid_ohe = [c for c in encoding_cfg["one_hot"] if c in df.columns]
        if valid_ohe:
            steps.append(('onehot_encoder', OneHotEncoder(drop_last_binary=True, variables=valid_ohe)))

    # --- Structural Boolean / Binary Casting Block ---
    if encoding_cfg.get("binary_cast"):
        valid_binary = [c for c in encoding_cfg["binary_cast"] if c in df.columns]
        if valid_binary:
            print(f"🔢 Locking Boolean columns to safe single numeric tracks: {valid_binary}")
            steps.append(('binary_caster', FunctionTransformer(
                func=cast_binary_features,
                kw_args={'columns': valid_binary},
                validate=False
            )))

    if trans_cfg.get("log"):
        valid_log = [c for c in trans_cfg["log"] if c in df.columns]
        if valid_log:
            steps.append(('log_transformer', LogCpTransformer(variables=valid_log)))
            
    if trans_cfg.get("yeo_johnson"):
        valid_yj = [c for c in trans_cfg["yeo_johnson"] if c in df.columns]
        if valid_yj:
            steps.append(('yeo_transformer', YeoJohnsonTransformer(variables=valid_yj)))

    if longtext_cols:
        valid_text_cols = [c for c in longtext_cols if c in df.columns]
        if valid_text_cols:
            steps.append(('onnx_embedder', FunctionTransformer(
                func=apply_onnx_embeddings,
                kw_args={'text_columns': valid_text_cols},
                validate=False
            )))

    if sentiment.get("strategy") != "none":
        steps.append(("sentiment_generator", FunctionTransformer(
            func=generate_sentiment,
            kw_args={'sentiment_col': sentiment["source_column"]},
            validate=False
        )))

    if scaling_cfg.get("robust"):
        valid_robust = [c for c in scaling_cfg["robust"] if c in df.columns]
        if valid_robust:
            steps.append(('robust_scaler', SklearnTransformerWrapper(
                transformer=RobustScaler(), variables=valid_robust
            )))

    if scaling_cfg.get("standard"):
        valid_standard = [c for c in scaling_cfg["standard"] if c in df.columns]
        if valid_standard:
            steps.append(('standard_scaler', SklearnTransformerWrapper(
                transformer=StandardScaler(), variables=valid_standard
            )))

    # Min-Max Scaling Block for Bounded Ordinal Anchors ---
    if scaling_cfg.get("min_max"):
        valid_min_max = [c for c in scaling_cfg["min_max"] if c in df.columns]
        if valid_min_max:
            print(f"🛡️ Shielding boundary limits via MinMax Scaling for: {valid_min_max}")
            steps.append(('minmax_scaler', SklearnTransformerWrapper(
                transformer=MinMaxScaler(), variables=valid_min_max
            )))

    # 1. Initialize the base tracking pipeline and execute raw transformations
    pipeline = Pipeline(steps)
    df_engineered = pipeline.fit_transform(df.copy())
    all_onnx_cols = [c for c in df_engineered.columns if "_onnx_" in c]

    umap_models_registry = {}
    if all_onnx_cols:
        print(f"📊 Total raw vector features detected: {len(all_onnx_cols)}")

        # Preserve original ordering
        source_text_cols = list(
            dict.fromkeys(
                c.split("_onnx_")[0]
                for c in all_onnx_cols
            )
        )

        print(
            f"🔀 Isolated {len(source_text_cols)} "
            f"independent text origins: {source_text_cols}"
        )

        umap_feature_dfs = []
        for source_col in source_text_cols:
            specific_vector_cols = [
                c
                for c in all_onnx_cols
                if c.startswith(f"{source_col}_onnx_")
            ]

            print(
                f"📉 Processing [{source_col}] "
                f"({len(specific_vector_cols)} embedding dimensions)"
            )

            # ----------------------------------------------------------
            # Extract raw embedding matrix
            # Shape: (rows, embedding_dims)
            # ----------------------------------------------------------

            raw_vectors = df_engineered[
                specific_vector_cols
            ].to_numpy()

            # ----------------------------------------------------------
            # Determine optimal UMAP parameters
            # ----------------------------------------------------------

            params = calculate_umap_params(
                no_of_rows=raw_vectors.shape[0],
                embedding_dims=raw_vectors.shape[1],
            )

            print(
                f"📌 [{source_col}] "
                f"Rows={raw_vectors.shape[0]} "
                f"Dims={raw_vectors.shape[1]} "
                f"→ "
                f"UMAP({params['n_components']}) "
                f"Neighbors={params['n_neighbors']}"
            )

            # ----------------------------------------------------------
            # Fit UMAP
            # ----------------------------------------------------------

            umap_model = UMAP(
                n_neighbors=params["n_neighbors"],
                n_components=params["n_components"],
                metric="cosine",
                min_dist=0.05,
                random_state=42,
            )

            umap_results = umap_model.fit_transform(raw_vectors)

            # ----------------------------------------------------------
            # Convert UMAP output into dataframe
            # ----------------------------------------------------------

            feature_names = [
                f"{source_col.upper()}_UMAP_{i}"
                for i in range(umap_results.shape[1])
            ]

            X_emb_single = pd.DataFrame(
                umap_results,
                columns=feature_names,
                index=df_engineered.index,
            )

            umap_feature_dfs.append(X_emb_single)

            # Save fitted model for runtime inference
            umap_models_registry[source_col] = umap_model

        # --------------------------------------------------------------
        # Merge all UMAP features
        # --------------------------------------------------------------

        X_emb_final = pd.concat(
            umap_feature_dfs,
            axis=1,
        )

        umap_feature_names = X_emb_final.columns.tolist()

        # Remove original 384-d embedding columns
        df_engineered = df_engineered.drop(
            columns=all_onnx_cols
        )

    else:

        X_emb_final = pd.DataFrame(
            index=df_engineered.index
        )
        umap_feature_names = []


    trace_text_cols = df_engineered.select_dtypes(include=['object', 'string']).columns.tolist()    

    payload_blocks = [
        df_engineered.drop(columns=trace_text_cols, errors='ignore'), 
        X_emb_final
    ]
        
    X_final = pd.concat(payload_blocks, axis=1)

    # Eliminate duplicated feature naming collisions safely
    if X_final.columns.duplicated().any():
        X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    # Restrict the scaling data strictly to mathematical arrays
    X_numeric_only = X_final.select_dtypes(include=[np.number])
    X_final_scaled_df = X_numeric_only.copy()
    print(f"🏁 Final reconstructed Gold Matrix dimensions: {X_final_scaled_df.shape}")

    # Bundle the final deployment dictionary mapping payload safely
    feature_engg_pipeline = {
        "subject_id": subject_id,
        "engineering_pipeline": pipeline,
        "embedding_cols": all_onnx_cols,
        "trace_text_cols": trace_text_cols,
        "umap_transformers_dict": umap_models_registry,
        "umap_feature_names": umap_feature_names
    }
    
    joblib.dump(feature_engg_pipeline, FEATURE_PIPELINE_PATH, compress=3)
    print(f"💾 Feature engineering pipeline saved to: {FEATURE_PIPELINE_PATH}")

    # Re-append structural identities for downstream dataset joins
    if subject_id and subject_id in df.columns:
        X_final_scaled_df[subject_id] = df[subject_id]
        
    X_final_scaled_df.index.name = AUTO_INC_ID
    Utils.save_to_parquet(X_final_scaled_df, GOLD_DATASET_PATH)


if __name__ == "__main__":
    run_gold_engineering()