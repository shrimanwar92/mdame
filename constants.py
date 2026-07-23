from pathlib import Path
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Base Configuration
# -----------------------------------------------------------------------------

current_dir = Path(__file__).resolve().parent

DATASET = "opella_product_reviews_v3"
TARGET_COLUMN = None
AUTO_INC_ID = "generated_id"
COMPACT_DIM = 128

AWS_REGION = "eu-west-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

CACHE_DIR = Path(".pipeline_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BOOLEAN_MAPPING = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "1": True,
    "0": False,
    "t": True,
    "f": False,
}

# -----------------------------------------------------------------------------
# Dataset Paths
# -----------------------------------------------------------------------------

DATASET_PATH = Path("dataset") / f"{DATASET}.csv"

CLEANED_PATH = Path("dataset") / "cleaned" / DATASET
CLEANED_PATH.mkdir(parents=True, exist_ok=True)

CLEANED_DATASET_PATH = CLEANED_PATH / "silver_cleaned_data.csv"
GOLD_DATASET_PATH = CLEANED_PATH / "gold_feature_engineered_data.parquet"
GOLD_LABELLED_DATASET_PATH = (
    CLEANED_PATH / "gold_feature_engineered_data_labeled.parquet"
)

# -----------------------------------------------------------------------------
# Inference Paths
# -----------------------------------------------------------------------------

INFERENCE_PATH = Path("inference") / DATASET
INFERENCE_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------

ARTIFACTS_PATH = Path("artifacts") / DATASET

PROFILER_PATH = ARTIFACTS_PATH / "profiler"
MODEL_PATH = ARTIFACTS_PATH / "model"
PRODUCT_REPORTS_PATH = ARTIFACTS_PATH / "reports"

# Create all required directories
for path in (
    ARTIFACTS_PATH,
    PROFILER_PATH,
    MODEL_PATH,
    PRODUCT_REPORTS_PATH
):
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Profiler Artifacts
# -----------------------------------------------------------------------------

PROFILER_REPORT_PATH = PROFILER_PATH / "report.json"
PROFILER_CLEAN_REPORT_PATH = PROFILER_PATH / "report-clean.json"
PROFILER_FEATURE_ENGG_REPORT_PATH = (
    PROFILER_PATH / "report-feature-engg.json"
)

# -----------------------------------------------------------------------------
# Model Artifacts
# -----------------------------------------------------------------------------

GOLD_MAPPED_DATASET_PATH = MODEL_PATH / "gold_mapping.parquet"
FEATURE_PIPELINE_PATH = MODEL_PATH / "feature_pipeline.joblib"
BEST_MODEL_PATH = MODEL_PATH / "model.joblib"
CLEANING_PIPELINE_PATH = MODEL_PATH / "cleaning_pipeline.joblib"
CLUSTER_EMBEDDINGS_PATH = MODEL_PATH / "cluster_embeddings.joblib"

# -----------------------------------------------------------------------------
# Metadata Artifacts
# -----------------------------------------------------------------------------

PRE_CLEAN_AUDIT_REPORT = ARTIFACTS_PATH / "silver_audit.json"
GOLD_AUDIT_REPORT = ARTIFACTS_PATH / "gold_audit.json"
TRAIN_AUDIT_REPORT = ARTIFACTS_PATH / "train_audit.json"
MODEL_METRICS_REPORT = ARTIFACTS_PATH / "metrics.json"
DOMAIN_POLICY_PATH = ARTIFACTS_PATH / "domain_policy.json"
FEATURE_ENGG_STRATEGY_PATH = ARTIFACTS_PATH / "feature_engg_strategy.json"
CLUSTERING_IMAGE_PATH = ARTIFACTS_PATH / "clustering_viz.png"
UNIVERSAL_PERSONAS_PATH = ARTIFACTS_PATH / "universal_macro_personas.json"
PRODUCT_ANALYSIS_PATH = PRODUCT_REPORTS_PATH / "product_analysis.json"
CLUSTER_BASELINE_PATH = ARTIFACTS_PATH / "cluster_baseline.json"


def mean_pooling(model_output, attention_mask):
    """Averages token-level embeddings into a unified sentence vector"""
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def get_onnx_model():
    model_dir = current_dir / "onnx" / "sentence_transformer"

    print(f"📁 Verified local model assets target: {model_dir}")
    print("🖥️ Initializing local ONNX Runtime Engine on CPU...")

    # 1. Load the local string tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 2. Spin up the ONNX session session configuration
    ort_session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )

    return tokenizer, ort_session


def get_bedrock_client():
    import boto3
    from botocore.config import Config

    bedrock_config = Config(
        connect_timeout=60,       
        read_timeout=300,
        retries={
            "max_attempts": 3,
            "mode": "standard"
        }
    )

    return boto3.client("bedrock-runtime", region_name=AWS_REGION, config=bedrock_config)