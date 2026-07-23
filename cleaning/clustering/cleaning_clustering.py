import json
import os
import sys
import joblib

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_engine.selection import DropFeatures
from feature_engine.imputation import (
    MeanMedianImputer,
    CategoricalImputer,
)

# ---------------------------------------------------------------------
# Repository Context
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from constants import (
    DATASET_PATH,
    PRE_CLEAN_AUDIT_REPORT,
    CLEANED_DATASET_PATH,
    CLEANING_PIPELINE_PATH,
    AUTO_INC_ID,
    BOOLEAN_MAPPING,
)

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def clean_categorical_columns(X, columns):

    df = X.copy()

    for col in columns:

        if col not in df.columns:
            continue

        mask = df[col].notna()

        df.loc[mask, col] = (
            df.loc[mask, col]
            .astype(str)
            .str.strip()
        )

        df[col] = df[col].replace("", np.nan)

    return df


def normalize_boolean_columns(X, columns):

    df = X.copy()

    for col in columns:

        if col not in df.columns:
            continue

        mask = df[col].notna()

        df.loc[mask, col] = (
            df.loc[mask, col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(BOOLEAN_MAPPING)
        )

    return df


# ---------------------------------------------------------------------
# Cleaning Engine
# ---------------------------------------------------------------------

def run_clustering_cleaning():

    print("⚙️ Phase 3: Cleaning Engine")

    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    with open(PRE_CLEAN_AUDIT_REPORT) as f:
        contract = json.load(f)

    # ----------------------------------------------------------
    # Remove duplicates
    # ----------------------------------------------------------

    duplicate_cfg = contract.get("duplicates", {})

    if duplicate_cfg.get("enabled", False):
        before = len(df)

        df = (df.drop_duplicates(keep=duplicate_cfg.get("keep", "first")).reset_index(drop=True))
        print(f"🧹 Removed {before-len(df)} duplicates")

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    imputation_cfg = contract.get("imputation", {})

    mean_cols = [
        c for c in imputation_cfg.get("mean", [])
        if c in df.columns
    ]

    median_cols = [
        c for c in imputation_cfg.get("median", [])
        if c in df.columns
    ]

    categorical_cols = [
        c for c in imputation_cfg.get("categorical_mode", [])
        if c in df.columns
    ]

    boolean_cols = [
        c for c in imputation_cfg.get("boolean_mode", [])
        if c in df.columns
    ]

    drop_cols = [
        c for c in contract.get("drop_features", [])
        if c in df.columns
    ]

    # ----------------------------------------------------------
    # Build Pipeline
    # ----------------------------------------------------------

    steps = []

    if drop_cols:
        steps.append(("drop_features", DropFeatures(features_to_drop=drop_cols)))

    if categorical_cols:
        steps.append(("clean_text", FunctionTransformer(
            clean_categorical_columns,
            kw_args={
                "columns": categorical_cols
            },
            validate=False
        )))

    if boolean_cols:
        steps.append(("normalize_booleans", FunctionTransformer(
            normalize_boolean_columns,
            kw_args={
                "columns": boolean_cols
            },
            validate=False
        )))

    if categorical_cols:
        steps.append(("categorical_mode", CategoricalImputer(
            imputation_method="frequent",
            variables=categorical_cols
        )))

    if boolean_cols:
        steps.append(("boolean_mode", CategoricalImputer(
            imputation_method="frequent",
            variables=boolean_cols
        )))

    if mean_cols:
        steps.append(("mean_imputer", MeanMedianImputer(
            imputation_method="mean",
            variables=mean_cols
        )))

    if median_cols:
        steps.append(("median_imputer", MeanMedianImputer(
            imputation_method="median",
            variables=median_cols
        )))

    cleaning_pipeline = Pipeline(steps)
    cleaning_pipeline.fit(df)

    joblib.dump(cleaning_pipeline, CLEANING_PIPELINE_PATH, compress=3)
    print(f"💾 Cleaning pipeline saved. {CLEANING_PIPELINE_PATH}")

    # ----------------------------------------------------------
    # Transform
    # ----------------------------------------------------------

    df = cleaning_pipeline.transform(df)
    df = df.reset_index(drop=True)
    df.index.name = AUTO_INC_ID

    subject_id = contract.get("subject_id")

    if subject_id and subject_id in df.columns:
        print(f"⚓ Subject ID preserved: {subject_id}")

    df.to_csv(CLEANED_DATASET_PATH, index=True)
    print( f"✅ Silver Dataset saved. Shape: {df.shape}")


if __name__ == "__main__":
    run_clustering_cleaning()