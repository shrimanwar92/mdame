import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder 
import joblib

# Path alignment for shared constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import CLEANED_DATASET_PATH, GOLD_DATASET_PATH, GOLD_AUDIT_REPORT, JOBLIB_PIPELINE_PATH

def apply_feature_selection(X, selection_params, target_col=None):
    """
    Acts as a 'Quality Filter' for numeric features. 
    Removes static features (Low Variance) and redundant features (High Correlation).
    """
    # 1. Setup our parameters and VIP list
    force_keep = selection_params.get("force_keep", [])
    min_variance = selection_params.get("min_variance", 0.01)
    max_corr = selection_params.get("max_correlation", 0.85)
    
    # Identify numeric candidates, ensuring we don't accidentally drop the target if it's present
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if len(numeric_cols) <= 1:
        print("⚠️ Not enough numeric columns to perform selection.")
        return X

    # --- STAGE 1: THE LOW VARIANCE FILTER ---
    # Removes 'Dead' features that don't change enough to provide a signal.
    features_to_drop_variance = []
    for col in numeric_cols:
        variance = X[col].var()
        if variance < min_variance and col not in force_keep:
            features_to_drop_variance.append(col)
            print(f"🗑️ Removing {col}: Low Variance ({variance:.4f})")

    X = X.drop(columns=features_to_drop_variance)
    
    # --- STAGE 2: THE CORRELATION FILTER ---
    # Removes 'Copycat' features to prevent Multicollinearity.
    remaining_numeric = [c for c in numeric_cols if c in X.columns]
    corr_matrix = X[remaining_numeric].corr().abs()
    
    # Mask the diagonal so we don't compare a feature to itself
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop_corr = []
    for col in upper_triangle.columns:
        # Check if this column is too similar to ANY other column we've already seen
        is_redundant = any(upper_triangle[col] > max_corr)
        
        if is_redundant and col not in force_keep:
            # Find the 'Twin' for better logging
            matching_feat = upper_triangle.index[upper_triangle[col] > max_corr].tolist()[0]
            features_to_drop_corr.append(col)
            print(f"👯 Removing {col}: Redundant (Too similar to {matching_feat})")

    X = X.drop(columns=features_to_drop_corr)
    
    print(f"✅ Feature Selection Complete. Dropped {len(features_to_drop_variance) + len(features_to_drop_corr)} columns.")
    return X

def apply_categorical_encoding(pipe_cfg, X, y):
    encoder = None
    encoding_cfg = pipe_cfg.get("encoding", "one_hot")
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X_encoded_list = []
    
    if categorical_cols:
        print(f"📦 Encoding Categoricals ({encoding_cfg}): {categorical_cols}")
        if encoding_cfg == "target":
            encoder = TargetEncoder(cols=categorical_cols, handle_missing='value', handle_unknown='value')
            X_cat_encoded = encoder.fit_transform(X[categorical_cols], y)
            X_encoded_list.append(X_cat_encoded.reset_index(drop=True))
        else:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_arrays = encoder.fit_transform(X[categorical_cols])
            X_cat_encoded = pd.DataFrame(encoded_arrays, columns=encoder.get_feature_names_out(categorical_cols))
            X_encoded_list.append(X_cat_encoded.reset_index(drop=True))
    return encoder, categorical_cols, X_encoded_list

def run_gold_engineering_supervised():
    print("🚀 Phase 5: Gold Engineering (Supervised Pipeline Execution)")
    
    # 1. Load context and data
    df = pd.read_csv(CLEANED_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        plan = json.load(f)

    meta = plan["metadata"]
    landscape = plan["feature_landscape"]
    pipe_cfg = plan["transformation_pipeline"]
    
    target = meta["target"]
    subject_id = meta["subject_id"]
    drop_columns = meta.get("drop_columns", []) # New sanitization block
    
    base_features = landscape["base_features"]
    interactions = landscape["interaction_priorities"]
    temporal_cfg = landscape.get("temporal_engineering", []) # New temporal block
    
    # 2. Isolate Target and Initial Feature Set
    # We explicitly drop unique IDs and the target before starting
    y = df[target]
    X = df[base_features].copy()
    cols_to_purge = list(set([col for col in drop_columns if col in X.columns] + [subject_id] if subject_id in X.columns else []))
    X = X.drop(columns=cols_to_purge, errors='ignore')

    # --- STEP 3: TEMPORAL ENGINEERING ---
    # MUST happen before interactions so 'days_between_document_and_due' exists!
    for temp in temporal_cfg:
        col = temp["column"]
        if col in df.columns:
            temp_date = pd.to_datetime(df[col], errors='coerce')
            
            # Handle standard extractions
            for task in temp.get("extract", []):
                if task == "month":
                    X[f"{col}_month"] = temp_date.dt.month
                elif task == "day_of_week":
                    X[f"{col}_dow"] = temp_date.dt.dayofweek
                
                # Handle the Date Difference logic specifically
                elif task == "days_between_dates":
                    ref_col = temp.get("reference_column")
                    new_date_name = temp.get("name", f"diff_{col}")
                    if ref_col in df.columns:
                        ref_date = pd.to_datetime(df[ref_col], errors='coerce')
                        X[new_date_name] = (temp_date - ref_date).dt.days
            print(f"📅 Temporal features processed for: {col}")

    # --- STEP 4: INTERACTION LOGIC ---
    for inter in interactions:
        cols = inter.get("pair", [])
        new_name = inter["name"]
        
        # CASE A: UNARY (1 Column) - Handling things like clipping
        if len(cols) == 1:
            c1 = cols[0]
            if c1 in X.columns:
                if inter["logic"] == "outlier_clipping":
                    pct = inter.get("outlier_clipping_percentile", 99) / 100
                    X[new_name] = X[c1].clip(upper=X[c1].quantile(pct))
                    print(f"✨ Applied Unary Logic ({inter['logic']}): {new_name}")

        # CASE B: BINARY (2 Columns) - Handling multiplication, ratios, etc.
        elif len(cols) == 2:
            c1, c2 = cols[0], cols[1]
            if all(c in X.columns for c in [c1, c2]):
                if np.issubdtype(X[c1].dtype, np.number) and np.issubdtype(X[c2].dtype, np.number):
                    if inter["logic"] == "multiplication":
                        X[new_name] = X[c1] * X[c2]
                    elif inter["logic"] == "addition":
                        X[new_name] = X[c1] + X[c2]
                    elif inter["logic"] == "ratio":
                        denom = X[c2].replace(0, np.nan)
                        X[new_name] = X[c1] / (denom + 1e-6)
                    
                    # Optional: Clip binary results too if requested in JSON
                    if pipe_cfg.get("outlier_clipping"):
                        pct = inter.get("outlier_clipping_percentile", 99) / 100
                        X[new_name] = X[new_name].clip(upper=X[new_name].quantile(pct))
                    
                    print(f"✨ Applied Binary Logic ({inter['logic']}): {new_name}")

    # 5. Feature Selection (Numeric Only)
    X = apply_feature_selection(X, landscape["selection_params"], target_col=target)

    # 6. Categorical Encoding
    encoder, categorical_cols, X_encoded_list = apply_categorical_encoding(pipe_cfg, X, y)    

    # 7. Numeric Imputation & Scaling
    numeric_cols_final = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=pipe_cfg.get("imputation", "median"))
    X_num_imputed = pd.DataFrame(imputer.fit_transform(X[numeric_cols_final]), columns=numeric_cols_final)
    
    scaler_type = pipe_cfg.get("scaler", "PowerTransformer")
    scaler = PowerTransformer() if scaler_type == "PowerTransformer" else RobustScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num_imputed), columns=numeric_cols_final)

    # 8. Final Gold Export
    # We combine the scaled features, encoded features, and the target
    final_components = [X_num_scaled.reset_index(drop=True)] + X_encoded_list + [y.reset_index(drop=True)]
    final_df = pd.concat(final_components, axis=1)

    # # Re-insert the subject_id for auditing purposes (NOT as a feature)
    # if subject_id and subject_id in df.columns:
    #     final_df[subject_id] = df[subject_id].reset_index(drop=True)

    # Capture final feature names (excluding ID and Target) for XGBoost
    final_feature_columns = [col for col in final_df.columns if col not in [target, subject_id]]

    # 9. Save Artifacts
    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols_final,
        "final_features": final_feature_columns, 
        "target_col": target,
        "subject_id": subject_id,
        "interactions": interactions,
        "temporal_features": temporal_cfg
    }

    joblib.dump(artifacts, JOBLIB_PIPELINE_PATH)
    
    final_df.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"🏆 Production-Ready Gold Dataset Finalized: {GOLD_DATASET_PATH} | Shape: {final_df.shape}")

if __name__ == "__main__":
    run_gold_engineering_supervised()