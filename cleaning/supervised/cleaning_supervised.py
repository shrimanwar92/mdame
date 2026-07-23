import pandas as pd
import json
import os
import sys
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.selection import DropFeatures

# Import shared constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import (DATASET_PATH, PRE_CLEAN_AUDIT_REPORT, 
                       CLEANED_DATASET_PATH, DOMAIN_POLICY_PATH)

def handle_duplicates(df, subject_id):
    """Uses the Policy Subject ID for intelligent deduplication."""
    initial_len = len(df)
    if subject_id and subject_id in df.columns:
        df = df.drop_duplicates(subset=[subject_id]).reset_index(drop=True)
        print(f"  ✨ Removed {initial_len - len(df)} duplicates based on {subject_id}.")
    else:
        df = df.drop_duplicates().reset_index(drop=True)
    return df

def autonomous_cleaning():
    print("\n🛠️ Phase 3: Executing Policy-Driven Cleaning...")
    
    # 1. Load Policy and Audit Reports
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)
    with open(PRE_CLEAN_AUDIT_REPORT, 'r') as f:
        audit = json.load(f)
        
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Define Total Drop List
    # Combine Policy Garbage + Audit Weak Features[cite: 7, 8]
    to_drop = list(set(policy['technical_garbage'] + audit['weak_features']))
    
    # Ensure Subject ID is not in training features but kept for Silver Layer index
    training_candidates = [c for c in policy['candidate_features'] if c not in to_drop]
    
    num_vars = df[training_candidates].select_dtypes(include=['number']).columns.tolist()
    cat_vars = df[training_candidates].select_dtypes(include=['object']).columns.tolist()

    # 3. Build the Cleaning Pipeline
    steps = [('policy_drop', DropFeatures(features_to_drop=to_drop))]
    
    if num_vars:
        steps.append(('num_impute', MeanMedianImputer(variables=num_vars)))
    if cat_vars:
        steps.append(('cat_impute', CategoricalImputer(variables=cat_vars, ignore_format=True)))

    # Execute Transformation
    cleaning_pipeline = Pipeline(steps)
    df_silver = cleaning_pipeline.fit_transform(df)

    # 4. Apply Final Integrity Logic
    df_silver = handle_duplicates(df_silver, policy['subject_id'])
    
    # 5. Export Silver Layer
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df_silver.to_csv(CLEANED_DATASET_PATH, index=False)
    
    print(f"✅ Silver Layer Created for {policy['domain']} domain.")
    print(f"📊 Features Retained: {len(training_candidates)} | Garbage Dropped: {len(to_drop)}")
    return df_silver

if __name__ == "__main__":
    autonomous_cleaning()