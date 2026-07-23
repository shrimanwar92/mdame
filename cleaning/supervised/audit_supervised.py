import pandas as pd
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
import json
import os
import sys
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates, FeatureLabelCorrelation

# Import shared constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import DATASET_PATH, TARGET_COLUMN, PRE_CLEAN_AUDIT_REPORT, DOMAIN_POLICY_PATH

def run_pre_clean_audit():
    print("⚖️ Phase 2: Auditing Signal Integrity via Domain Policy...")
    
    # 1. Load Domain Policy
    if not os.path.exists(DOMAIN_POLICY_PATH):
        raise FileNotFoundError("Domain Policy JSON not found. Run Architect script first.")
    
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)
    
    df = pd.read_csv(DATASET_PATH)
    target = policy['target_variable']
    candidates = policy['candidate_features']
    
    # 2. Scope the Audit to Candidate Features + Target
    audit_df = df[candidates + [target]]
    
    # Identify categorical features for Deepchecks
    cat_features = audit_df.select_dtypes(include=['object']).columns.tolist()
    if target in cat_features: cat_features.remove(target)
    
    # 3. Perform Deepchecks Audit
    ds = Dataset(audit_df, label=target, cat_features=cat_features)
    pps_check = FeatureLabelCorrelation().run(ds)
    dup_check = DataDuplicates().run(ds)
    
    # Dynamic Thresholding based on candidate count
    pps_threshold = 0.005 if len(candidates) < 10 else 0.02
    
    # Identify features that failed the signal test
    weak_features = [f for f, pps in pps_check.value.items() if pps < pps_threshold]

    # 4. Compile Audit Results
    audit_results = {
        "domain": policy['domain'],
        "config": {
            "pps_threshold": pps_threshold,
            "vif_threshold": 10.0 # Default VIF
        },
        "weak_features": weak_features,
        "duplicate_ratio": float(dup_check.value),
        "policy_id_cols": [policy['subject_id']] if policy['subject_id'] else []
    }

    with open(PRE_CLEAN_AUDIT_REPORT, 'w') as f:
        json.dump(audit_results, f, indent=4)
    
    print(f"✅ Audit complete. Found {len(weak_features)} weak features in {policy['domain']} domain.")
    return audit_results

if __name__ == "__main__":
    run_pre_clean_audit()