import json
import os
import sys

# Maintain relative pathing for shared constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_CLEAN_REPORT_PATH, GOLD_AUDIT_REPORT, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH

def run_gold_audit_supervised():
    print("🧠 Phase 4: Supervised Gold Audit (Handling Sanitization & Temporal Intelligence)")
    
    # Load context files
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)
    with open(FEATURE_ENGG_STRATEGY_PATH, 'r') as f:
        strategy = json.load(f)

    target = policy.get("target_variable")
    subject_id = policy.get("subject_id")
    
    # 1. HANDLE SANITIZATION (The "Anti-Explosion" Logic)
    sanitization = strategy.get("sanitization", {})
    llm_suggested_drops = sanitization.get("drop_columns", [])
    
    # Define columns that MUST be dropped from training features
    # We include the LLM's findings (e.g., Invoice_ID) and the Policy's subject_id
    drop_list = list(set(llm_suggested_drops + [target, subject_id]))
    
    available_columns = list(report.get("variables", {}).keys())
    
    # 2. HANDLE TEMPORAL ENGINEERING
    # Identify which columns are being turned into features so we don't encode the raw strings
    temporal_cfg = strategy.get("temporal_engineering", [])
    temporal_source_cols = [item["column"] for item in temporal_cfg]
    
    # Base features: Exclude drops AND raw date strings that will be engineered
    base_features = [
        col for col in available_columns 
        if col not in drop_list and col not in temporal_source_cols
    ]

    # 3. RESOLVE TECHNICAL STRATEGY
    prep_meta = strategy.get("preprocessing_metadata", {})
    f_selection = strategy.get("feature_selection", {})
    interactions = strategy.get("interaction_priorities", [])
    algorithm_selection = strategy.get("algorithm_selection", {})
    
    # Precise Encoding Logic: Use Strategy's method (e.g., 'target') or fallback
    encoding_strategy = prep_meta.get("encoding_strategy")

    # Interaction names must be kept to avoid pruning
    force_keep = [inter["name"] for inter in interactions]

    # 4. CONSTRUCT THE ENHANCED GOLD PLAN
    gold_plan = {
        "metadata": {
            "domain": policy.get("domain"),
            "target": target,
            "subject_id": subject_id,
            "drop_columns": drop_list  # Explicitly tell Phase 5 what to drop
        },
        "feature_landscape": {
            "base_features": base_features,
            "temporal_engineering": temporal_cfg, # New block for Phase 5
            "interaction_priorities": interactions,
            "selection_params": {
                "max_correlation": f_selection.get("max_correlation", 0.85),
                "min_variance": f_selection.get("min_variance", 0.01),
                "force_keep": force_keep
            }
        },
        "transformation_pipeline": {
            "scaler": prep_meta.get("recommended_scaler", "PowerTransformer"),
            "imputation": prep_meta.get("imputation_strategy", "median"),
            "encoding": encoding_strategy,
            "outlier_clipping": True 
        },
        "algorithm_selection": algorithm_selection
    }

    # 5. FINAL WRITE-BACK
    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(gold_plan, f, indent=4)
        
    print(f"✅ Supervised Gold Plan finalized.")
    print(f"   - 🚫 Dropping (IDs/Target): {drop_list}")
    print(f"   - 📅 Temporal Features: {temporal_source_cols}")
    print(f"   - 🛠️ Encoding Method: {encoding_strategy}")

if __name__ == "__main__":
    run_gold_audit_supervised()