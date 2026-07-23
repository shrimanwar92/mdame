import json
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_FEATURE_ENGG_REPORT_PATH, GOLD_AUDIT_REPORT, TRAIN_AUDIT_REPORT

def generate_train_audit_config(profiler_report_dict, gold_audit):
    """
    Parses the profile report metadata to dynamically route features 
    based on cardinality and data types.
    """
    training_features = []
    omitted_features = []
    subject_id, recommended_algo, use_parquet = gold_audit["subject_id"], gold_audit["recommended_algorithm"], gold_audit["use_parquet"]
    variables = profiler_report_dict.get('variables', {})
    
    # Process each column block from the profiler report
    for col_name, metadata in variables.items():
        # Safeguard: Skip structural IDs or target columns if they exist
        if col_name in [subject_id, 'Cluster', 'Certainty']:
            continue
            
        n_distinct = metadata.get("n_distinct", 0)
        col_type = metadata.get("type", "")
        
        # Rule 1: Binary Flags -> Profiling Only
        if n_distinct <= 2:
            omitted_features.append(col_name)
            
        # Rule 2: High Cardinality / Numeric Ordinals -> Active Training Space
        elif col_type == "Numeric" and n_distinct > 2:
            training_features.append(col_name)
            
        # Fallback: Anything else (like raw unstructured text columns)
        else:
            omitted_features.append(col_name)

    # Construct the decoupled Audit JSON schema
    train_audit_schema = {
        "subject_id": subject_id,
        "use_parquet": use_parquet,
        "recommended_algorithm": recommended_algo,
        "training_features": training_features,
        "omitted_features": omitted_features
    }
    
    # Overwrite the stale config file cleanly
    with open(TRAIN_AUDIT_REPORT, 'w') as f:
        json.dump(train_audit_schema, f, indent=4)
        
    print(f"✅ Generated Train Audit configuration with {len(training_features)} training fields.")
    return train_audit_schema

if __name__ == "__main__":
    # Example usage: Load a profiler report and generate the audit config
    with open(PROFILER_FEATURE_ENGG_REPORT_PATH, 'r') as f:
        profiler_report = json.load(f)

    with open(GOLD_AUDIT_REPORT, 'r') as f:
        gold_audit = json.load(f)
    
    generate_train_audit_config(profiler_report, gold_audit)