import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_CLEAN_REPORT_PATH, GOLD_AUDIT_REPORT, DOMAIN_POLICY_PATH

def estimate_metadata_outliers(stats: dict) -> float:
    min_val = stats.get('min', 0.0)
    max_val = stats.get('max', 0.0)
    iqr = stats.get('iqr', 0.0)
    
    p5 = stats.get('5%', min_val)
    p25 = stats.get('25%', min_val)
    p75 = stats.get('75%', max_val)
    p95 = stats.get('95%', max_val)
    
    # Standard Tukey IQR outlier fences
    lower_fence = p25 - (1.5 * iqr)
    upper_fence = p75 + (1.5 * iqr)
    
    p_lower_outliers = 0.0
    p_upper_outliers = 0.0
    
    # 1. Lower Tail Estimation (Looking for values below the lower fence)
    if lower_fence > min_val:
        if lower_fence <= p5:
            # Outliers are between min_val and lower_fence (inside the 0% to 5% bucket)
            # The percentage depends on where the fence sits relative to min and p5
            denom = p5 - min_val
            p_lower_outliers = 0.05 * (1.0 - ((lower_fence - min_val) / denom)) if denom > 0 else 0.05
        elif lower_fence < p25:
            # Outliers take up all of the 0-5% bucket, plus some of the 5-25% bucket
            denom = p25 - p5
            p_lower_outliers = 0.05 + 0.20 * (1.0 - ((lower_fence - p5) / denom)) if denom > 0 else 0.25

    # 2. Upper Tail Estimation (Looking for values above the upper fence)
    if upper_fence < max_val:
        if upper_fence >= p95:
            # Outliers are between upper_fence and max_val (inside the 95% to 100% bucket)
            denom = max_val - p95
            p_upper_outliers = 0.05 * ((max_val - upper_fence) / denom) if denom > 0 else 0.05
        elif upper_fence > p75:
            # Outliers take up all of the 95-100% bucket, plus some of the 75-95% bucket
            denom = p95 - p75
            p_upper_outliers = 0.05 + 0.20 * ((p95 - upper_fence) / denom) if denom > 0 else 0.25
            
    return p_lower_outliers + p_upper_outliers

def normalize_domain_and_get_algorithm(raw_domain_str: str) -> tuple[str, str]:
    """
    Normalizes fuzzy LLM strings into structural domains and locks the matching algorithm.
    """
    domain_clean = raw_domain_str.lower().strip()
    
    if any(k in domain_clean for k in ["credit card", "financ", "bank", "wealth", "invest"]):
        return "FINANCE", "GMM"
    elif any(k in domain_clean for k in ["traffic", "network", "geospatial", "telecom", "routing", "transit"]):
        return "INFRASTRUCTURE_NODES", "HDBSCAN"
    elif any(k in domain_clean for k in ["fraud", "cyber", "security", "anomaly", "intrusion"]):
        return "SECURITY_ANOMALIES", "DBSCAN"
    elif any(k in domain_clean for k in ["ecommerce", "retail", "marketing", "churn", "sales", "shop", "e-commerce", "e-comm", "ecomm", "product"]):
        return "COMMERCE", "KMeans"
    else:
        return "GENERAL_BEHAVIORAL", "GMM"

def run_gold_audit():
    print("🧠 Phase 4: Gold Audit (Deterministic Feature Engineering Audit.)")

    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)

    variables = report.get('variables', {})
    protected = policy.get("protected_features", [])
    subject_id = policy.get("subject_id")
    sentiment = policy.get("sentiment", {})

    # 1. Resolve Target Domain and Clustering Algorithm Rules
    raw_llm_domain = policy.get("domain", "Unknown")
    normalized_domain, target_algorithm = normalize_domain_and_get_algorithm(raw_llm_domain)
    
    print(f"🔄 Raw Domain: '{raw_llm_domain}' -> Normalized: {normalized_domain}")

    # 2. Extract and Categorize Plan Items
    longtext_columns = []
    
    gold_plan = {
        "subject_id": subject_id,
        "recommended_algorithm": target_algorithm,
        "encoding": {
            "one_hot": [],
            "rare_label": [],
            "binary_cast": []  # New: Direct 0/1 mapping for clean Booleans
        },
        "transformations": {
            "yeo_johnson": []
        },
        "scaling": {
            "robust": [],
            "standard": [],
            "min_max": []
        },
        "text_embeddings": {
            "embed_columns": []
        },
        "use_parquet": False,
        "sentiment": sentiment
    }

    for col, stats in variables.items():
        if col == subject_id:
            continue
            
        col_type = stats.get('type')

        # 1. Optimized Numeric Engine (Aligned with Scaler Decision Tree)
        if col_type == 'Numeric':
            n_distinct = stats.get('n_distinct', 0)
            skewness = stats.get('skewness', 0.0)
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 0.0)
            
            # [Flowchart Node]: Execute Programmatic Outlier Calculation
            p_outliers = estimate_metadata_outliers(stats)
            abs_skew = abs(skewness)

            # -----------------------------------------------------------------
            # FLOWCHART BRANCH 1: GUARD RAIL A (Bounded Scale Verification)
            # -----------------------------------------------------------------
            if n_distinct <= 5 and min_val >= 0 and max_val <= 10:
                print(f"🎯 Flowchart Path -> [Bounded Ordinal Scale] -> MinMax Scaling for '{col}'")
                gold_plan["scaling"]["min_max"].append(col)
                continue

            # -----------------------------------------------------------------
            # FLOWCHART BRANCH 2: GUARD RAIL B (Directional Mathematical Transforms)
            # -----------------------------------------------------------------
            # Check for any severe asymmetry (Positive or Negative skew)
            if abs(skewness) > 0.75:
                print(f"🔄 Flowchart Path -> [Severe Skewness ({skewness:.2f})] -> Registering Yeo-Johnson for '{col}'")
                gold_plan["transformations"]["yeo_johnson"].append(col)
            else:
                print(f"🧼 Flowchart Path -> [Symmetrical Distribution ({skewness:.2f})] -> No transformation needed for '{col}'")
            
            # -----------------------------------------------------------------
            # FLOWCHART BRANCH 3: SCALER ROUTING DECISION TREE
            # -----------------------------------------------------------------
            
            # Check Node A: Massive Outlier Footprint -> High Risk
            if p_outliers > 0.05:
                print(f"🚨 Flowchart Path -> [Outliers > 5% ({p_outliers*100:.1f}%)] -> Locking RobustScaler for '{col}'")
                gold_plan["scaling"]["robust"].append(col)
                
            # Check Node B: Severe Asymmetry -> Means/Variances are distorted
            elif abs_skew > 1.0:
                print(f"📈 Flowchart Path -> [Severe Skewness ({skewness:.2f})] -> Locking RobustScaler for '{col}'")
                gold_plan["scaling"]["robust"].append(col)
            
            # Check Node C: Moderate Outliers but low skew -> Still risky for Standard
            elif p_outliers > 0.01:
                print(f"⚠️ Flowchart Path -> [Moderate Outlier Footprint ({p_outliers*100:.1f}%)] -> Deploying RobustScaler for '{col}'")
                gold_plan["scaling"]["robust"].append(col)
                
            # Check Node D: Safe Zone -> Low skew, minimal/zero outliers
            else:
                print(f"🧼 Flowchart Path -> [Clean/Safe Profile] -> Defaulting to StandardScaler for '{col}'")
                gold_plan["scaling"]["standard"].append(col)

        # 2. Optimized Boolean Engine
        elif col_type == 'Boolean':
            # Stop the double-column one-hot expansion; lock it to a single binary track
            gold_plan["encoding"]["binary_cast"].append(col)

        # 3. Categorical Text Engine
        elif col_type in ['Text', 'Categorical']:
            n_distinct = stats.get('n_distinct', 0)
            
            # 1. Catch open-ended text or unique IDs early
            if n_distinct > 30:
                print(f"⚠️ Excluding high-cardinality open text feature '{col}' ({n_distinct} unique values).")
                continue
                
            # 2. If it has between 11 and 30 categories, it NEEDS Rare Label processing FIRST
            if n_distinct > 10:
                print(f"🎭 Registering Rare Label consolidation for '{col}' ({n_distinct} categories)")
                gold_plan["encoding"]["rare_label"].append(col)
                
            # 3. EVERY categorical column that survives must ultimately be One-Hot Encoded
            print(f"🔢 Registering One-Hot Encoding for final output of '{col}'")
            gold_plan["encoding"]["one_hot"].append(col)

        # Track LongText Columns requiring Embedding Block mapping + PCA
        elif col_type == 'LongText':
            longtext_columns.append(col)
            gold_plan["text_embeddings"]["embed_columns"].append(col)

    # 3. Handle Fallback Overrides for Embedding Blocks
    if len(longtext_columns) > 0:
        gold_plan["use_parquet"] = True
        
        if normalized_domain == "COMMERCE":
            print(f"⚠️ Detected {len(longtext_columns)} longtext column(s) in COMMERCE domain. Overriding target algorithm to GMM for better distribution handling.")
            gold_plan["recommended_algorithm"] = "GMM"

    print(f"🔒 Pipeline Locked Clustering Algorithm to: {gold_plan['recommended_algorithm']}")

    with open(GOLD_AUDIT_REPORT, 'w') as f:
        json.dump(gold_plan, f, indent=4)
        
    print(f"✅ Gold Audit Plan generated successfully.")


if __name__ == "__main__":
    run_gold_audit()