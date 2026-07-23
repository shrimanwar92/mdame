import json
import os
import time
import sys
import boto3  
from dotenv import load_dotenv
import re

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_CLEAN_REPORT_PATH, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH, MODEL_ID, AWS_REGION

load_dotenv()

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

def generate_feature_engg_strategy(max_retries=5):
    """
    Consults AWS Bedrock to create a Semantic Feature Engineering Strategy
    with strict isolation rules for dense embedding blocks vs standard tabular inputs.
    """
    client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    
    if not os.path.exists(PROFILER_CLEAN_REPORT_PATH):
        raise FileNotFoundError(f"Cleaned report not found at {PROFILER_CLEAN_REPORT_PATH}")
    if not os.path.exists(DOMAIN_POLICY_PATH):
        raise FileNotFoundError(f"Domain policy configuration file not found at {DOMAIN_POLICY_PATH}")
    
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)

    variables = report.get('variables', {})
    raw_llm_domain = policy.get("domain", "Unknown")
    protected = policy.get("protected_features", [])
    
    normalized_domain, target_algorithm = normalize_domain_and_get_algorithm(raw_llm_domain)
    print(f"🔄 Raw Domain: '{raw_llm_domain}' -> Normalized: {normalized_domain}")
    print(f"🔒 Pipeline Locked Clustering Algorithm to: {target_algorithm}")

    # --- COMPACT VECTOR DETECTOR & SCHEMA STRIPPER ---
    # Group dense token columns (e.g., NAME_onnx_0...127) into a single logical block description
    # to prevent the LLM from trying to design interactions on single indexing rows.
    # --- COMPACT VECTOR DETECTOR & SCHEMA STRIPPER ---
    numeric_fields = []
    categorical_fields = []
    embedding_blocks_detected = set()

    for col, details in variables.items():
        # 1. Capture Embedding Blocks
        match = re.match(r"(.+)_onnx_\d+", col)
        if match:
            embedding_blocks_detected.add(f"{match.group(1)}_onnx_*")
        else:
            col_type = details.get('type', 'Unknown').lower()
            # 2. Separate Pure Numeric Tabular Features
            if any(t in col_type for t in ["int", "float", "number", "numeric"]):
                numeric_fields.append(f"- {col} ({details.get('type', 'Unknown')})")
            # 3. Separate Categorical/String/Boolean Features
            else:
                categorical_fields.append(f"- {col} ({details.get('type', 'Unknown')})")
            
    numeric_schema_str = "\n".join(numeric_fields) if numeric_fields else "- None"
    categorical_schema_str = "\n".join(categorical_fields) if categorical_fields else "- None"
    embedding_blocks_str = "\n".join([f"- {block}" for block in embedding_blocks_detected])

    detected_blocks_list = list(embedding_blocks_detected)
    example_wildcard = detected_blocks_list[0] if detected_blocks_list else "COLUMN_NAME_onnx_*"

    if normalized_domain == "COMMERCE" and len(detected_blocks_list) > 0:
        print(f"⚠️ Detected {len(detected_blocks_list)} embedding blocks in COMMERCE domain. Changing the target algorithm to GMM for better distribution handling.")
        target_algorithm = "GMM"
    
    system_instruction = (
        "You are a Senior Feature Engineer API. Your sole role is to construct a production feature "
        "engineering plan tailored to a specific data layout and assigned clustering algorithm. "
        "Output your choice strictly inside <json></json> tags."
    )
    
    prompt = f"""
    Act as a Senior Data Scientist and Domain Expert. 
    Analyze the schema and constraints to generate a Feature Engineering Strategy optimized for the specified clustering execution matrix.

    CORE BUSINESS TARGET DOMAIN: {normalized_domain}
    PROTECTED FEATURES (DO NOT UTILIZE FOR INTERACTIONS OR AGGREGATIONS): {json.dumps(protected)}
    
    NUMERIC TABULAR FIELDS (ELIGIBLE FOR LOGICAL INTERACTIONS):
    {numeric_schema_str}

    CATEGORICAL / TEXT TABULAR FIELDS (ONLY ELIGIBLE FOR ONE-HOT ENCODING):
    {categorical_schema_str}

    DETECTED HIGH-DIMENSIONAL EMBEDDING BLOCKS (WILDCARDS AVAILABLE):
    {embedding_blocks_str}

    🚨 MANDATORY PIPELINE ALGORITHM LOCK:
    The downstream system has locked processing framework to: {target_algorithm}
    
    EMBEDDING & DATATYPE STRUCTURAL RULES:
    1. Interaction pairs MUST consist exclusively of two DISTINCT fields chosen from the 'NUMERIC TABULAR FIELDS' list. They should not be same column.
    2. NEVER attempt 'ratio' or 'multiplication' math using a field from the 'CATEGORICAL / TEXT TABULAR FIELDS' list. Doing so causes runtime type crashes.
    3. NEVER select an individual embedding index variable (e.g., 'TEXT_onnx_0') for logical interaction pairs.
    4. NEVER select the same column twice in a single interaction pair. This breaks distribution math. 
    5. If an embedding block needs to be aggregated or scaled, treat the entire array block as a unit using wildcards (e.g., '{example_wildcard}').
    
    EMBEDDING ARRAY BLOCK RULES:
    1. NEVER select an individual embedding index variable (e.g., 'TEXT_onnx_0') for logical interaction pairs. Doing so breaks distribution math.
    2. Any interaction priorities MUST be built exclusively using standard tabular features (e.g., interacting 'STAR_RATING' with other base columns).
    3. If an embedding block needs to be aggregated or scaled, treat the entire array block as a unit using wildcards (e.g., '{example_wildcard}').
    4. For dimensionality reduction, indicate how to manage global PCA settings across these text blocks.

    Because the chosen engine is {target_algorithm}, you MUST tailor your recommendations to align with its architectural math profile:
    - If KMeans: Select parameters optimized for continuous, clear, variance-minimized boundaries.
    - If GMM: Focus heavily on scalers like StandardScaler. Build ratios/interactions that preserve distribution covariance.
    - If DBSCAN / HDBSCAN: Select RobustScaler if noise exists, and ensure interaction clipping supports spatial density stability.

    TASK:
    1. Define 'Logical Interactions': Create high-value behavioral ratios/multiplications based on the {normalized_domain} domain context. NEVER include protected fields or single embedding items.
    2. Preprocessing Policy: Select the absolute best scaling framework ('StandardScaler', 'PowerTransformer', or 'RobustScaler') tailored perfectly for {target_algorithm}.
    3. Aggregation Strategy: Detail structural data grouping criteria centered on the discovered subject_id. For embedding blocks, use the exact base wildcard name provided (e.g. '{example_wildcard}').
    4. Feature Selection: Set operational thresholds for maximum permissible correlation and minimum variance.

    OUTPUT JSON FORMAT REQUIREMENT:
    You MUST output "{target_algorithm}" in the "recommended_algorithm" key. Do not substitute this parameter.

    <json>
    {{
        "interaction_priorities": [
            {{
                "pair": ["col1", "col2"], 
                "logic": "multiplication" | "ratio", 
                "name": "meaningful_name",
                "outlier_clipping_percentile": 99,
                "error_handling": "zero_fill"
            }}
        ],
        "preprocessing_metadata": {{
            "recommended_scaler": "StandardScaler" | "PowerTransformer" | "RobustScaler",
            "imputation_strategy": "median" | "mean",
            "categorical_encoding": "one-hot",
            "dimensionality_reduction_recommendation": {{
                "apply_global_pca": true | false,
                "variance_retention_threshold": 0.95,
                "target_embedding_blocks": {json.dumps(detected_blocks_list)}
            }}
        }},
        "subject_id": "string_or_null",
        "aggregation_strategy": {{
            "columns_to_sum": [],
            "columns_to_mean": {json.dumps(detected_blocks_list)},
            "columns_to_mode": []
        }},
        "feature_selection": {{
            "max_correlation_threshold": 0.85,
            "variance_threshold": 0.01
        }},
        "recommended_algorithm": "{target_algorithm}"
    }}
    </json>
    """

    for attempt in range(max_retries):
        try:
            print(f"🧠 Generating Feature Strategy via Bedrock (Attempt {attempt + 1}/{max_retries})...")
            
            response = client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system=[{"text": system_instruction}],
                inferenceConfig={"temperature": 0.0}
            )

            response_text = response['output']['message']['content'][0]['text']
            json_match = re.search(r'<json>(.*?)</json>', response_text, re.DOTALL)
            
            if not json_match:
                raise ValueError("Could not find structured <json></json> syntax markers in inference output.")
                
            raw_json_string = json_match.group(1).strip()
            strategy = json.loads(raw_json_string)
            
            with open(FEATURE_ENGG_STRATEGY_PATH, 'w') as f:
                json.dump(strategy, f, indent=4)
            
            print(f"✅ Structural Feature Strategy Compiled for {normalized_domain} using {target_algorithm}.")
            return strategy

        except Exception as e:
            if any(err in str(e) for err in ["ThrottlingException", "LimitExceededException", "429"]):
                wait_time = (2 ** attempt) + 5 
                print(f"⚠️ Scale limit reached. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Execution Failure: {e}")
                if attempt == max_retries - 1:
                    raise e

    raise Exception("Critical: Feature compilation processing timed out permanently via AWS Bedrock endpoints.")

if __name__ == "__main__":
    generate_feature_engg_strategy()