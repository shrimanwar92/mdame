import json
import os
import time
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_CLEAN_REPORT_PATH, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH

load_dotenv()

def generate_feature_engg_strategy(max_retries=5):
    api_key = os.getenv("MDAME_API_KEY")
    client = genai.Client(api_key=api_key)
    
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)

    alerts = report.get('alerts', [])
    variables = report.get('variables', {})
    schema_str = "\n".join([f"- {col} ({details.get('type')})" for col, details in variables.items()])
    
    target = policy.get("target_variable")
    domain = policy.get("domain")

    system_instruction = "You are a Domain-Agnostic Feature Engineering API and senior mlops architect. Output ONLY raw JSON."
    
    prompt = f"""
    Analyze the data profile for a Supervised Learning task in the {domain} domain.
    TARGET: {target}
    ALERTS: {json.dumps(alerts)}
    SCHEMA: {schema_str}

    TASK 0: DATA SANITIZATION & DROPPING
    1. IDENTIFY UNIQUE IDs: If an alert indicates "[Column] has unique values", flag it for removal if it acts as a Primary Key (e.g., Invoice_ID, Customer_ID) to prevent overfitting.
    2. TEMPORAL DATA: Identify all DATE/DATETIME columns. These MUST NOT be encoded. Instead, recommend extracting "day_of_week", "month", or "days_until_due".

    TASK 1: FEATURE ENGINEERING
    Identify universal data patterns to generate 'interaction_priorities':
    1. SPARSE SUMMATION: If multiple columns have high 'zero' alerts, suggest an 'addition' logic to create a cumulative signal.
    2. MAGNITUDE RATIOS: Identify pairs where col1/col2 represents a rate, percentage, or efficiency (e.g., cost per unit, success rate).
    3. SKEW MITIGATION: For columns with outlier alerts, specify 'outlier_clipping_percentile'.
    4. CROSS-PRODUCT SIGNALS: Suggest 'multiplication' for features that amplify each other's impact on the {target}.
    5. ENCODING STRATEGY: 
       - NEVER suggest encoding for Unique IDs or Date columns.
       - Suggest 'target' encoding for high-cardinality categorical features (>15 levels).
       - Suggest 'one_hot' only for low-cardinality (<10 levels).

    TASK 2: ALGORITHM SELECTION [NEW]
    Based on the Target Type and Alerts:
    1. Determine if this is REGRESSION or CLASSIFICATION.
    2. Suggest the best algorithm (e.g., XGBoost for tabular complexity, LightGBM for large/sparse data, Random Forest for robust baselines, or Logistic/Linear Regression for interpretability).
    3. Justify the choice based on the ALERTS (e.g., "XGBoost chosen due to high skew and non-linear relationships").

    OUTPUT SCHEMA:
    {{
        "sanitization": {{
            "drop_columns": ["col1", "col2"],
            "reasoning": "string"
        }},
        "temporal_engineering": [
            {{ "column": "date_col", "extract": ["month", "day_of_week", "diff_days"] }}
        ],
        "interaction_priorities": [
            {{ 
                "pair": ["col1", "col2"], 
                "logic": "ratio" | "multiplication" | "addition", 
                "name": "descriptive_feature_name", 
                "outlier_clipping_percentile": 99, 
                "error_handling": "zero_fill" 
            }}
        ],
        "preprocessing_metadata": {{
            "recommended_scaler": "PowerTransformer" | "RobustScaler",
            "imputation_strategy": "median" | "mean",
            "encoding_strategy": "one_hot" | "target"
        }},
        "algorithm_selection": {{
            "problem_type": "regression" | "classification",
            "algorithm": "string",
            "justification": "string",
            "base_hyperparameters": {{
                "n_estimators": 100,
                "learning_rate": 0.1
            }}
        }},
        "feature_selection": {{
            "max_correlation": 0.85,
            "min_variance": 0.01
        }}
    }}
    """

    for attempt in range(max_retries):
        print(f"Attempt: {attempt}")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1
                ),
                contents=prompt
            )
            strategy = json.loads(response.text)
            with open(FEATURE_ENGG_STRATEGY_PATH, 'w') as f:
                json.dump(strategy, f, indent=4)
            print(f"✅ Generalized Strategy generated for {domain}.")
            return strategy
        except Exception as e:
            print(e)
            time.sleep(2 ** attempt)

if __name__ == "__main__":
    generate_feature_engg_strategy()