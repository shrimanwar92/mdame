import json
import os
import time
import sys
import boto3  # Swapped from google-genai
from dotenv import load_dotenv
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_REPORT_PATH, DOMAIN_POLICY_PATH, AWS_REGION, MODEL_ID

load_dotenv()

ROLE_TO_BUCKET = {
    "behavior": "protected",
    "context": "protected",
    "temporal": "protected",
    "identifier": "technical_garbage",
    "metadata": "technical_garbage",
    "unsupported": "technical_garbage",
}

def get_autonomous_domain_policy(max_retries=5):
    # Initialize AWS Bedrock Runtime Client
    # It will automatically pick up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, 
    # and AWS_REGION from your environment or .env file.
    client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    
    with open(PROFILER_REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    vars_summary = report.get('variables', {})
    alerts = report.get('alerts', [])
    dataset_description = "This dataset belongs to amazon reviews dataset"

    system_instruction = "You are an Autonomous Data Architect. Output your final decision inside <json></json> tags."
    
    prompt = f"""
You are an Autonomous Machine Learning Data Architect.

Your task is to understand an unknown dataset and generate the semantic policy
required by an autonomous customer behavior clustering pipeline.

You have three sources of information:

1. Dataset description
2. Dataset profiling report
3. Dataset quality alerts

Use all available information to determine the most accurate semantic
understanding of the dataset.

The dataset description provides the business context.

The profiler validates the schema, sample values and statistics.

The alerts provide information about data quality.

Use your own reasoning.

Do not explain your reasoning.

Do not describe your thought process.

Return only the requested JSON inside <json></json> tags.



========================================================
DATASET DESCRIPTION
========================================================

{dataset_description}



========================================================
PROFILER REPORT
========================================================

{json.dumps(vars_summary)}



========================================================
DATA QUALITY ALERTS
========================================================

{json.dumps(alerts)}



========================================================
YOUR TASK
========================================================

Generate a semantic policy for an autonomous customer behavior clustering
pipeline.

Determine the following:



1. domain

Identify the business domain.

Examples

e-commerce

healthcare

finance

insurance

telecommunications

manufacturing

education

etc.



--------------------------------------------------------



2. subject_id

Identify the identifier representing the primary entity being analyzed.

The subject_id should identify the business object whose behaviour is being
clustered.

Examples

Customer_ID

Product_ID

Patient_ID

Device_ID

Employee_ID

Ticket_ID

Order_ID


If no suitable identifier exists, return null.



--------------------------------------------------------



3. protected_features

Return every column containing meaningful business information that should be
preserved for feature engineering.

These columns are NOT required to be numeric.

These columns may later undergo

• embeddings

• categorical encoding

• datetime feature extraction

• numerical scaling

• normalization

• feature engineering

before clustering.

Typical protected features include

• review text

• summaries

• ratings

• categories

• product names

• brands

• timestamps

• date

• datetime

• numerical measurements

Keep any column containing useful behavioural, contextual or temporal
information.

Do NOT discard a column simply because it is text.



--------------------------------------------------------



4. technical_garbage

Return columns that should NOT participate in clustering.

These typically include

• identifiers not representing behaviour

• UUIDs

• hashes

• import metadata

• acquisition metadata

• review source

• pipeline metadata

• duplicated identifiers

• unsupported columns

• constant columns

• empty columns

Only place a column here if it contributes no meaningful behavioural,
contextual or temporal information.



--------------------------------------------------------



5. sentiment

Identify the best available structured representation of customer sentiment.

Prefer the following order.

1. Existing sentiment labels

Examples

Positive

Negative

Neutral



2. Numerical rating

Examples

1–5 stars

NPS

CSAT

Likert scales

If a reliable rating exists, prefer it over free text because it is already an
explicit customer sentiment signal.



3. Continuous sentiment score

Examples

0.91

-0.62



4. Free-text feedback

Only choose this when no structured sentiment signal exists.

Use

"strategy": "classifier"

and return the best text column.



5. None

If customer sentiment cannot reasonably be inferred.



========================================================
OUTPUT
========================================================

Return ONLY

<json>

{{
    "domain": "",

    "subject_id": "",

    "columns": {{
        "COLUMN_NAME": {{
            "role": "behavior | context | temporal | identifier | metadata | unsupported"
        }}
    }},

    "protected_features": [],

    "technical_garbage": [],

    "sentiment": {{

        "strategy": "",

        "source_column": "",

        "confidence": 0.0

    }}

}}

</json>

Return no additional text.
    """

    for attempt in range(max_retries):
        try:
            print(f"🧠 AWS Bedrock Pipeline: Identifying Domain & Filtering Signal...")
            
            # Utilizing the modern Bedrock Converse API
            response = client.converse(
                modelId=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                system=[
                    {"text": system_instruction}
                ],
                inferenceConfig={
                    "temperature": 0.1
                }
            )
            
            # Extract the raw text from the Bedrock response payload
            response_text = response['output']['message']['content'][0]['text']
            
            json_match = re.search(r'<json>(.*?)</json>', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("Could not find <json></json> blocks in the model output.")
                
            raw_json_string = json_match.group(1).strip()
            return json.loads(raw_json_string)
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep((2 ** attempt) + 2)
            
    raise RuntimeError("Failed to generate Domain Policy after maximum retries.")

def generate_domain_policy():
    policy = get_autonomous_domain_policy()

    protected_features = []
    technical_garbage = []

    for column, info in policy["columns"].items():
        role = info["role"]

        if ROLE_TO_BUCKET.get(role) == "protected":
            protected_features.append(column)
        else:
            technical_garbage.append(column)

    final_policy = {
        "domain": policy["domain"],
        "subject_id": policy["subject_id"],
        "protected_features": sorted(protected_features),
        "technical_garbage": sorted(technical_garbage),
        "sentiment": policy["sentiment"]
    }

    with open(DOMAIN_POLICY_PATH, 'w') as f:
        json.dump(final_policy, f, indent=4)
    print(f"✅ Autonomous Policy Finalized: {policy['domain']} context applied.")

if __name__ == "__main__":
    generate_domain_policy()