import json
import os
import time
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Path management for your shared library structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import PROFILER_REPORT_PATH, DOMAIN_POLICY_PATH, TARGET_COLUMN

load_dotenv()

class SupervisedDomainArchitect:
    """
    Instance-based architect to generate domain policies using Profiler Alerts.
    """
    def __init__(self, target_variable: str):
        self.target_variable = target_variable
        self.api_key = os.getenv("MDAME_API_KEY")
        if not self.api_key:
            raise ValueError("MDAME_API_KEY environment variable is missing.")
        
        self.client = genai.Client(api_key=self.api_key)

    def _load_profiler_report(self):
        if not os.path.exists(PROFILER_REPORT_PATH):
            raise FileNotFoundError(f"Report not found: {PROFILER_REPORT_PATH}")
        with open(PROFILER_REPORT_PATH, 'r') as f:
            return json.load(f)

    def generate_policy(self, max_retries=3):
        report = self._load_profiler_report()
        
        # Extract components for efficient prompting
        alerts = report.get('alerts', [])
        all_columns = list(report.get('variables', {}).keys())

        system_instruction = "You are a Senior MLOps Architect. Output ONLY raw JSON."
        
        prompt = f"""
        Analyze the following Profiler Alerts and Column List for a Supervised ML task.
        Target Variable: '{self.target_variable}'

        Full Column List: {all_columns}
        Profiler Alerts: {json.dumps(alerts)}

        Logic Requirements:
        1. domain: Identify the specific industry domain (e.g., Maritime, Healthcare, Finance, Pharma, Banks, Entertainment, Textile).
        2. subject_id: Identify the primary entity identifier based on the domain.
        3. technical_garbage: 
           - Columns with "has unique values" alert that are NOT the subject_id.
           - Columns with >70% "missing values" alert.
           - Columns with "has constant value" alert.
        4. candidate_features: All columns that are NOT the target, subject_id, or technical_garbage.

        JSON SCHEMA:
        {{
            "domain": "string",
            "subject_id": "string or null",
            "target_variable": "{self.target_variable}",
            "technical_garbage": [],
            "candidate_features": []
        }}
        """

        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating Supervised Policy for target: {self.target_variable}...")
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.1
                    ),
                    contents=prompt
                )
                
                policy = json.loads(response.text)
                self._save_policy(policy)
                return policy

            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        
        return None

    def _save_policy(self, policy):
        with open(DOMAIN_POLICY_PATH, 'w') as f:
            json.dump(policy, f, indent=4)
        print(f"✅ Domain Policy saved to {DOMAIN_POLICY_PATH}")

if __name__ == "__main__":
    # Example usage in your pipeline
    architect = SupervisedDomainArchitect(target_variable=TARGET_COLUMN)
    architect.generate_policy()