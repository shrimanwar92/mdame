import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline():

    # ---- AUDIT ----
    subprocess.run([
        sys.executable,
        "feature_engg/clustering/gold_audit_clustering.py"
    ], check=True)

    # ---- FEATURE ENGINEERING ----
    subprocess.run([
        sys.executable,
        "feature_engg/clustering/feature_engg_clustering.py"
    ], check=True)

    # ---- FEATURE ENGG DATA PROFILER ----
    subprocess.run([
        sys.executable,
        "profiler.py",
        "feature_engg"
    ], check=True)


if __name__ == "__main__":
    print("🚀 Running feature engineering clustering pipeline...")
    run_pipeline()