import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline():
    # ---- GENERATE DOMAIN POLICY ----
    subprocess.run([
        sys.executable,
        "cleaning/clustering/generate_domain_policy.py"
    ], check=True)

    # ---- AUDIT ----
    subprocess.run([
        sys.executable,
        "cleaning/clustering/audit_clustering.py"
    ], check=True)

    # ---- CLEANING ----
    subprocess.run([
        sys.executable,
        "cleaning/clustering/cleaning_clustering.py"
    ], check=True)

    # ---- CLEANED DATA PROFILER ----
    subprocess.run([
        sys.executable,
        "profiler.py",
        "clean"
    ], check=True)


if __name__ == "__main__":
    print("🚀 Running cleaning clustering pipeline...")
    run_pipeline()