import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline():

    # ---- CLEANING AUDIT ----
    subprocess.run([
        sys.executable,
        "cleaning/supervised/audit_supervised.py"
    ], check=True)

    # ---- CLEANING EXECUTION ----
    subprocess.run([
        sys.executable,
        "cleaning/supervised/cleaning_supervised.py"
    ], check=True)

    # ---- CLEANED DATA PROFILER ----
    subprocess.run([
        sys.executable,
        "clean_profiler.py"
    ], check=True)


if __name__ == "__main__":
    print("🚀 Running supervised pipeline...")
    run_pipeline()