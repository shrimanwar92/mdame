import json
import subprocess
import sys
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline():

    # ---- TRAIN AUDIT ----
    subprocess.run([
        sys.executable,
        "training/clustering/train_audit_clustering.py"
    ], check=True)

    # ---- TRAINING ----
    subprocess.run([
        sys.executable,
        "training/clustering/train_clustering.py"
    ], check=True)


if __name__ == "__main__":
    print("🚀 Running training...")
    run_pipeline()