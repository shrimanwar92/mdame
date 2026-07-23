import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOLD_DATASET_PATH, MODEL_METRICS_REPORT, GOLD_AUDIT_REPORT, BEST_MODEL_PATH

def run_supervised_training():
    print("🧠 Phase 6: Supervised Model Training & Evaluation")
    
    # 1. Load Data
    df = pd.read_csv(GOLD_DATASET_PATH)
    with open(GOLD_AUDIT_REPORT, 'r') as f:
        gold_plan = json.load(f)

    target = gold_plan["metadata"]["target"]
    subject_id = gold_plan["metadata"]["subject_id"]
    algo_meta = gold_plan.get("algorithm_selection", {})
    
    # Drop IDs and isolate features
    X = df.drop(columns=[target, subject_id]) if subject_id in df.columns else df.drop(columns=[target])
    y = df[target]

    # 2. Train/Test Split (Generality Check)
    # We use a 20% hold-out set to detect overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Dynamic Model Initialization
    algo_name = algo_meta.get("algorithm", "RandomForest")
    problem_type = algo_meta.get("problem_type", "classification")
    params = algo_meta.get("base_hyperparameters", {})

    if algo_name == "XGBoost":
        model = XGBClassifier(**params) if problem_type == "classification" else XGBRegressor(**params)
    elif algo_name == "RandomForest":
        model = RandomForestClassifier(**params) if problem_type == "classification" else RandomForestRegressor(**params)
    else:
        model = LogisticRegression() if problem_type == "classification" else LinearRegression()

    # 4. Training
    print(f"🚀 Training {algo_name} for {problem_type}...")
    model.fit(X_train, y_train)

    # 5. Evaluation (Metrics Comparison)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {"algorithm": algo_name, "problem_type": problem_type}

    if problem_type == "classification":
        metrics.update({
            "train_accuracy": accuracy_score(y_train, train_preds),
            "test_accuracy": accuracy_score(y_test, test_preds),
            "f1_score": f1_score(y_test, test_preds, average='weighted'),
            "overfit_gap": accuracy_score(y_train, train_preds) - accuracy_score(y_test, test_preds)
        })
    else:
        metrics.update({
            "train_r2": r2_score(y_train, train_preds),
            "test_r2": r2_score(y_test, test_preds),
            "rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
            "overfit_gap": r2_score(y_train, train_preds) - r2_score(y_test, test_preds)
        })

    # 6. Save Model & Metrics
    joblib.dump(model, BEST_MODEL_PATH)
    with open(MODEL_METRICS_REPORT, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"📊 Training Results: Test Score: {metrics.get('test_accuracy', metrics.get('test_r2')):.4f}")
    if metrics["overfit_gap"] > 0.1:
        print("⚠️ WARNING: High overfitting detected! Consider increasing regularization.")
    else:
        print("✅ Model is generalized.")

if __name__ == "__main__":
    run_supervised_training()