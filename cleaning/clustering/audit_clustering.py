import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from constants import (
    PROFILER_REPORT_PATH,
    PRE_CLEAN_AUDIT_REPORT,
    DOMAIN_POLICY_PATH,
)


def run_clustering_audit():
    print("🧠 Phase 2: Audit Brain (Strict Baseline Mode)")

    with open(PROFILER_REPORT_PATH, "r") as f:
        report = json.load(f)

    with open(DOMAIN_POLICY_PATH, "r") as f:
        policy = json.load(f)

    variables = report.get("variables", {})
    protected = policy.get("protected_features", [])
    garbage = policy.get("technical_garbage", [])
    subject_id = policy.get("subject_id")
    sentiment = policy.get("sentiment", {})

    contract = {
        "subject_id": subject_id,
        "duplicates": {
            "enabled": True,
            "keep": "first"
        },
        "drop_features": [],
        "imputation": {
            "mean": [],
            "median": [],
            "boolean_mode": [],
            "categorical_mode": []
        },
        "sentiment": sentiment
    }

    # ------------------------------------------------------------------
    # Mandatory Drops
    # ------------------------------------------------------------------
    contract["drop_features"] = [
        col for col in garbage
        if col != subject_id
    ]

    # ------------------------------------------------------------------
    # Column Audit
    # ------------------------------------------------------------------
    for col, stats in variables.items():

        if col == subject_id or col in contract["drop_features"]:
            continue

        col_type = stats.get("type")
        p_missing = stats.get("p_missing", 0.0)
        n_distinct = stats.get("n_distinct", 0)
        n_missing = stats.get("n_missing", 0)

        # --------------------------------------------------------------
        # Quality Gates
        # --------------------------------------------------------------
        if col not in protected:

            # Too many missing values
            if p_missing > 0.60:
                contract["drop_features"].append(col)
                continue

            # Constant column
            if n_distinct <= 1:
                contract["drop_features"].append(col)
                continue

        # --------------------------------------------------------------
        # Imputation Routing
        # --------------------------------------------------------------
        if n_missing > 0:

            # Numerical
            if col_type == "Numeric":
                skew = abs(stats.get("skewness", 0.0))

                if skew > 1.0:
                    contract["imputation"]["median"].append(col)
                else:
                    contract["imputation"]["mean"].append(col)

            # Boolean
            elif col_type == "Boolean":
                contract["imputation"]["boolean_mode"].append(col)

            # Text / Categorical
            elif col_type in ["Text", "Categorical"]:
                contract["imputation"]["categorical_mode"].append(col)

    with open(PRE_CLEAN_AUDIT_REPORT, "w") as f:
        json.dump(contract, f, indent=4)

    print("✅ Cleaning baseline audit complete.")


if __name__ == "__main__":
    run_clustering_audit()