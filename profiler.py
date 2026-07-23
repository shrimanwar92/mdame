import argparse
import json
import pandas as pd
from data_profiling import ProfileReport
from constants import (
    DATASET_PATH, CLEANED_DATASET_PATH, GOLD_DATASET_PATH,
    PROFILER_REPORT_PATH, PROFILER_CLEAN_REPORT_PATH, 
    PROFILER_FEATURE_ENGG_REPORT_PATH, PRE_CLEAN_AUDIT_REPORT, AUTO_INC_ID
)

class BaseDataProfiler:
    """Base class providing standard data profiling extraction capabilities."""
    def __init__(self, mode: str, path: str, output: str, is_parquet: bool, index_col):
        self.mode = mode
        self.path = path
        self.output = output
        self.is_parquet = is_parquet
        self.index_col = index_col

    def run(self):
        print(f"🔍 Phase: Profiling {self.mode.upper()} data...")
        
        # Load Data
        if self.is_parquet:
            data = pd.read_parquet(self.path).set_index(self.index_col)
        else:
            data = pd.read_csv(self.path, encoding="latin1", index_col=self.index_col)

        # Generate Profile Report
        profile = ProfileReport(data, minimal=True, explorative=True, correlations=None,
                                interactions=None, missing_diagrams=None, samples=None, duplicates=None)
        profile = json.loads(profile.to_json())

        # Process statistics & inject sample values
        for name, col in profile["variables"].items():
            profile["variables"][name] = {
                k: v
                for k, v in col.items()
                if not isinstance(v, (dict, list))
            }
            series = data[name].dropna()
            if len(series):
                samples = (
                    series
                    .astype(str)
                    .drop_duplicates()
                    .sample(
                        n=min(5, series.nunique()),
                        random_state=42
                    )
                    .tolist()
                )
                profile["variables"][name]["sample_values"] = samples
        
        # Save output
        with open(self.output, "w") as f:
            json.dump(profile, f, indent=4)
        print(f"✅ Profiling complete. Saved to {self.output}")


class RawProfile(BaseDataProfiler):
    """Handles raw data ingestion metrics profile generation."""
    def __init__(self):
        super().__init__(
            mode="raw", 
            path=DATASET_PATH, 
            output=PROFILER_REPORT_PATH, 
            is_parquet=False, 
            index_col=None
        )


class CleanProfile(BaseDataProfiler):
    """Handles cleaned data profiling and custom schema schema expansions."""
    def __init__(self):
        super().__init__(
            mode="clean", 
            path=CLEANED_DATASET_PATH, 
            output=PROFILER_CLEAN_REPORT_PATH, 
            is_parquet=False, 
            index_col=AUTO_INC_ID
        )

    def run(self):
        # Run core profiling process
        super().run()
        # Execute secondary custom schema logic
        self.enhance_clean_schema(self.output)

    def enhance_clean_schema(self, report_path):
        print("🧠 Screening variables for 'LongText' type adjustments...")
        with open(report_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Thresholds
        LONG_TEXT_THRESHOLD_CHARS = 50
        UNIQUENESS_THRESHOLD = 0.10

        if "variables" in schema:
            for col_name, col_stats in schema["variables"].items():
                if col_stats.get("type") == "Text":
                    mean_len = col_stats.get("mean_length", 0)
                    p_dist = col_stats.get("p_distinct", 0.0)
                    
                    if mean_len > LONG_TEXT_THRESHOLD_CHARS and p_dist >= UNIQUENESS_THRESHOLD:
                        col_stats["type"] = "LongText"
                        print(f" ↳ 🏷️ Upgraded '{col_name}'")
                    else:
                        print(f" ↳ ⏭️ Kept '{col_name}'")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4, default=str)
        print("🎉 Inplace updates complete.")


class FeatureEnggProfile(BaseDataProfiler):
    """Handles Feature Engineered dataset profiling evaluation."""
    def __init__(self):
        super().__init__(
            mode="feature_engg", 
            path=GOLD_DATASET_PATH, 
            output=PROFILER_FEATURE_ENGG_REPORT_PATH, 
            is_parquet=True, 
            index_col=AUTO_INC_ID
        )

if __name__ == "__main__":
    # Class mapping router
    PROFILER_ROUTER = {
        "raw": RawProfile,
        "clean": CleanProfile,
        "feature_engg": FeatureEnggProfile
    }

    parser = argparse.ArgumentParser(description="Data Profiler & Analysis CLI")
    parser.add_argument("mode", choices=list(PROFILER_ROUTER.keys()), help="Profiling / analysis phase target")
    args = parser.parse_args()
    
    # Instantiate and run the corresponding class polymorphic setup
    profiler_class = PROFILER_ROUTER[args.mode]
    profiler_instance = profiler_class()
    profiler_instance.run()