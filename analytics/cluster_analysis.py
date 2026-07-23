import pandas as pd
import sys
import os
import json
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    CLEANED_DATASET_PATH,
    GOLD_LABELLED_DATASET_PATH, 
    GOLD_MAPPED_DATASET_PATH, 
    UNIVERSAL_PERSONAS_PATH,
    AUTO_INC_ID
)
from utils import Utils
from analytics.pipelines.amazon import AmazonReportGenerator
from analytics.engines.llm_behavior_engine import LLMBehaviorEngine

class ClusterAnalysis:
    MODEL_OUTPUT_COLUMNS = ['Cluster', 'Certainty', 'Feature_Sentiment_Score']

    def add_llm_behavior(self, macro_persona):
        llm = LLMBehaviorEngine()
        llm_cluster_info = llm.generate(macro_persona)

        llm_lookup = {
            item["cluster"]: item
            for item in llm_cluster_info
        }
        merged = deepcopy(macro_persona)
        for cluster_id, cluster_data in merged.items():
            cluster_id = int(cluster_id)
            
            if cluster_id not in llm_lookup:
                continue

            profile = llm_lookup[cluster_id]

            cluster_data["cluster_profile"] = {
                "name": profile.get("cluster_profile_name"),
                "summary": profile.get("cluster_summary"),
                "decision_drivers": profile.get("decision_drivers", []),
                "satisfaction_drivers": profile.get("satisfaction_drivers", []),
                "frustration_drivers": profile.get("frustration_drivers", []),
                "confidence": profile.get("confidence", "Unknown"),
            }

        return merged


    """Handles post-training consolidation: aligns label metadata and builds cluster analytics."""
    def run(self):
        print("📊 Starting Cluster Analysis Post-Training Routine...")
        df = self.generate_gold_mapping()
        generator = AmazonReportGenerator(df)
        macro_persona = self.generate_cluster_statistics(df)
        macro_persona = generator.update_macro_personas(macro_persona)
        macro_persona = self.add_llm_behavior(macro_persona)

        with open(UNIVERSAL_PERSONAS_PATH, 'w') as f:
            json.dump(macro_persona, f, indent=4)
        print(f"🎉 Analysis full workflow complete. Personas stored at: {UNIVERSAL_PERSONAS_PATH}")        

    def generate_gold_mapping(self):
        print("📖 Generating gold mapping dataset...")
        df_labelled = Utils.read_data(GOLD_LABELLED_DATASET_PATH)
        df_clean = Utils.read_data(CLEANED_DATASET_PATH)
        
        # Ensure the identifier key is preserved for the join operation
        if AUTO_INC_ID in df_labelled.columns:
            labelled_subset_cols = list(set([AUTO_INC_ID] + ClusterAnalysis.MODEL_OUTPUT_COLUMNS))
            df_labelled_sub = df_labelled[labelled_subset_cols].copy()
        else:
            df_labelled_sub = df_labelled[ClusterAnalysis.MODEL_OUTPUT_COLUMNS].copy()

        # Align indexes using the auto-increment ID to prepare for the join
        for df in [df_clean, df_labelled_sub]:
            if AUTO_INC_ID in df.columns:
                df.set_index(AUTO_INC_ID, inplace=True)

        print("🤝 Joining dataframes (Keeping all clean columns + cluster metadata)...")
        # Join keeping all columns from clean, matching against the filtered labelled data
        df_joined = df_clean.join(df_labelled_sub, how='inner')

        # Bring the ID back from the index into a standard column for the CSV output
        df_joined.reset_index(inplace=True)
        Utils.save_to_parquet(df_joined, GOLD_MAPPED_DATASET_PATH)
        print("✅ Mapping successfully created and stored.")
        return df_joined

    def generate_cluster_statistics(self, df) -> dict:
        """Build deterministic cluster statistics from Gold Mapping dataset."""
        print("📖 Loading Gold Mapping Dataset...")

        clusters = {}
        total_rows = len(df)

        for cluster_id, cluster_df in df.groupby("Cluster"):        
            sentiment = cluster_df["Feature_Sentiment_Score"]
            positive_pct = (sentiment >= 0.70).mean() * 100
            neutral_pct = ((sentiment > 0.35) & (sentiment < 0.70)).mean() * 100
            negative_pct = (sentiment <= 0.35).mean() * 100

            if positive_pct >= max(neutral_pct, negative_pct):
                dominant_sentiment = "Positive"
            elif negative_pct >= max(positive_pct, neutral_pct):
                dominant_sentiment = "Negative"
            else:
                dominant_sentiment = "Neutral"

            positive = (
                cluster_df[
                    cluster_df["Feature_Sentiment_Score"] >= 0.70
                ]
                .sort_values(
                    ["Feature_Sentiment_Score", "Certainty"],
                    ascending=False
                )
                .head(3)
            )

            neutral = (
                cluster_df[
                    (cluster_df["Feature_Sentiment_Score"] > 0.35)
                    &
                    (cluster_df["Feature_Sentiment_Score"] < 0.70)
                ]
                .assign(
                    distance=lambda x: (
                        x["Feature_Sentiment_Score"] - 0.5
                    ).abs()
                )
                .sort_values(
                    ["distance", "Certainty"],
                    ascending=[True, False]
                )
                .head(2)
            )

            negative = (
                cluster_df[
                    cluster_df["Feature_Sentiment_Score"] <= 0.35
                ]
                .sort_values(
                    ["Feature_Sentiment_Score", "Certainty"],
                    ascending=[True, False]
                )
                .head(3)
            )

            clusters[int(cluster_id)] = {
                "cluster_id": int(cluster_id),
                "population": len(cluster_df),
                "population_pct": round(len(cluster_df) / total_rows * 100, 2),
                "certainty": round(cluster_df["Certainty"].mean(), 3),
                "positive_pct": round(positive_pct, 2),
                "neutral_pct": round(neutral_pct, 2),
                "negative_pct": round(negative_pct, 2),
                "avg_rating": round(cluster_df["STAR_RATING"].mean(), 2) if "STAR_RATING" in cluster_df.columns else None,
            }

        print(f"✅ Generated statistics for {len(clusters)} clusters")
        return clusters

if __name__ == "__main__":
    post_training_analysis = ClusterAnalysis()
    post_training_analysis.run()
