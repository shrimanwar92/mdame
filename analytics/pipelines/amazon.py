import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from analytics.engines.review_compressor import SemanticReviewCompressor
from analytics.pipelines.orchestrator import Orchestrator
from constants import GOLD_MAPPED_DATASET_PATH, UNIVERSAL_PERSONAS_PATH
from utils import Utils

DEFAULT_SCHEMA = {
    "id_column": "ASIN",
    "product_name_column": "BRAND_NAME",
    "review_column": "TEXT",
    "rating_column": "STAR_RATING",
    "date_column": "REVIEW_DATE",
    "sentiment_column": "Feature_Sentiment_Score",
    "cluster_column": "Cluster",
    "certainty_column": "Certainty"
}


class AmazonReportGenerator:
    def __init__(self, df):
        self.__dict__.update(DEFAULT_SCHEMA)
        self.main_df = df
        self.orchestrator = Orchestrator(self, df=self.main_df)
        self.compressor = SemanticReviewCompressor(self)
    
    @staticmethod
    def slugify(text: str, max_length: int = 80) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")[:max_length]

    def update_macro_personas(self, universal_personas) -> dict:
        macro_semantic = self.compressor.compress_macro_clusters(gold_df=self.main_df)

        semantic_lookup = {
            int(item["cluster"]): item["semantic_evidence"]
            for item in macro_semantic
        }

        brand_lookup = {}
        for cluster_id, cluster_df in self.main_df.groupby("Cluster"):
            total = len(cluster_df)
            brand_distribution = (
                cluster_df
                .groupby(self.product_name_column)
                .size()
                .sort_values(ascending=False)
            )
            brand_lookup[int(cluster_id)] = {
                brand: {
                    "reviews": int(count),
                    "share": round(count / total * 100, 2),
                }
                for brand, count in brand_distribution.items()
            }

        for cluster_id, cluster_data in universal_personas.items():
            cluster_id = int(cluster_id)
            cluster_data["brand_distribution"] = brand_lookup.get(cluster_id, {})
            cluster_data["semantic_evidence"] = semantic_lookup.get(cluster_id, [])

        print("✓ Generated universal personas with semantic evidence")
        return universal_personas

    def generate_asin_reports(self, min_reviews: int = 300):
        asin_df = (
            self.main_df
            .groupby([self.id_column, self.product_name_column])
            .size()
            .reset_index(name="review_count")
            .query("review_count >= @min_reviews")
            .sort_values("review_count", ascending=False)
            .reset_index(drop=True)
        )

        print("\n" + "=" * 80 + "\nGenerating ASIN Reports\n" + "=" * 80)
        for row in asin_df.itertuples(index=False):
            asin = getattr(row, self.id_column)
            brand = self.slugify(getattr(row, self.product_name_column))
            print(f"[ASIN] {asin} ({brand})")

            save_path = self.orchestrator.run(
                identifier=asin,
                product_name=brand,
                group_by=self.id_column,
                report_type="asin",
            )
            print(f"✓ {save_path}")

    def generate_brand_reports(self, min_reviews: int = 300):
        brand_df = (
            self.main_df
            .groupby(self.product_name_column)
            .size()
            .reset_index(name="review_count")
            .query("review_count >= @min_reviews")
            .sort_values("review_count", ascending=False)
            .reset_index(drop=True)
        )

        print("\n" + "=" * 80 + "\nGenerating Brand Reports\n" + "=" * 80)
        for row in brand_df.itertuples(index=False):
            brand = getattr(row, self.product_name_column)
            slug = self.slugify(brand)
            print(f"[BRAND] {brand}")

            save_path = self.orchestrator.run(
                identifier=brand,
                product_name=slug,
                group_by=self.product_name_column,
                report_type="brand",
            )
            print(f"✓ {save_path}")

    def run(self, include_asin: bool = False, min_reviews: int = 300):
        if include_asin:
            self.generate_asin_reports(min_reviews=min_reviews)

        self.generate_brand_reports(min_reviews=min_reviews)
        print("\n" + "=" * 80 + "\nCompleted\n" + "=" * 80)


if __name__ == "__main__":
    df = Utils.read_data(GOLD_MAPPED_DATASET_PATH)
    generator = AmazonReportGenerator(df)
    generator.run(include_asin=False, min_reviews=300)