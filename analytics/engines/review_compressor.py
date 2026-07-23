import pandas as pd
from utils import Utils
import math
from sklearn.metrics.pairwise import cosine_similarity

class SemanticReviewCompressor:

    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)


    def filter_reviews_by_certainty_quantiles(
        self,
        df: pd.DataFrame,
        n_quantiles: int = 10,
        samples_per_quantile: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:

        # Nothing to do for empty dataframe
        if df.empty:
            return df.copy()

        # For small clusters, keep everything
        max_samples = n_quantiles * samples_per_quantile
        if len(df) <= max_samples:
            return df.copy()

        # Sort by certainty
        df = df.sort_values(self.certainty_column).copy()

        bucket_size = math.ceil(len(df) / n_quantiles)

        sampled = []

        for i in range(n_quantiles):

            start = i * bucket_size
            end = min((i + 1) * bucket_size, len(df))

            bucket = df.iloc[start:end]

            if bucket.empty:
                continue

            sampled.append(
                bucket.sample(
                    n=min(samples_per_quantile, len(bucket)),
                    random_state=random_state,
                )
            )

        if not sampled:
            return df.copy()

        sampled_df = (
            pd.concat(sampled)
            .sort_values(self.certainty_column)
            .reset_index(drop=True)
        )

        return sampled_df


    def compress_macro_clusters(
        self,
        gold_df: pd.DataFrame,
        similarity_threshold: float = 0.90,
    ):
        macro_semantic = []
        MAX_SEMANTIC_REVIEWS, MIN_WORDS, MAX_WORDS = 60, 8, 50

        for cluster_id, cluster_df in gold_df.groupby(self.cluster_column):
            print(f"\nMacro Cluster {cluster_id}: {len(cluster_df):,} reviews")
            sampled_products = []

            # Process each product independently
            for _, product_df in cluster_df.groupby(self.product_name_column):
                positive = product_df[product_df[self.sentiment_column] >= 0.70]
                negative = product_df[product_df[self.sentiment_column] <= 0.35]
                sampled = []

                for subset in (positive, negative):
                    if not subset.empty:
                        sampled.append(
                            self.filter_reviews_by_certainty_quantiles(
                                subset,
                                n_quantiles=10,
                                samples_per_quantile=5,
                            )
                        )

                if sampled:
                    product_sample = (
                        pd.concat(sampled, ignore_index=True)
                        .drop_duplicates(subset=[self.review_column])
                        .sort_values(self.certainty_column, ascending=False)
                    )
                    sampled_products.append(product_sample)

            # Merge all products
            if sampled_products:
                sampled_df = (
                    pd.concat(sampled_products, ignore_index=True)
                    .drop_duplicates(subset=[self.review_column])
                    .sort_values(self.certainty_column, ascending=False)
                    .reset_index(drop=True)
                )
            else:
                sampled_df = cluster_df.copy()

            # Normalize text and filter length
            sampled_df[self.review_column] = (
                sampled_df[self.review_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            sampled_df = sampled_df[sampled_df[self.review_column] != ""]
            sampled_df = sampled_df[sampled_df[self.review_column].str.split().str.len() >= MIN_WORDS]
            sampled_df[self.review_column] = sampled_df[self.review_column].apply(
                lambda x: " ".join(x.split()[:MAX_WORDS])
            )

            print(f"Representative Sample: {len(cluster_df):,} → {len(sampled_df):,}")

            reviews = sampled_df[self.review_column].tolist()
            products = sampled_df[self.product_name_column].tolist()
            sentiments = sampled_df[self.sentiment_column].tolist()

            # Deduplicate & extract semantic evidence
            if len(reviews) <= 1:
                semantic_reviews = [
                    f"{p}|{s}|{r}"
                    for p, s, r in zip(products, sentiments, reviews)
                ]
            else:
                embeddings = Utils.execute_onnx_embedding(reviews)
                sim_matrix = cosine_similarity(embeddings)
                keep = []

                for i in range(len(reviews)):
                    if not any(sim_matrix[i, j] >= similarity_threshold for j in keep):
                        keep.append(i)
                    if len(keep) >= MAX_SEMANTIC_REVIEWS:
                        break

                semantic_reviews = [
                    f"{products[i]}|{'positive' if sentiments[i] >= 0.70 else 'negative'}|{reviews[i]}"
                    for i in keep
                ]

            print(f"Semantic Deduplication: {len(reviews)} → {len(semantic_reviews)}")
            macro_semantic.append(
                {
                    "cluster": int(cluster_id),
                    "semantic_evidence": semantic_reviews,
                }
            )

        return macro_semantic