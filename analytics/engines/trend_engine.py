import pandas as pd


class TrendEngine:
    def __init__(self, parent):
       self.__dict__.update(parent.__dict__)

    def weighted_average(self, df, value_col, weight_col="reviews"):
        if df.empty or df[weight_col].sum() == 0:
            return 0.0

        return (
            (df[value_col] * df[weight_col]).sum()
            / df[weight_col].sum()
        )

    def extract_representative_reviews(
        self,
        reviews_df: pd.DataFrame,
        top_n: int = 5,
        min_words: int = 20,
    ) -> dict:
        if reviews_df.empty:
            return {
                "positive_reviews": [],
                "negative_reviews": []
            }

        df = reviews_df.copy()

        df["word_count"] = (
            df[self.review_column]
            .fillna("")
            .str.split()
            .str.len()
        )

        df = df[df["word_count"] >= min_words]

        if df.empty:
            return {
                "positive_reviews": [],
                "negative_reviews": []
            }

        df = df.drop_duplicates(
            subset=[self.review_column]
        )

        positive = (
            df.sort_values(
                by=[
                    self.sentiment_column,
                    self.certainty_column,
                    "word_count",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .head(top_n)
        )

        negative = (
            df.sort_values(
                by=[
                    self.sentiment_column,
                    self.certainty_column,
                    "word_count",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .head(top_n)
        )

        return {
            "positive_reviews": [
                row[self.review_column]
                for _, row in positive.iterrows()
            ],
            "negative_reviews": [
                row[self.review_column]
                for _, row in negative.iterrows()
            ],
        }

    def analyze(self, product_df: pd.DataFrame, window_months: int = 2) -> dict:
        if product_df.empty:
            return {}

        df = product_df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        monthly = (
            df.assign(period=df[self.date_column].dt.to_period("M"))
            .groupby("period")
            .apply(
                lambda x: pd.Series({
                    "reviews": len(x),
                    "average_rating": round(x[self.rating_column].mean(), 3),
                    "average_sentiment": round(x[self.sentiment_column].mean(), 3),
                    "cluster_distribution": (
                        x[self.cluster_column]
                        .value_counts(normalize=True)
                        .round(3)
                        .to_dict()
                    ),
                    "representative_reviews": self.extract_representative_reviews(x)
                }),
                include_groups=False
            )
            .reset_index()
        )

        monthly["period"] = monthly["period"].astype(str)

        timeline = monthly.to_dict("records")

        if len(monthly) < 2:
            return {"timeline": timeline}

        current = monthly.tail(window_months)

        previous = monthly.iloc[
            max(0, len(monthly) - 2 * window_months):
            len(monthly) - window_months
        ]

        review_volume = {
            "current": int(current["reviews"].sum()),
            "previous": int(previous["reviews"].sum()),
            "change_pct": round(
                (
                    current["reviews"].sum()
                    - previous["reviews"].sum()
                )
                / max(previous["reviews"].sum(), 1)
                * 100,
                2,
            ),
        }

        current_rating = self.weighted_average(current, "average_rating")
        previous_rating = self.weighted_average(previous, "average_rating")
        rating = {
            "current": round(current_rating, 3),
            "previous": round(previous_rating, 3),
            "change": round(
                current_rating - previous_rating,
                3,
            ),
        }

        current_sentiment = self.weighted_average(current, "average_sentiment")
        previous_sentiment = self.weighted_average(previous, "average_sentiment")
        sentiment = {
            "current": round(current_sentiment, 3),
            "previous": round(previous_sentiment, 3),
            "change": round(
                current_sentiment - previous_sentiment,
                3,
            ),
        }

        prev_dist = self._aggregate_distribution(previous)
        curr_dist = self._aggregate_distribution(current)

        changes = []
        for cluster in sorted(set(prev_dist) | set(curr_dist)):
            before = prev_dist.get(cluster, 0)
            after = curr_dist.get(cluster, 0)
            changes.append({
                "cluster": int(cluster),
                "share": round(after, 3),
                "change": round(after - before, 3),
            })

        emerging = sorted(changes, key=lambda x: x["change"], reverse=True)[:5]
        declining = sorted(changes, key=lambda x: x["change"])[:5]

        return {
            "timeline": timeline,
            "review_volume": review_volume,
            "rating": rating,
            "sentiment": sentiment,
            "emerging_clusters": emerging,
            "declining_clusters": declining,
        }

    @staticmethod
    def _aggregate_distribution(window_df: pd.DataFrame) -> dict:
        dist = {}
        total = 0.0

        for row in window_df["cluster_distribution"]:
            for cluster, share in row.items():
                dist[cluster] = dist.get(cluster, 0.0) + share
                total += share

        if total == 0:
            return {}

        return {k: v / total for k, v in dist.items()}
