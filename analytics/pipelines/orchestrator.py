import json
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from analytics.engines.report_builder import ReportBuilder
from analytics.engines.review_compressor import SemanticReviewCompressor
from analytics.engines.trend_engine import TrendEngine
from constants import PRODUCT_ANALYSIS_PATH, UNIVERSAL_PERSONAS_PATH


class Orchestrator:
    def __init__(self, parent, df):
        self.__dict__.update(parent.__dict__)
        self.df = df
        self.has_sentiment = self.sentiment_column is not None
        self.has_reviews = self.review_column is not None
        self.has_rating = self.rating_column is not None

        self.trend_engine = TrendEngine(self)
        self.report_builder = ReportBuilder()

    def run(
        self,
        identifier,
        product_name,
        group_by=None,
        report_type="product",
    ):
        group_by = group_by or self.id_column
        product_df = self._filter(identifier, group_by)

        if product_df.empty:
            raise ValueError(f"No rows found for {group_by}={identifier}")

        print(f"{report_type.upper()} | {group_by}={identifier} | Reviews={len(product_df)}")

        with open(UNIVERSAL_PERSONAS_PATH) as f:
            universal_macro_personas = json.load(f)

        product_summary = self._product_summary(
            product_df,
            identifier,
            group_by,
            report_type,
        )

        cluster_payload = self._cluster_payload(product_df=product_df)
        trends = self.trend_engine.analyze(product_df)
        report = self.report_builder.build(
            product=product_summary,
            cluster=cluster_payload,
            trend=trends,
            macro=universal_macro_personas,
        )

        filename = f"{report_type}_[{identifier}]_[{product_name}]_analysis.json"
        save_path = PRODUCT_ANALYSIS_PATH.with_name(filename)

        self.report_builder.save(report, output_path=save_path)

        return save_path

    def _filter(self, identifier, group_by):
        return self.df[self.df[group_by] == identifier].copy().reset_index(drop=True)

    def _product_summary(
        self,
        df,
        identifier,
        group_by,
        report_type,
    ):
        summary = {
            "report_type": report_type,
            "group_by": group_by,
            "identifier": identifier,
        }

        if group_by == self.id_column:
            summary[self.id_column] = identifier
            summary[self.product_name_column] = df[self.product_name_column].mode().iloc[0]

        elif group_by == self.product_name_column:
            summary[self.product_name_column.lower()] = identifier
            summary["products"] = df[self.id_column].nunique()
            summary[self.id_column.lower()] = df[self.id_column].drop_duplicates().tolist()

        if self.has_reviews:
            summary["reviews"] = len(df)

        if self.has_rating:
            summary["average_rating"] = round(df[self.rating_column].mean(), 2)

        if self.has_sentiment:
            summary["average_sentiment"] = round(df[self.sentiment_column].mean(), 3)

        return summary

    def _cluster_payload(self, product_df):
        payload = []

        grouped = (
            product_df.groupby(self.cluster_column)
            .size()
            .sort_values(ascending=False)
            .index
        )

        total = len(product_df)

        for cluster in grouped:
            cdf = product_df[product_df[self.cluster_column] == cluster]

            item = {
                "cluster": int(cluster),
                "share": round(len(cdf) / total, 3),
                "certainty": round(cdf[self.certainty_column].mean(), 3),
            }

            if self.has_reviews:
                item["reviews"] = len(cdf)

            if self.has_rating:
                item["average_rating"] = round(cdf[self.rating_column].mean(), 2)

            if self.has_sentiment:
                item["average_sentiment"] = round(cdf[self.sentiment_column].mean(), 3)

            payload.append(item)

        return payload