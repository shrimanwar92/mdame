import json


class ReportBuilder:
    def build(self, product, cluster, trend, macro):
        macro_lookup = {str(k): v for k, v in macro.items()}
        macro_personas = []

        for persona in cluster:
            cluster_id = str(persona["cluster"])
            if cluster_id not in macro_lookup:
                continue

            universal_persona = macro_lookup[cluster_id].copy()
            universal_persona.pop("cluster_id", None)
            universal_persona.pop("semantic_evidence", None)
            product_persona = persona.copy()

            comparison = {}
            pop_pct = universal_persona.get("population_pct")
            if pop_pct and pop_pct > 0:
                comparison["share_lift"] = round(product_persona["share"] / (pop_pct / 100), 2)

            avg_rating = universal_persona.get("avg_rating")
            prod_rating = product_persona.get("average_rating")
            if avg_rating is not None and prod_rating is not None:
                comparison["rating_delta"] = round(prod_rating - avg_rating, 2)

            avg_sentiment = universal_persona.get("avg_sentiment")
            prod_sentiment = product_persona.get("average_sentiment")
            if avg_sentiment is not None and prod_sentiment is not None:
                comparison["sentiment_delta"] = round(prod_sentiment - avg_sentiment, 3)

            macro_personas.append({
                "cluster": int(cluster_id),
                "universal_persona": universal_persona,
                "product_persona": product_persona,
                "comparison": comparison,
            })

        macro_personas.sort(key=lambda x: x["product_persona"]["share"], reverse=True)

        return {
            "product": product,
            "macro_personas": macro_personas,
            "trend_analysis": trend,
        }

    def save(self, report: dict, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print("Product Intelligence Report saved.")