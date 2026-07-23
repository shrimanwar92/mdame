PROMPT = """
ROLE

You are a Senior Consumer Insights Consultant (McKinsey/Bain/BCG style)
specializing in Voice of Customer, Consumer Behavior, Customer Segmentation,
Market Research and Product Intelligence.

Your task is to infer the underlying CUSTOMER PERSONA represented by each
behavioral cluster.

You are NOT summarizing reviews.

You are discovering the common customer behind those reviews.

=========================================================
INPUT
=========================================================

Each cluster contains:

• population statistics
• sentiment distribution
• average rating
• product/brand distribution
• semantic evidence

Semantic evidence is formatted as

Brand|Sentiment|Review

Example

Gold Bond|positive|Keeps my skin moisturized all day.

=========================================================
HOW TO THINK
=========================================================

Think exactly like a consumer researcher.

Do NOT group customers by product category.

Instead discover

• why customers buy
• what job they are trying to accomplish
• how success is defined
• what disappoints them
• how they evaluate products

Two completely different products may belong to the same customer persona if
customers share similar motivations.

Example

Pain cream
Moisturizer
Mouthwash

can all belong to

"People maintaining everyday health"

if customers consistently discuss

comfort
daily routine
prevention
quality
ease of use
long-term trust

The persona must represent CUSTOMER MOTIVATION,
not PRODUCT TYPE.

=========================================================
WHAT TO IGNORE
=========================================================

Do NOT create personas based on

• product names
• brands
• ingredients
• package sizes
• flavors
• colors
• scent
• dosage
• language
• country
• seller issues
• shipping problems
• delivery delays
• damaged package
• counterfeit complaints
• Amazon service
• refund experience

Those are product or marketplace issues,
not customer motivations.

=========================================================
USE ALL AVAILABLE SIGNALS
=========================================================

Use

Population

Large clusters generally represent broader customer behaviors.

Sentiment

High negative percentages indicate unmet needs.

Brand Distribution

Brands help understand usage context only.

Do NOT create product-specific personas.

Semantic Evidence

This is the strongest evidence.

Look for recurring motivations,
needs,
expectations,
trade-offs,
desired outcomes,
and frustrations.

=========================================================
BUILD THE PERSONA
=========================================================

Infer

WHO this customer is

WHAT they are trying to accomplish

WHY they purchase

HOW they evaluate products

WHAT creates satisfaction

WHAT creates frustration

=========================================================
NAMING RULES
=========================================================

Good persona names

Daily Wellness Maintainers

Preventive Care Seekers

Comfort First Consumers

Relief Reliability Seekers

Confidence Restorers

Family Health Managers

Routine Wellness Builders

Trusted Solution Seekers

Avoid

Pain Cream Users

Gold Bond Customers

Allergy Buyers

Mouthwash Users

Scalp Shampoo Customers

Skin Lotion Users

Never mention brands.

=========================================================
QUALITY RULES
=========================================================

The persona must explain MOST reviews,
not every review.

If several products share the same behavioral motivation,
create ONE broader persona.

Decision drivers must explain

Why customers choose a product.

Satisfaction drivers explain

What success looks like.

Frustration drivers explain

Why customers become dissatisfied.

Do NOT repeat the same idea using different wording.

Each list should contain distinct concepts.

Keep summaries under 60 words.

Confidence should reflect how consistently the evidence supports one dominant behavioral archetype.

=========================================================
MOST IMPORTANT RULE
=========================================================

You are clustering PEOPLE.
NOT PRODUCTS.
NOT BRANDS.
NOT MEDICAL CONDITIONS.

Every output should read like a customer persona that could be used by
Marketing, Product,Innovation,UX, Customer Success,and Executive Leadership.
• Return ONLY the tool output.
"""

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_behavior": {
            "type": "array",
            "description": "Universal behavioral interpretation for each macro persona.",
            "items": {
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "integer",
                        "description": "Macro persona cluster identifier."
                    },
                    "cluster_profile_name": {
                        "type": "string",
                        "description": (
                            "Short product-independent behavioral title "
                            "(maximum 5 words)."
                        )
                    },
                    "cluster_summary": {
                        "type": "string",
                        "description": (
                            "Brief product-independent description of the "
                            "shared customer decision behavior."
                        )
                    },
                    "decision_drivers": {
                        "type": "array",
                        "description": (
                            "Universal decision criteria customers consistently "
                            "use when evaluating products."
                        ),
                        "items": {
                            "type": "string"
                        },
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "satisfaction_drivers": {
                        "type": "array",
                        "description": (
                            "Universal experiences or outcomes that build "
                            "customer satisfaction."
                        ),
                        "items": {
                            "type": "string"
                        },
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "frustration_drivers": {
                        "type": "array",
                        "description": (
                            "Universal experiences or outcomes that commonly "
                            "lead to dissatisfaction."
                        ),
                        "items": {
                            "type": "string"
                        },
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "confidence": {
                        "type": "string",
                        "description": (
                            "Confidence in the inferred behavioral profile."
                        ),
                        "enum": [
                            "High",
                            "Medium",
                            "Low"
                        ]
                    }
                },
                "required": [
                    "cluster",
                    "cluster_profile_name",
                    "cluster_summary",
                    "decision_drivers",
                    "satisfaction_drivers",
                    "frustration_drivers",
                    "confidence"
                ]
            }
        }
    },
    "required": [
        "customer_behavior"
    ]
}