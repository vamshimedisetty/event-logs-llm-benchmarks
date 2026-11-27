"""
schemas.py
----------
Pydantic v2 data contracts used across all four benchmark tiers:
1. SessionIntent: Structured behavioral classification extracted from compressed action sequences.
2. OntologyNode & OntologyEdge: Typed graph entities and relationship rules for Tier 3 & Tier 4.
3. BenchmarkResponse: Standardized output contract enabling apples-to-apples automated grading
   across accuracy, token consumption, latency, and execution errors.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Semantic Session Intent Schema (Structured LLM Tagging)
# ==============================================================================
class SessionIntent(BaseModel):
    """Structured behavioral classification extracted from a compressed action_sequence."""

    primary_intent: Literal[
        "direct_impulse_buy",
        "price_sensitive_comparison",
        "brand_switching_exploration",
        "checkout_or_cart_friction",
        "casual_window_shopping",
    ] = Field(..., description="Dominant behavioral pattern observed in the session.")
    friction_detected: bool = Field(
        ..., description="True if the user showed hesitation, price step-downs, or cart abandonment."
    )
    friction_reason: Optional[str] = Field(
        None, description="Concise 1-sentence explanation of drop-off or hesitation driver."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score bounded between 0.0 and 1.0."
    )


# ==============================================================================
# 2. Tier 3 & Tier 4 Lightweight Ontology Contracts (Typed Objects & Links)
# ==============================================================================
class OntologyNode(BaseModel):
    """Typed entity node in our behavioral domain ontology."""

    node_id: str = Field(..., description="Unique node identifier (e.g., 'brand:samsung', 'category:electronics').")
    node_type: Literal["SessionEpisode", "BrandEntity", "CategoryEntity"]
    label: str
    properties: dict = Field(default_factory=dict)


class OntologyEdge(BaseModel):
    """Typed causal or behavioral relationship edge between domain entities."""

    source_id: str = Field(..., description="Origin node ID (e.g., 'brand:samsung' or 'category:electronics').")
    target_id: str = Field(..., description="Destination node ID (e.g., 'brand:apple').")
    relation_type: Literal[
        "PURCHASED_BRAND",
        "ABANDONED_BRAND_CART",
        "SWITCHED_BRAND_TO",
        "BYPASSED_CART_DIRECT_BUY",
        "STEPPED_DOWN_PRICE_IN",
    ]
    weight: int = Field(default=1, ge=1, description="Number of clean non-bot sessions exhibiting this edge.")
    revenue_impact: float = Field(default=0.0, description="Total monetary value associated with this edge.")


# ==============================================================================
# 3. Standardized 4-Tier Benchmark Response Contract
# ==============================================================================
class BenchmarkResponse(BaseModel):
    """Uniform output returned by all four tiers so evaluate.py can grade objectively."""

    tier: Literal[
        "Tier1_RawLogs",
        "Tier2_SemanticLayer",
        "Tier3_Ontology",
        "Tier4_Ontology_PromptEng",
    ]
    query_id: str
    numeric_answer: Optional[float] = Field(
        None, description="Primary metric value extracted for ±2% tolerance check against Ground Truth."
    )
    entity_answer: Optional[str] = Field(
        None, description="Primary brand, category, or segment name extracted for entity matching."
    )
    root_cause_explanation: str = Field(
        default="", description="Concise diagnostic explanation grounded in the queried data layer."
    )
    generated_code_or_tool: str = Field(
        default="", description="Exact Spark SQL query generated (Tiers 1-2) or Python tool invoked (Tiers 3-4)."
    )
    prompt_tokens: int = Field(default=0, description="Total input tokens consumed across LLM calls.")
    completion_tokens: int = Field(default=0, description="Total output tokens generated across LLM calls.")
    latency_sec: float = Field(default=0.0, description="End-to-end wall-clock execution time in seconds.")
    execution_error: Optional[str] = Field(
        default=None, description="Captures SQL syntax errors, timeouts, or API exceptions if execution failed."
    )
