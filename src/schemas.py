from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Semantic Session Intent Schema (Used for Structured LLM Tagging)
# ==============================================================================
class SessionIntent(BaseModel):
    """Structured behavioral classification extracted from a compressed action_sequence."""
    primary_intent: Literal[
        "direct_impulse_buy",
        "price_sensitive_comparison",
        "brand_switching_exploration",
        "checkout_or_cart_friction",
        "casual_window_shopping"
    ] = Field(..., description="Dominant behavioral pattern observed in the session.")
    friction_detected: bool = Field(
        ..., description="True if the user showed hesitation, price step-downs, or cart abandonment."
    )
    friction_reason: Optional[str] = Field(
        None, description="Concise 1-sentence explanation of drop-off or hesitation driver."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence between 0.0 and 1.0."
    )


# ==============================================================================
# 2. Tier 3 Lightweight Ontology Models (Typed Objects, Links & Axioms)
# ==============================================================================
class OntologyNode(BaseModel):
    """Typed entity node in our behavioral domain ontology."""
    node_id: str
    node_type: Literal["SessionEpisode", "BrandEntity", "CategoryEntity"]
    label: str
    properties: dict = Field(default_factory=dict)


class OntologyEdge(BaseModel):
    """Typed causal or behavioral relationship edge between domain entities."""
    source_id: str
    target_id: str
    relation_type: Literal[
        "PURCHASED_BRAND",
        "ABANDONED_BRAND_CART",
        "SWITCHED_BRAND_TO",
        "BYPASSED_CART_DIRECT_BUY",
        "STEPPED_DOWN_PRICE_IN"
    ]
    weight: int = Field(default=1, ge=1)
    revenue_impact: float = Field(default=0.0)


# ==============================================================================
# 3. Standardized Benchmark Response Schema (Ensures Fair 3-Way Scoring)
# ==============================================================================
class BenchmarkResponse(BaseModel):
    """Uniform output returned by Tier 1, Tier 2, and Tier 3 agents for automated grading."""
    tier: Literal["Tier1_RawLogs", "Tier2_SemanticLayer", "Tier3_Ontology"]
    query_id: str
    numeric_answer: Optional[float] = Field(
        None, description="Primary metric value extracted to compare against Ground Truth SQL."
    )
    entity_answer: Optional[str] = Field(
        None, description="Primary category, brand, or funnel segment identified."
    )
    root_cause_explanation: str = Field(
        ..., description="Concise diagnostic explanation citing evidence from the queried layer."
    )
    generated_code_or_tool: str = Field(
        ..., description="The Spark SQL query or Python ontology tool invoked."
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    execution_error: Optional[str] = None
