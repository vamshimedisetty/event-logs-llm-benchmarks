"""
golden_queries.py
-----------------
Curated Golden Query Bank & Zero-Dependency TF-IDF Cosine Retriever for Tier 4.

Architectural Role in Tier 4 (`Tier4_Ontology_PromptEng`):
Instead of cramming dozens of static examples into the system prompt (which inflates
token costs and dilutes model attention), Tier 4 uses lightweight TF-IDF cosine
similarity to retrieve ONLY the top-k (k=2) structurally closest verified examples
at runtime and injects them dynamically into the prompt.

Experimental Rigor (Zero Data Leakage):
Every entity in `GOLDEN_EXAMPLES` (`sony`, `huawei`, `computers`, `apparel`, `auto`)
is strictly held out from our evaluation test suite (`samsung`, `apple`, `xiaomi`,
`electronics`, `appliances`). This guarantees Tier 4 learns structural tool-routing
patterns and ontology axioms rather than memorizing benchmark answers.
"""

import math
import re
from collections import Counter
from typing import List, Dict


# ==============================================================================
# 1. Curated Golden Query Bank (Held-Out Training / Retrieval Examples)
# ==============================================================================
GOLDEN_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "Which brand do shoppers switch to most often after viewing sony products?",
        "tool_call": 'trace_brand_leakage(brand="sony")',
        "reasoning": "Brand-to-brand competitive leakage traverses directed SWITCHED_BRAND_TO edges.",
    },
    {
        "question": "What is the top competitor capturing users who abandon huawei in computers?",
        "tool_call": 'trace_brand_leakage(brand="huawei", category="computers")',
        "reasoning": "Category-scoped brand switching passes both brand and optional category to trace_brand_leakage.",
    },
    {
        "question": "Which brand suffers the highest number of abandoned carts in computers?",
        "tool_call": 'analyze_category_friction(category="computers")',
        "reasoning": "Category-level cart abandonment ranks ABANDONED_BRAND_CART edges.",
    },
    {
        "question": "What percentage of purchasing sessions in apparel bypassed the cart via direct purchase?",
        "tool_call": 'get_category_revenue_and_funnel(category="apparel")',
        "reasoning": "Direct 1-click checkout share uses get_category_revenue_and_funnel.",
    },
    {
        "question": "What is the total purchase revenue in the auto category excluding bot traffic?",
        "tool_call": 'get_category_revenue_and_funnel(category="auto")',
        "reasoning": "Category purchase revenue aggregates revenue_impact across all purchase edges.",
    },
]


# ==============================================================================
# 2. Zero-Dependency TF-IDF Cosine Retriever
# ==============================================================================
class GoldenQueryRetriever:
    """
    Pure-Python TF-IDF cosine similarity retriever for dynamic few-shot tool routing.

    Why Pure Python instead of an external Vector DB?
    - Runs in <1 millisecond in-memory with zero extra API calls or embedding latency.
    - Deterministic and reproducible across any Python environment without external services.
    """

    def __init__(self, examples: List[Dict[str, str]] = GOLDEN_EXAMPLES):
        self.examples = examples
        # Pre-tokenize all golden questions at initialization time
        self.docs = [self._tokenize(ex["question"]) for ex in examples]
        # Pre-compute Inverse Document Frequency (IDF) weights across the golden bank
        self.idf = self._compute_idf(self.docs)

    def _tokenize(self, text: str) -> List[str]:
        """Normalizes text to lowercase and extracts alphanumeric word tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _compute_idf(self, docs: List[List[str]]) -> Dict[str, float]:
        """
        Computes smooth Inverse Document Frequency (IDF):
            idf(t) = log((N + 1) / (df(t) + 1)) + 1.0
        Rare structural keywords (e.g., 'switch', 'abandoned', 'bypassed', 'revenue')
        receive higher weights than generic filler words.
        """
        n = len(docs)
        df = Counter(tok for doc in docs for tok in set(doc))
        return {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        """Converts a token list into a normalized sparse TF-IDF vector dictionary."""
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        return {tok: (cnt / total) * self.idf.get(tok, 1.0) for tok, cnt in tf.items()}

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Computes cosine similarity (0.0 to 1.0) between two sparse TF-IDF vectors."""
        dot = sum(val * v2.get(tok, 0.0) for tok, val in v1.items())
        norm1 = math.sqrt(sum(val * val for val in v1.values()))
        norm2 = math.sqrt(sum(val * val for val in v2.values()))
        return dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        """
        Scores an incoming user question against all golden examples and returns
        the top-k closest structural matches for dynamic prompt injection.
        """
        q_vec = self._tfidf_vec(self._tokenize(query))
        scored = [
            (self._cosine(q_vec, self._tfidf_vec(doc)), ex)
            for doc, ex in zip(self.docs, self.examples)
        ]
        # Sort descending by cosine similarity score (index 0 of each tuple)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]
