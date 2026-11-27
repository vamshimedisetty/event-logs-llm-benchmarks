"""
golden_queries.py
-----------------
Curated Golden Query Bank & Zero-Dependency TF-IDF Cosine Retriever for Tier 4.

Design Principle (Zero Data Leakage):
All golden examples use held-out brands and categories (`sony`, `huawei`, `computers`,
`apparel`, `auto`) that NEVER appear in our evaluation suite (`samsung`, `apple`,
`xiaomi`, `electronics`, `appliances`). This guarantees Tier 4 learns structural tool
routing patterns without memorizing test answers.
"""

import math
import re
from collections import Counter
from typing import List, Dict

# Curated Golden Query Bank: Verified (Question -> Ontology Tool Call + Domain Reasoning) pairs
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


class GoldenQueryRetriever:
    """Lightweight TF-IDF cosine similarity retriever for dynamic few-shot tool routing."""

    def __init__(self, examples: List[Dict[str, str]] = GOLDEN_EXAMPLES):
        self.examples = examples
        self.docs = [self._tokenize(ex["question"]) for ex in examples]
        self.idf = self._compute_idf(self.docs)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alphanumeric tokens for bag-of-words TF-IDF scoring."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _compute_idf(self, docs: List[List[str]]) -> Dict[str, float]:
        """Computes smooth inverse document frequency across the golden query bank."""
        n = len(docs)
        df = Counter(tok for doc in docs for tok in set(doc))
        return {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        """Builds a sparse TF-IDF dictionary vector for a tokenized query."""
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        return {tok: (cnt / total) * self.idf.get(tok, 1.0) for tok, cnt in tf.items()}

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Computes cosine similarity between two sparse TF-IDF vectors."""
        dot = sum(val * v2.get(tok, 0.0) for tok, val in v1.items())
        norm1 = math.sqrt(sum(val * val for val in v1.values()))
        norm2 = math.sqrt(sum(val * val for val in v2.values()))
        return dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        """Returns the top-k closest golden query examples for dynamic prompt injection."""
        q_vec = self._tfidf_vec(self._tokenize(query))
        scored = [
            (self._cosine(q_vec, self._tfidf_vec(doc)), ex)
            for doc, ex in zip(self.docs, self.examples)
        ]
        scored.sort(key=lambda x: x, reverse=True)
        return [ex for _, ex in scored[:top_k]]
