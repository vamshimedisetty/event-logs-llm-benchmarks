"""
agents.py
---------
Implements our 4-Tier Progressive Data & Prompt Architecture Benchmark:
  - Tier 1 (Raw Event Logs): Stateless Text-to-SQL directly over 42.45M `bronze_events` rows.
  - Tier 2 (Semantic Layer): Text-to-SQL over stateful, bot-filtered `silver_sessions`.
  - Tier 3 (Bare Ontology): Deterministic NetworkX graph traversal tools with zero prompt engineering.
  - Tier 4 (Ontology + Prompt Eng + Golden Retrieval): Combines Tier 3 graph tools with explicit
    domain axioms, dynamic TF-IDF golden query retrieval, and strict JSON extraction rules.
"""

import json
import time
from typing import Dict, Any, Optional
import networkx as nx
from openai import OpenAI
from schemas import BenchmarkResponse
from golden_queries import GoldenQueryRetriever


# ==============================================================================
# 1. Deterministic Ontology Graph Engine (Backend for Tiers 3 & 4)
# ==============================================================================
class OntologyGraphEngine:
    """In-memory NetworkX MultiDiGraph built from `workspace.default.ont_edges`."""

    def __init__(self, spark):
        self.spark = spark
        self.graph = nx.MultiDiGraph()
        self._load_graph()

    def _load_graph(self):
        """Loads typed object nodes and weighted relationship edges from Delta into NetworkX."""
        edges = self.spark.table("workspace.default.ont_edges").collect()
        for r in edges:
            self.graph.add_edge(
                r["source_id"],
                r["target_id"],
                relation_type=r["relation_type"],
                context_category=r["context_category"],
                weight=int(r["weight"]),
                revenue_impact=float(r["revenue_impact"]),
            )

    def trace_brand_leakage(self, brand: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Traverses directed `SWITCHED_BRAND_TO` edges to rank top competitor brands."""
        src = f"brand:{brand.lower()}"
        if src not in self.graph:
            return {"error": f"Brand '{brand}' not found in ontology."}

        competitors: Dict[str, int] = {}
        for _, target, data in self.graph.out_edges(src, data=True):
            if data["relation_type"] == "SWITCHED_BRAND_TO":
                if category and data["context_category"] != category.lower():
                    continue
                comp_name = target.replace("brand:", "")
                competitors[comp_name] = competitors.get(comp_name, 0) + data["weight"]

        ranked = sorted(competitors.items(), key=lambda x: x, reverse=True)[:5]
        return {
            "source_brand": brand.lower(),
            "category_filter": category or "all",
            "top_switched_to_brands": [{"brand": b, "switch_sessions": w} for b, w in ranked],
            "top_competitor": ranked if ranked else None,
            "top_competitor_switches": ranked if ranked else 0,
        }

    def analyze_category_friction(self, category: str) -> Dict[str, Any]:
        """Traverses `ABANDONED_BRAND_CART` edges from a category node to identify top friction brands."""
        src = f"category:{category.lower()}"
        if src not in self.graph:
            return {"error": f"Category '{category}' not found in ontology."}

        abandoned_by_brand: Dict[str, int] = {}
        for _, target, data in self.graph.out_edges(src, data=True):
            if data["relation_type"] == "ABANDONED_BRAND_CART":
                b_name = target.replace("brand:", "")
                abandoned_by_brand[b_name] = abandoned_by_brand.get(b_name, 0) + data["weight"]

        ranked = sorted(abandoned_by_brand.items(), key=lambda x: x, reverse=True)[:5]
        return {
            "category": category.lower(),
            "top_abandoned_brand": ranked if ranked else None,
            "top_abandoned_cart_count": ranked if ranked else 0,
            "top_5_abandoned_brands": [{"brand": b, "abandoned_carts": w} for b, w in ranked],
        }

    def get_category_revenue_and_funnel(self, category: str) -> Dict[str, Any]:
        """Aggregates purchase revenue and 1-click direct-buy share across category purchase edges."""
        src = f"category:{category.lower()}"
        if src not in self.graph:
            return {"error": f"Category '{category}' not found in ontology."}

        rev_total = 0.0
        direct_cnt = 0
        std_cnt = 0

        for _, _, data in self.graph.out_edges(src, data=True):
            rel = data["relation_type"]
            if rel == "BYPASSED_CART_DIRECT_BUY":
                direct_cnt += data["weight"]
                rev_total += data["revenue_impact"]
            elif rel == "PURCHASED_BRAND":
                std_cnt += data["weight"]
                rev_total += data["revenue_impact"]

        total_buys = direct_cnt + std_cnt
        return {
            "category": category.lower(),
            "total_purchase_revenue": round(rev_total, 2),
            "direct_buy_sessions": direct_cnt,
            "standard_purchase_sessions": std_cnt,
            "direct_buy_share_pct": round((direct_cnt / total_buys * 100) if total_buys > 0 else 0.0, 2),
        }


# ==============================================================================
# 2. Four-Tier Agent Orchestrator
# ==============================================================================
class ThreeTierBenchmarkAgents:
    """Runs identical diagnostic questions across Tiers 1, 2, 3, and 4 for controlled ablation."""

    def __init__(
        self,
        spark,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        base_url: Optional[str] = None,
    ):
        self.spark = spark
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ontology = OntologyGraphEngine(spark)
        self.retriever = GoldenQueryRetriever()

    def _call_llm_with_retry(self, **kwargs):
        """Wraps chat completions with exponential backoff on HTTP 429 rate-limit errors."""
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                else:
                    raise

    def _run_spark_sql(self, sql_query: str) -> str:
        """Executes LLM-generated Spark SQL with a 45-second timeout guardrail against runaway self-joins."""
        clean_sql = (
            sql_query.strip()
            .removeprefix("```sql")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        self.spark.sql("SET STATEMENT_TIMEOUT = 45")
        try:
            rows = self.spark.sql(clean_sql).limit(10).collect()
            return json.dumps([r.asDict() for r in rows], default=str)
        finally:
            self.spark.sql("SET STATEMENT_TIMEOUT = 0")

    def _ontology_tools_schema(self):
        """OpenAI-compatible function calling definitions shared identically by Tier 3 and Tier 4."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "trace_brand_leakage",
                    "description": "Finds top competitor brands users switch TO when abandoning a source brand.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "category": {"type": "string"},
                        },
                        "required": ["brand"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_category_friction",
                    "description": "Finds top abandoned brand and cart friction counts in a category.",
                    "parameters": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_category_revenue_and_funnel",
                    "description": "Returns purchase revenue and direct-buy funnel share % for a category.",
                    "parameters": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                    },
                },
            },
        ]

    def _execute_ontology_tool(self, fn_name: str, fn_args: dict) -> dict:
        """Dispatches tool calls to the deterministic NetworkX OntologyGraphEngine."""
        if fn_name == "trace_brand_leakage":
            return self.ontology.trace_brand_leakage(**fn_args)
        elif fn_name == "analyze_category_friction":
            return self.ontology.analyze_category_friction(**fn_args)
        return self.ontology.get_category_revenue_and_funnel(**fn_args)

    # --------------------------------------------------------------------------
    # TIER 1: Raw Bronze Event Logs (Stateless Text-to-SQL Baseline)
    # --------------------------------------------------------------------------
    def run_tier1_raw_logs(self, query_id: str, question: str) -> BenchmarkResponse:
        t0 = time.perf_counter()
        prompt = f"""You are a SQL analyst querying raw event logs in `workspace.default.bronze_events`.
Columns: event_ts, event_type (view, cart, purchase), product_id, category_code, brand, price, user_id, user_session.
Write a single valid Spark SQL query to answer: "{question}"
Return ONLY raw SQL without markdown."""
        try:
