import json
import time
from typing import Dict, Any, Optional
import networkx as nx
from openai import OpenAI
from schemas import BenchmarkResponse


# ==============================================================================
# 1. Deterministic Ontology Graph Tools (Tier 3 Backend)
# ==============================================================================
class OntologyGraphEngine:
    """In-memory NetworkX graph built from workspace.default.ont_edges for fast causal traversal."""

    def __init__(self, spark):
        self.spark = spark
        self.graph = nx.MultiDiGraph()
        self._load_graph()

    def _load_graph(self):
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
        """Multi-hop traversal: finds top competitor brands users switch TO from `brand`."""
        src = f"brand:{brand.lower()}"
        if src not in self.graph:
            return {"error": f"Brand '{brand}' not found in ontology."}

        competitors = {}
        for _, target, data in self.graph.out_edges(src, data=True):
            if data["relation_type"] == "SWITCHED_BRAND_TO":
                if category and data["context_category"] != category.lower():
                    continue
                comp_name = target.replace("brand:", "")
                competitors[comp_name] = competitors.get(comp_name, 0) + data["weight"]

        ranked = sorted(competitors.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "source_brand": brand,
            "category_filter": category or "all",
            "top_switched_to_brands": [{"brand": b, "switch_sessions": w} for b, w in ranked],
            "top_competitor": ranked[0][0] if ranked else None,
            "top_competitor_switches": ranked[0][1] if ranked else 0,
        }

    def analyze_category_friction(self, category: str) -> Dict[str, Any]:
        """Traverses ABANDONED_BRAND_CART, BYPASSED_CART_DIRECT_BUY, and PURCHASED_BRAND links."""
        src = f"category:{category.lower()}"
        if src not in self.graph:
            return {"error": f"Category '{category}' not found in ontology."}

        summary = {"abandoned": 0, "direct_buy": 0, "standard_purchase": 0, "top_abandoned_brand": None}
        max_abandon = -1

        for _, target, data in self.graph.out_edges(src, data=True):
            rel = data["relation_type"]
            w = data["weight"]
            if rel == "ABANDONED_BRAND_CART":
                summary["abandoned"] += w
                if w > max_abandon:
                    max_abandon = w
                    summary["top_abandoned_brand"] = target.replace("brand:", "")
            elif rel == "BYPASSED_CART_DIRECT_BUY":
                summary["direct_buy"] += w
            elif rel == "PURCHASED_BRAND":
                summary["standard_purchase"] += w

        total_purchases = summary["direct_buy"] + summary["standard_purchase"]
        summary["direct_buy_share_pct"] = round(
            (summary["direct_buy"] / total_purchases * 100) if total_purchases > 0 else 0.0, 2
        )
        return summary


# ==============================================================================
# 2. Three-Tier Agent Orchestrator
# ==============================================================================
class ThreeTierBenchmarkAgents:
    """Executes the same diagnostic prompt across Tier 1 (Raw), Tier 2 (Semantic), and Tier 3 (Ontology)."""

    def __init__(self, spark, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.spark = spark
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ontology = OntologyGraphEngine(spark)

    def _run_spark_sql(self, sql_query: str) -> str:
        """Executes LLM-generated Spark SQL safely and returns compact JSON rows."""
        clean_sql = sql_query.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        rows = self.spark.sql(clean_sql).limit(10).collect()
        return json.dumps([r.asDict() for r in rows], default=str)

    def run_tier1_raw_logs(self, query_id: str, question: str) -> BenchmarkResponse:
        """Tier 1 Baseline: LLM writes raw Spark SQL directly against 42.4M stateless bronze_events."""
        t0 = time.perf_counter()
        prompt = f"""You are a SQL analyst querying raw event logs in table `workspace.default.bronze_events`.
Columns: event_ts (timestamp), event_type (string: view, cart, purchase), product_id (long), category_code (string, nullable), brand (string, nullable), price (double), user_id (long), user_session (string).
Write a single valid Spark SQL query to answer: "{question}"
Return ONLY raw SQL without markdown formatting."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
            )
            sql_code = resp.choices[0].message.content
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens

            sql_result = self._run_spark_sql(sql_code)

            extract_prompt = f"""Question: {question}
SQL Output: {sql_result}
Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), "entity_answer" (string or null), "root_cause_explanation" (string)."""
            resp2 = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": extract_prompt}],
            )
            parsed = json.loads(resp2.choices[0].message.content)
            in_tok += resp2.usage.prompt_tokens
            out_tok += resp2.usage.completion_tokens

            return BenchmarkResponse(
                tier="Tier1_RawLogs",
                query_id=query_id,
                numeric_answer=parsed.get("numeric_answer"),
                entity_answer=parsed.get("entity_answer"),
                root_cause_explanation=parsed.get("root_cause_explanation", ""),
                generated_code_or_tool=sql_code,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                latency_sec=round(time.perf_counter() - t0, 3),
            )
        except Exception as e:
            return BenchmarkResponse(
                tier="Tier1_RawLogs",
                query_id=query_id,
                root_cause_explanation="Execution failed",
                generated_code_or_tool="ERROR",
                latency_sec=round(time.perf_counter() - t0, 3),
                execution_error=str(e)[:300],
            )

    def run_tier2_semantic_layer(self, query_id: str, question: str) -> BenchmarkResponse:
        """Tier 2 Challenger: LLM queries modeled, bot-filtered stateful `silver_sessions`."""
        t0 = time.perf_counter()
        prompt = f"""You are a SQL analyst querying modeled session data in `workspace.default.silver_sessions`.
Columns: clean_session_id, user_id, duration_sec, total_events, view_count, cart_count, purchase_count, brand_switch_count, price_step_down_count, primary_category, primary_brand, final_brand, avg_price_viewed, session_revenue, is_bot_suspect (boolean), funnel_outcome (direct_purchase_no_cart, standard_purchase, cart_abandoned, single_event_bounce, multi_view_browse).
Always filter `WHERE is_bot_suspect = false`.
Write a single valid Spark SQL query to answer: "{question}"
Return ONLY raw SQL without markdown."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
            )
            sql_code = resp.choices[0].message.content
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens

            sql_result = self._run_spark_sql(sql_code)

            extract_prompt = f"""Question: {question}
SQL Output: {sql_result}
Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), "entity_answer" (string or null), "root_cause_explanation" (string)."""
            resp2 = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": extract_prompt}],
            )
            parsed = json.loads(resp2.choices[0].message.content)
            in_tok += resp2.usage.prompt_tokens
            out_tok += resp2.usage.completion_tokens

            return BenchmarkResponse(
                tier="Tier2_SemanticLayer",
                query_id=query_id,
                numeric_answer=parsed.get("numeric_answer"),
                entity_answer=parsed.get("entity_answer"),
                root_cause_explanation=parsed.get("root_cause_explanation", ""),
                generated_code_or_tool=sql_code,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                latency_sec=round(time.perf_counter() - t0, 3),
            )
        except Exception as e:
            return BenchmarkResponse(
                tier="Tier2_SemanticLayer",
                query_id=query_id,
                root_cause_explanation="Execution failed",
                generated_code_or_tool="ERROR",
                latency_sec=round(time.perf_counter() - t0, 3),
                execution_error=str(e)[:300],
            )

    def run_tier3_ontology(self, query_id: str, question: str) -> BenchmarkResponse:
        """Tier 3 Ontology Agent: Uses deterministic Python graph traversal tools over typed objects & links."""
        t0 = time.perf_counter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "trace_brand_leakage",
                    "description": "Finds top competitor brands users switch TO when abandoning a source brand.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "category": {"type": "string", "description": "Optional top-level category filter"},
                        },
                        "required": ["brand"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_category_friction",
                    "description": "Traverses category-to-brand edges for cart abandonment, 1-click direct purchases, and standard checkouts.",
                    "parameters": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                    },
                },
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                messages=[{"role": "user", "content": question}],
            )
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            msg = resp.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if fn_name == "trace_brand_leakage":
                    tool_out = self.ontology.trace_brand_leakage(**fn_args)
                else:
                    tool_out = self.ontology.analyze_category_friction(**fn_args)
                tool_trace = f"{fn_name}({fn_args})"
            else:
                tool_out = {"note": "Fallback to semantic layer"}
                tool_trace = "no_tool_invoked"

            synth_prompt = f"""Question: {question}
Ontology Graph Traversal Result: {json.dumps(tool_out)}
Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), "entity_answer" (string or null), "root_cause_explanation" (string)."""

            resp2 = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": synth_prompt}],
            )
            parsed = json.loads(resp2.choices[0].message.content)
            in_tok += resp2.usage.prompt_tokens
            out_tok += resp2.usage.completion_tokens

            return BenchmarkResponse(
                tier="Tier3_Ontology",
                query_id=query_id,
                numeric_answer=parsed.get("numeric_answer"),
                entity_answer=parsed.get("entity_answer"),
                root_cause_explanation=parsed.get("root_cause_explanation", ""),
                generated_code_or_tool=tool_trace,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                latency_sec=round(time.perf_counter() - t0, 3),
            )
        except Exception as e:
            return BenchmarkResponse(
                tier="Tier3_Ontology",
                query_id=query_id,
                root_cause_explanation="Execution failed",
                generated_code_or_tool="ERROR",
                latency_sec=round(time.perf_counter() - t0, 3),
                execution_error=str(e)[:300],
            )
