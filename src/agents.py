"""
agents.py
---------
4-Tier Progressive Data & Prompt Architecture Benchmark:
  - Tier 1 (`Tier1_RawLogs`): Baseline Text-to-SQL over 42.45M `bronze_events` rows.
  - Tier 2 (`Tier2_SemanticLayer`): Text-to-SQL over stateful `silver_sessions`.
  - Tier 3 (`Tier3_Ontology`): Deterministic NetworkX graph traversal (no prompt engineering).
  - Tier 4 (`Tier4_Ontology_PromptEng`): Graph tools + Domain Axioms + TF-IDF Golden Retrieval.
"""

import json
import time
from typing import Dict, Any, Optional
import networkx as nx
from openai import OpenAI
from schemas import BenchmarkResponse
from golden_queries import GoldenQueryRetriever


class OntologyGraphEngine:
    """In-memory NetworkX MultiDiGraph built from `workspace.default.ont_edges`."""

    def __init__(self, spark):
        self.spark = spark
        self.graph = nx.MultiDiGraph()
        self._load_graph()

    def _load_graph(self):
        """Materializes all weighted relationship edges from Delta into NetworkX."""
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
        """Traverses outgoing `SWITCHED_BRAND_TO` edges from `brand` to rank top competitors."""
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

        ranked = sorted(competitors.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "source_brand": brand.lower(),
            "category_filter": category or "all",
            "top_switched_to_brands": [{"brand": b, "switch_sessions": w} for b, w in ranked],
            "top_competitor": ranked[0][0] if ranked else None,
            "top_competitor_switches": ranked[0][1] if ranked else 0,
        }

    def analyze_category_friction(self, category: str) -> Dict[str, Any]:
        """Traverses `ABANDONED_BRAND_CART` edges from a category node to find top friction brands."""
        src = f"category:{category.lower()}"
        if src not in self.graph:
            return {"error": f"Category '{category}' not found in ontology."}

        abandoned_by_brand: Dict[str, int] = {}
        for _, target, data in self.graph.out_edges(src, data=True):
            if data["relation_type"] == "ABANDONED_BRAND_CART":
                b_name = target.replace("brand:", "")
                abandoned_by_brand[b_name] = abandoned_by_brand.get(b_name, 0) + data["weight"]

        ranked = sorted(abandoned_by_brand.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "category": category.lower(),
            "top_abandoned_brand": ranked[0][0] if ranked else None,
            "top_abandoned_cart_count": ranked[0][1] if ranked else 0,
            "top_5_abandoned_brands": [{"brand": b, "abandoned_carts": w} for b, w in ranked],
        }

    def get_category_revenue_and_funnel(self, category: str) -> Dict[str, Any]:
        """Aggregates known-brand purchase revenue and 1-click direct-buy share % for a category."""
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

    # def _call_llm_with_retry(self, **kwargs):
    #     """Retries up to 3 times on HTTP 429 rate-limit errors with exponential backoff."""
    #     for attempt in range(3):
    #         try:
    #             return self.client.chat.completions.create(**kwargs)
    #         except Exception as e:
    #             if "429" in str(e) and attempt < 2:
    #                 time.sleep(15 * (attempt + 1))
    #             else:
    #                 raise

    def _call_llm_with_retry(self, **kwargs):
     """Retries up to 4 times on HTTP 429 rate-limit or 503 overload errors."""
     for attempt in range(4):
         try:
             return self.client.chat.completions.create(**kwargs)
         except Exception as e:
             err = str(e)
             if ("429" in err or "503" in err) and attempt < 3:
                 time.sleep(25 * (attempt + 1))
             else:
                 raise


    # def _run_spark_sql(self, sql_query: str) -> str:
    #     """Executes LLM-generated Spark SQL with a 45-second timeout guardrail."""
    #     clean_sql = (
    #         sql_query.strip()
    #         .removeprefix("```sql")
    #         .removeprefix("```")
    #         .removesuffix("```")
    #         .strip()
    #     )
    #     self.spark.sql("SET STATEMENT_TIMEOUT = 45")
    #     try:
    #         rows = self.spark.sql(clean_sql).limit(10).collect()
    #         return json.dumps([r.asDict() for r in rows], default=str)
    #     finally:
    #         self.spark.sql("SET STATEMENT_TIMEOUT = 0")
    
    def _run_spark_sql(self, sql_query: str) -> str:
     """Executes LLM-generated Spark SQL and returns compact JSON rows."""
     clean_sql = (
         sql_query.strip()
         .removeprefix("```sql")
         .removeprefix("```")
         .removesuffix("```")
         .strip()
     )
     rows = self.spark.sql(clean_sql).limit(10).collect()
     return json.dumps([r.asDict() for r in rows], default=str)


    def _ontology_tools_schema(self):
        """Function-calling schemas shared identically by Tier 3 and Tier 4."""
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
        """Routes validated tool arguments to the corresponding OntologyGraphEngine method."""
        if fn_name == "trace_brand_leakage":
            return self.ontology.trace_brand_leakage(**fn_args)
        elif fn_name == "analyze_category_friction":
            return self.ontology.analyze_category_friction(**fn_args)
        return self.ontology.get_category_revenue_and_funnel(**fn_args)

    # def run_tier1_raw_logs(self, query_id: str, question: str) -> BenchmarkResponse:
    #     """Tier 1 Baseline: Stateless Text-to-SQL over 42.45M bronze_events rows."""
    #     t0 = time.perf_counter()
    #     prompt = (
    #         "You are a SQL analyst querying raw event logs in `workspace.default.bronze_events`.\n"
    #         "Columns: event_ts, event_type (view, cart, purchase), product_id, category_code, brand, price, user_id, user_session.\n"
    #         f'Write a single valid Spark SQL query to answer: "{question}"\n'
    #         "Return ONLY raw SQL without markdown."
    #     )
    #     try:
    #         resp = self._call_llm_with_retry(
    #             model=self.model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
    #         )
    #         sql_code = resp.choices.message.content
    #         in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
    #         sql_result = self._run_spark_sql(sql_code)

    #         extract_prompt = (
    #             f"Question: {question}\n"
    #             f"SQL Output: {sql_result}\n"
    #             'Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), '
    #             '"entity_answer" (string or null), "root_cause_explanation" (string).'
    #         )
    #         resp2 = self._call_llm_with_retry(
    #             model=self.model,
    #             temperature=0.0,
    #             response_format={"type": "json_object"},
    #             messages=[{"role": "user", "content": extract_prompt}],
    #         )
    #         parsed = json.loads(resp2.choices.message.content)
    #         in_tok += resp2.usage.prompt_tokens
    #         out_tok += resp2.usage.completion_tokens

    #         return BenchmarkResponse(
    #             tier="Tier1_RawLogs",
    #             query_id=query_id,
    #             numeric_answer=parsed.get("numeric_answer"),
    #             entity_answer=parsed.get("entity_answer"),
    #             root_cause_explanation=parsed.get("root_cause_explanation", ""),
    #             generated_code_or_tool=sql_code,
    #             prompt_tokens=in_tok,
    #             completion_tokens=out_tok,
    #             latency_sec=round(time.perf_counter() - t0, 3),
    #         )
    #     except Exception as e:
    #         return BenchmarkResponse(
    #             tier="Tier1_RawLogs",
    #             query_id=query_id,
    #             root_cause_explanation="Execution failed",
    #             generated_code_or_tool="ERROR",
    #             latency_sec=round(time.perf_counter() - t0, 3),
    #             execution_error=str(e)[:300],
    #         )

    def run_tier1_raw_logs(self, query_id: str, question: str) -> BenchmarkResponse:
        """Tier 1 Baseline: Stateless Text-to-SQL over 42.45M bronze_events rows."""
        t0 = time.perf_counter()
        prompt = (
            "You are a SQL analyst querying raw event logs in `workspace.default.bronze_events`.\n"
            "Columns: event_ts, event_type (view, cart, purchase), product_id, category_code, brand, price, user_id, user_session.\n"
            f'Write a single valid Spark SQL query to answer: "{question}"\n'
            "Return ONLY raw SQL without markdown."
        )
        try:
            resp = self._call_llm_with_retry(
                model=self.model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
            )
            sql_code = resp.choices[0].message.content
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            sql_result = self._run_spark_sql(sql_code)

            extract_prompt = (
                f"Question: {question}\n"
                f"SQL Output: {sql_result}\n"
                'Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), '
                '"entity_answer" (string or null), "root_cause_explanation" (string).'
            )
            resp2 = self._call_llm_with_retry(
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
        """Tier 2 Semantic Layer: Text-to-SQL over stateful, bot-filtered silver_sessions."""
        t0 = time.perf_counter()
        prompt = (
            "You are a SQL analyst querying modeled session data in `workspace.default.silver_sessions`.\n"
            "Columns: clean_session_id, user_id, duration_sec, total_events, view_count, cart_count, purchase_count, "
            "brand_switch_count, price_step_down_count, primary_category, primary_brand, final_brand, avg_price_viewed, "
            "session_revenue, is_bot_suspect (boolean), funnel_outcome (direct_purchase_no_cart, standard_purchase, "
            "cart_abandoned, single_event_bounce, multi_view_browse).\n"
            "Rules:\n"
            "- Always filter `WHERE is_bot_suspect = false`.\n"
            "- Purchasing sessions are rows where `purchase_count > 0`.\n"
            "- When asked for a percentage, return a 0-100 number (e.g., 54.25, NOT 0.5425).\n"
            f'Write a single valid Spark SQL query to answer: "{question}"\n'
            "Return ONLY raw SQL without markdown."
        )
        try:
            resp = self._call_llm_with_retry(
                model=self.model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
            )
            sql_code = resp.choices[0].message.content
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            sql_result = self._run_spark_sql(sql_code)

            extract_prompt = (
                f"Question: {question}\n"
                f"SQL Output: {sql_result}\n"
                'Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), '
                '"entity_answer" (string or null), "root_cause_explanation" (string).'
            )
            resp2 = self._call_llm_with_retry(
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
        """Tier 3 Bare Ontology: Graph traversal tools with zero domain prompt engineering."""
        t0 = time.perf_counter()
        try:
            resp = self._call_llm_with_retry(
                model=self.model,
                temperature=0.0,
                tools=self._ontology_tools_schema(),
                tool_choice="auto",
                messages=[{"role": "user", "content": question}],
            )
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            msg = resp.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                fn_name, fn_args = tc.function.name, json.loads(tc.function.arguments)
                tool_out = self._execute_ontology_tool(fn_name, fn_args)
                tool_trace = f"{fn_name}({fn_args})"
            else:
                tool_out = {"note": "No tool invoked"}
                tool_trace = "no_tool_invoked"

            synth_prompt = (
                f"Question: {question}\n"
                f"Ontology Output: {json.dumps(tool_out)}\n"
                'Extract the final answer strictly as JSON with keys: "numeric_answer" (float or null), '
                '"entity_answer" (string or null), "root_cause_explanation" (string).'
            )
            resp2 = self._call_llm_with_retry(
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

    def run_tier4_ontology_prompt_eng(self, query_id: str, question: str) -> BenchmarkResponse:
        """Tier 4: Ontology Graph Tools + Domain Axioms + Dynamic TF-IDF Golden Query Retrieval."""
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=2)
        golden_block = "\n".join(
            f'- User: "{ex["question"]}"\n  -> Call `{ex["tool_call"]}` ({ex["reasoning"]})'
            for ex in retrieved
        )

        system_prompt = (
            "You are a Principal Behavioral Data Architect querying an enterprise e-commerce Ontology.\n\n"
            "### DOMAIN AXIOMS (STRICT RULES)\n"
            "1. Bot traffic is already excluded from all ontology nodes and edges.\n"
            "2. Funnel Axiom: 1-click checkouts bypass the cart via `BYPASSED_CART_DIRECT_BUY`. "
            "Total conversions = `direct_buy_sessions + standard_purchase_sessions`.\n"
            "3. Competitive Leakage Axiom: Brand switching is modeled via directed `SWITCHED_BRAND_TO` edges "
            "(`source_brand -> target_brand`).\n"
            "4. Always invoke the most specific ontology traversal tool first; never guess numbers.\n\n"
            "### RETRIEVED GOLDEN EXAMPLES (DYNAMICALLY MATCHED VIA TF-IDF)\n"
            f"{golden_block}"
        )

        try:
            resp = self._call_llm_with_retry(
                model=self.model,
                temperature=0.0,
                tools=self._ontology_tools_schema(),
                tool_choice="auto",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            msg = resp.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                fn_name, fn_args = tc.function.name, json.loads(tc.function.arguments)
                tool_out = self._execute_ontology_tool(fn_name, fn_args)
                tool_trace = f"{fn_name}({fn_args})"
            else:
                tool_out = {"error": "No ontology tool invoked"}
                tool_trace = "no_tool_invoked"

            synth_prompt = (
                "Extract verified benchmark metrics from the Ontology traversal output.\n"
                f"Question: {question}\n"
                f"Ontology Output: {json.dumps(tool_out)}\n\n"
                "Strict Extraction Rules:\n"
                "- `numeric_answer`: Must be the exact primary numeric metric answering the question "
                "(e.g., `top_competitor_switches`, `direct_buy_share_pct`, `top_abandoned_cart_count`, or `total_purchase_revenue`). "
                "Never leave null when a number exists.\n"
                "- `entity_answer`: Must be the exact lowercase brand or category name answering the question.\n"
                "- `root_cause_explanation`: One concise sentence citing exact numbers from Ontology Output.\n\n"
                'Return strictly valid JSON with keys: "numeric_answer", "entity_answer", "root_cause_explanation".'
            )

            resp2 = self._call_llm_with_retry(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": synth_prompt}],
            )
            parsed = json.loads(resp2.choices[0].message.content)
            in_tok += resp2.usage.prompt_tokens
            out_tok += resp2.usage.completion_tokens

            return BenchmarkResponse(
                tier="Tier4_Ontology_PromptEng",
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
                tier="Tier4_Ontology_PromptEng",
                query_id=query_id,
                root_cause_explanation="Execution failed",
                generated_code_or_tool="ERROR",
                latency_sec=round(time.perf_counter() - t0, 3),
                execution_error=str(e)[:300],
            )

