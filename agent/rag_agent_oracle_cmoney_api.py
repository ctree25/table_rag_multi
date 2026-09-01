# coding=utf-8
"""Single-table CMoney oracle agent using schema retrieval + SQL API actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.rag_agent_multi_cmoney_api import (
    TableRAGMultiCMoneyAPIAgent,
    unique_keep_order,
)
from prompts.cmoney_api import CMONEY_SCHEMA_SQL_ORACLE_SOLVE_PROMPT
from prompts.wtq import tablerag_extract_column_prompt


class TableRAGOracleCMoneyAPIAgent(TableRAGMultiCMoneyAPIAgent):
    """CMoney SQL-API oracle with the gold table given directly."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("candidate_k", None)
        super().__init__(candidate_k=1, **kwargs)

    def _build_oracle_schema_evidence(
        self,
        *,
        table_name: str,
        column_queries: list[str],
    ) -> tuple[str, dict[str, list[str]], list[str]]:
        docs = self.schema_retriever.retrieve_for_table(
            table_name=table_name,
            queries=column_queries,
        )

        retrieved_columns = {table_name: docs}
        missing_schema_tables: list[str] = []

        if not self.schema_retriever.has_table(table_name):
            missing_schema_tables.append(table_name)

        table_evidence = "\n".join(
            [
                f"Table: {table_name}",
                "Schema Retrieval Queries: " + ", ".join(column_queries),
                "Schema Retrieval Results:",
                "\n".join(docs) if docs else "(schema not found)",
            ]
        )

        return (
            table_evidence,
            retrieved_columns,
            missing_schema_tables,
        )

    def run(self, data: dict, sc_id: int = 0) -> dict:
        log_path = os.path.join(
            self.log_dir,
            "log",
            f'{data["id"]}-{sc_id}.json',
        )

        if os.path.exists(log_path) and self.load_exist:
            with open(log_path, encoding="utf-8") as fp:
                return json.load(fp)

        for counter_name in [
            "total_input_token_count",
            "total_output_token_count",
            "total_reasoning_token_count",
            "total_api_token_count",
            "total_token_count",
        ]:
            if hasattr(self, counter_name):
                setattr(self, counter_name, 0)

        if self.verbose:
            print("=" * 25 + f' {data["id"]} ' + "=" * 25)

        query = str(data["question"])

        gt_tables = unique_keep_order(
            [
                str(name).strip()
                for name in data.get("gt_tables", [])
                if str(name).strip()
            ]
        )

        if not gt_tables:
            raise ValueError(
                f'No gt_tables for Oracle question {data["id"]}'
            )

        # CMoney questions are treated as single-gold-table questions.
        # If multiple acceptable GT tables are listed, use the first one.
        oracle_table = gt_tables[0]

        if self.verbose:
            print(f"Oracle table: {oracle_table}")

        column_prompt = tablerag_extract_column_prompt.format(
            query=query
        )

        column_queries = self.generate_query_list_cached(
            prompt=column_prompt,
            remove_numeric=False,
            expansion_type="cmoney_column",
        )

        if not column_queries:
            column_queries = [query]

        (
            table_evidence,
            retrieved_columns,
            missing_schema_tables,
        ) = self._build_oracle_schema_evidence(
            table_name=oracle_table,
            column_queries=column_queries,
        )

        prompt = CMONEY_SCHEMA_SQL_ORACLE_SOLVE_PROMPT.format(
            query=query,
            table_name=oracle_table,
            table_evidence=table_evidence,
        )
        prompt = self._apply_qwen_sql_react_instruction(
            prompt
        )

        init_prompt_token_count = self.model.get_token_count(prompt)

        (
            answer,
            table_source,
            n_iter,
            solution,
            raw_final,
            api_calls,
        ) = self.solver_loop_sql_api(
            candidate_tables=[oracle_table],
            prompt=prompt,
        )

        selected_rank = 1 if table_source == oracle_table else None

        result: dict[str, Any] = {
            "id": data["id"],
            "question_id": data.get("question_id", data["id"]),
            "sc_id": sc_id,
            "query": query,
            "answer": answer,
            "raw_final_answer": raw_final,
            "table_source": table_source,
            "oracle_table": oracle_table,
            "selected_rank": selected_rank,
            "gt_tables": gt_tables,
            "selected_is_gold": (
                table_source in gt_tables
                if table_source is not None
                else False
            ),
            "gold_in_candidates": True,
            "candidate_k": 1,
            "candidate_tables": [oracle_table],
            "column_queries": column_queries,
            "retrieved_columns": retrieved_columns,
            "missing_schema_tables": missing_schema_tables,
            "api_calls": api_calls,
            "solution": solution,
            "n_iter": n_iter,
            "init_prompt_token_count": init_prompt_token_count,
            "input_token_count": getattr(
                self, "total_input_token_count", 0
            ),
            "output_token_count": getattr(
                self, "total_output_token_count", 0
            ),
            "total_token_count": getattr(
                self, "total_token_count", 0
            ),
            "oracle": True,
        }

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)

        with open(
            log_path.replace(".json", ".txt"),
            "w",
            encoding="utf-8",
        ) as fp:
            fp.write(prompt + solution)

        return result
