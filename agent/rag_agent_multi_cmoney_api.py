"""CMoney multi-table TableRAG agent using schema retrieval + SQL API actions."""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from agent.cmoney_schema_retriever import CMoneySchemaRetriever
from agent.rag_agent_multi import TableRAGMultiAgent, format_databench_answer
from prompts.cmoney_api import CMONEY_SCHEMA_SQL_SOLVE_PROMPT
from prompts.wtq import tablerag_extract_column_prompt
from table_query import query_table_light


_BANNED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|exec|execute|"
    r"grant|revoke|backup|restore)\b",
    flags=re.IGNORECASE,
)


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


class TableRAGMultiCMoneyAPIAgent(TableRAGMultiAgent):
    """Use candidate table names + schema CSV, then reason through SQL API."""

    def __init__(
        self,
        *,
        schema_path: str,
        candidate_k: int = 5,
        schema_top_k: int = 5,
        api_timeout: float = 30.0,
        observation_max_rows: int = 50,
        observation_max_chars: int = 12000,
        schema_embed_device: str = "cpu",
        schema_embed_batch_size: int = 64,
        schema_encode_dim: int = 1024,
        **kwargs,
    ) -> None:
        # Parent init is reused only for Model/query/token accounting/query-expansion
        # cache. Force its UNUSED original Retriever to BM25 so it never tries to
        # load Jina through HuggingFaceEmbeddings. The real schema retriever below
        # owns Jina v3 dense retrieval.
        requested_retrieve_mode = str(kwargs.get("retrieve_mode", "hybrid"))
        requested_embed_model = str(
            kwargs.get("embed_model_name", "jinaai/jina-embeddings-v3")
        )
        parent_kwargs = dict(kwargs)
        parent_kwargs["retrieve_mode"] = "bm25"
        parent_kwargs["embed_model_name"] = requested_embed_model
        super().__init__(candidate_k=candidate_k, **parent_kwargs)

        # Restore the actual CMoney schema-retrieval configuration after parent
        # initialization. self.retriever created by the parent is intentionally
        # unused in this agent.
        self.retrieve_mode = requested_retrieve_mode
        self.embed_model_name = requested_embed_model

        self.schema_path = str(schema_path)
        self.schema_top_k = int(schema_top_k)
        self.api_timeout = float(api_timeout)
        self.observation_max_rows = int(observation_max_rows)
        self.observation_max_chars = int(observation_max_chars)

        schema_db_dir = str(Path(self.db_dir or "db") / "cmoney_schema_only")
        self.schema_retriever = CMoneySchemaRetriever(
            schema_path=self.schema_path,
            mode=self.retrieve_mode,
            embed_model_name=self.embed_model_name,
            top_k=self.schema_top_k,
            db_dir=schema_db_dir,
            embed_device=schema_embed_device,
            embed_batch_size=schema_embed_batch_size,
            encode_dim=schema_encode_dim,
            verbose=self.verbose,
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
            elif text.lower().startswith("sql"):
                text = text[3:].strip()
        return text

    @classmethod
    def _parse_action(
        cls,
        action_text: str,
        candidate_tables: list[str],
    ) -> tuple[str, str]:
        """Parse raw SELECT SQL and infer the exact table from FROM."""
        sql = cls._strip_code_fence(action_text).strip()
        sql = sql.splitlines()[0].strip().strip('"').strip("'")

        match = re.search(
            r"\bFROM\s+(?:\[([^\]]+)\]|([^\s;,]+))",
            sql,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError("Cannot determine table name from SQL FROM clause.")

        table_name = (match.group(1) or match.group(2)).strip()
        if table_name not in candidate_tables:
            raise ValueError(
                f"Table {table_name!r} is not in candidate tables: "
                f"{candidate_tables}"
            )

        return table_name, sql

    @staticmethod
    def _validate_select_sql(
        *,
        table_name: str,
        sql: str,
        candidate_tables: list[str],
    ) -> None:
        if table_name not in candidate_tables:
            raise ValueError(
                f"Table {table_name!r} is not in the candidate table list."
            )

        normalized = sql.strip()
        if not re.match(r"^select\b", normalized, flags=re.IGNORECASE):
            raise ValueError("Only SELECT statements are allowed.")

        if _BANNED_SQL.search(normalized):
            raise ValueError("Unsafe/non-read-only SQL keyword detected.")

        # Permit one optional trailing semicolon, but reject stacked statements.
        without_trailing = normalized[:-1] if normalized.endswith(";") else normalized
        if ";" in without_trailing:
            raise ValueError("Only one SQL statement is allowed per Action.")

        if table_name not in normalized and f"[{table_name}]" not in normalized:
            raise ValueError(
                "SQL must reference the same candidate table named in Action.table."
            )

    def _format_observation(
        self,
        *,
        table_name: str,
        sql: str,
        rows: list[dict[str, Any]],
        headers: list[str],
    ) -> str:
        shown_rows = rows[: self.observation_max_rows]
        payload: dict[str, Any] = {
            "table": table_name,
            "sql": sql,
            "rows_count": len(rows),
            "columns": headers,
            "rows_shown": len(shown_rows),
            "truncated": len(rows) > len(shown_rows),
            "rows": shown_rows,
        }
        if payload["truncated"]:
            payload["note"] = (
                "Only a preview is shown. Issue a narrower SQL query or an "
                "aggregate query before answering."
            )

        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self.observation_max_chars:
            return text

        # Reduce row preview until the observation fits. Keep metadata intact.
        reduced = shown_rows
        while reduced and len(text) > self.observation_max_chars:
            reduced = reduced[: max(1, len(reduced) // 2)]
            payload["rows"] = reduced
            payload["rows_shown"] = len(reduced)
            payload["truncated"] = True
            payload["note"] = (
                "Observation was truncated for context length. Issue a "
                "narrower SQL query or aggregate in SQL."
            )
            text = json.dumps(payload, ensure_ascii=False, default=str)

        if len(text) > self.observation_max_chars:
            # Last resort: return headers/counts but no potentially cut JSON row.
            payload["rows"] = []
            payload["rows_shown"] = 0
            text = json.dumps(payload, ensure_ascii=False, default=str)
        return text

    def _build_schema_evidence(
        self,
        *,
        candidate_tables: list[str],
        column_queries: list[str],
    ) -> tuple[str, dict[str, list[str]], list[str]]:
        sections: list[str] = []
        retrieved_columns: dict[str, list[str]] = {}
        missing_schema_tables: list[str] = []

        for rank, table_name in enumerate(candidate_tables, start=1):
            docs = self.schema_retriever.retrieve_for_table(
                table_name=table_name,
                queries=column_queries,
            )
            retrieved_columns[table_name] = docs
            if not self.schema_retriever.has_table(table_name):
                missing_schema_tables.append(table_name)

            sections.append(
                "\n".join(
                    [
                        "=" * 72,
                        f"Candidate rank={rank} | table={table_name}",
                        "Schema Retrieval Queries: "
                        + ", ".join(column_queries),
                        "Schema Retrieval Results:",
                        "\n".join(docs) if docs else "(schema not found)",
                    ]
                )
            )

        return "\n\n".join(sections), retrieved_columns, missing_schema_tables

    def solver_loop_sql_api(
        self,
        *,
        candidate_tables: list[str],
        prompt: str,
    ) -> tuple[str, str | None, int, str, str, list[dict[str, Any]]]:
        """ReAct loop where Action executes SQL via QueryTableLight."""
        if self.verbose:
            print(prompt, end="")

        n_iter = self.max_depth
        solution = ""
        init_prompt = prompt
        text = ""
        api_calls: list[dict[str, Any]] = []

        for i in range(self.max_depth):
            solution += "Thought: "
            current_prompt = init_prompt + solution
            text = self.query(current_prompt).strip()
            solution += text

            if self.verbose:
                print("Thought: " + text)

            terminal_payload, _ = self.parse_multi_final(text)
            if terminal_payload is not None or self.is_terminal(text):
                n_iter = i + 1
                break

            if "Action:" not in text:
                observation = "Error: no Action provided."
            else:
                raw_action = text.split("Action:", 1)[1].strip().splitlines()[0].strip()
                call_log: dict[str, Any] = {"raw_action": raw_action}

                try:
                    table_name, sql = self._parse_action(
                        raw_action,
                        candidate_tables,
                    )
                    self._validate_select_sql(
                        table_name=table_name,
                        sql=sql,
                        candidate_tables=candidate_tables,
                    )
                    call_log.update({"table": table_name, "sql": sql})

                    rows, headers = query_table_light(
                        sql,
                        [table_name],
                        timeout=self.api_timeout,
                    )
                    call_log.update(
                        {
                            "status": "ok",
                            "rows_count": len(rows),
                            "cols_count": len(headers),
                            "headers": headers,
                        }
                    )
                    observation = self._format_observation(
                        table_name=table_name,
                        sql=sql,
                        rows=rows,
                        headers=headers,
                    )
                    call_log["observation"] = observation
                except Exception as exc:
                    call_log.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    observation = (
                        f"SQL/API Error: {type(exc).__name__}: {exc}. "
                        "Revise the SQL or try another candidate table."
                    )

                api_calls.append(call_log)

            if "\n" in str(observation):
                observation = "\n" + str(observation)
            solution += f"\nObservation: {observation}\n"

            if self.verbose:
                print(f"Observation: {observation}")

        payload, raw_final = self.parse_multi_final(text)
        table_source = None
        answer_value: Any = raw_final
        if payload is not None:
            table_source = payload.get("table_source")
            answer_value = payload.get("answer", "")

        answer = format_databench_answer(answer_value)
        return answer, table_source, n_iter, solution, raw_final, api_calls

    def run(self, data: dict, sc_id: int = 0) -> dict:
        log_path = os.path.join(
            self.log_dir,
            "log",
            f'{data["id"]}-{sc_id}.json',
        )
        if os.path.exists(log_path) and self.load_exist:
            with open(log_path, encoding="utf-8") as fp:
                return json.load(fp)

        # This runner reuses one agent across questions; reset per-question
        # usage counters so each log remains comparable with the original run.py.
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
        candidate_tables = [
            str(name).strip()
            for name in data.get("candidate_tables", [])[: self.candidate_k]
            if str(name).strip()
        ]
        candidate_tables = unique_keep_order(candidate_tables)
        if not candidate_tables:
            raise ValueError(f'No candidate tables for {data["id"]}')

        # Same TableRAG idea: expand the question into likely column names once,
        # then run those schema queries independently against each candidate.
        column_prompt = tablerag_extract_column_prompt.format(query=query)
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
        ) = self._build_schema_evidence(
            candidate_tables=candidate_tables,
            column_queries=column_queries,
        )

        candidate_table_map = "\n".join(
            f"rank={rank}: {table_name}"
            for rank, table_name in enumerate(candidate_tables, start=1)
        )

        prompt = CMONEY_SCHEMA_SQL_SOLVE_PROMPT.format(
            query=query,
            candidate_table_map=candidate_table_map,
            table_evidence=table_evidence,
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
            candidate_tables=candidate_tables,
            prompt=prompt,
        )

        selected_rank = None
        if table_source in candidate_tables:
            selected_rank = candidate_tables.index(table_source) + 1

        gt_tables = [str(x) for x in data.get("gt_tables", [])]
        result = {
            "id": data["id"],
            "question_id": data.get("question_id", data["id"]),
            "sc_id": sc_id,
            "query": query,
            "answer": answer,
            "raw_final_answer": raw_final,
            "table_source": table_source,
            "selected_rank": selected_rank,
            "gt_tables": gt_tables,
            "selected_is_gold": (
                table_source in gt_tables if table_source is not None else False
            ),
            "gold_in_candidates": any(
                table in candidate_tables for table in gt_tables
            ),
            "candidate_k": len(candidate_tables),
            "candidate_tables": candidate_tables,
            "column_queries": column_queries,
            "retrieved_columns": retrieved_columns,
            "missing_schema_tables": missing_schema_tables,
            "api_calls": api_calls,
            "solution": solution,
            "n_iter": n_iter,
            "init_prompt_token_count": init_prompt_token_count,
            "input_token_count": getattr(self, "total_input_token_count", 0),
            "output_token_count": getattr(self, "total_output_token_count", 0),
            "total_token_count": getattr(self, "total_token_count", 0),
        }

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        with open(log_path.replace(".json", ".txt"), "w", encoding="utf-8") as fp:
            fp.write(prompt + solution)

        return result
