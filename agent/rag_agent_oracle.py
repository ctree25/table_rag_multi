# coding=utf-8
"""Single-table DataBench oracle agent.

Oracle means:
- the input record already provides the gold table_id and table_path;
- no outer table retrieval / candidate selection is performed;
- original TableRAG query expansion + schema/cell retrieval are still used;
- reasoning operates on one dataframe named `df`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agent.rag_agent import TableRAGAgent
from prompts.wtq import (
    tablerag_databench_oracle_solve_table_prompt,
    tablerag_cmoney_oracle_solve_table_prompt,
)
from utils.utils import infer_dtype


class TableRAGOracleAgent(TableRAGAgent):
    """TableRAG with the gold DataBench table given directly."""

    def run(self, data: dict, sc_id: int = 0) -> dict:
        log_path = os.path.join(
            self.log_dir,
            "log",
            f'{data["id"]}-{sc_id}.json',
        )

        if os.path.exists(log_path) and self.load_exist:
            with open(log_path, encoding="utf-8") as fp:
                return json.load(fp)

        if self.verbose:
            print(
                "=" * 25
                + f' {data["id"]} '
                + "=" * 25
            )

        query = data.get("statement") or data["question"]

        dataset_name = str(
            data.get("dataset", "")
        ).strip().lower()


        # ============================================================
        # CMoney Oracle
        # ============================================================
        if dataset_name == "cmoney":
            table_path = Path(data["table_path"])

            # JSONL intentionally only stores table_path.
            # Derive a stable internal ID for TableRAG BM25 cache.
            table_id = f"cmoney_oracle_{table_path.stem}"

            table_caption = table_path.stem

            solve_prompt_template = (
                tablerag_cmoney_oracle_solve_table_prompt
            )


        # ============================================================
        # DataBench Oracle
        # ============================================================
        else:
            gold_table_id = str(data["gold_table_id"])

            gold_candidate = next(
                (
                    candidate
                    for candidate in data["candidate_tables"]
                    if str(candidate["table_id"]) == gold_table_id
                ),
                None,
            )

            if gold_candidate is None:
                raise ValueError(
                    f'Gold table "{gold_table_id}" not found '
                    f'in candidate_tables for {data["id"]}'
                )

            table_id = gold_table_id

            table_caption = (
                gold_candidate.get("title")
                or gold_table_id
            )

            table_path = Path(
                gold_candidate["table_path"]
            )

            solve_prompt_template = (
                tablerag_databench_oracle_solve_table_prompt
            )


        if self.verbose:
            print(f"Oracle dataset: {dataset_name or 'databench'}")
            print(f"Oracle table_id: {table_id}")
            print(f"Oracle table_path: {table_path}")
        if not table_path.exists():
            raise FileNotFoundError(
                f"Oracle gold table not found: {table_path}"
            )

        # The JSONL already points directly to the gold table.
        df = pd.read_parquet(table_path)
        df = infer_dtype(df)

        # Original TableRAG inner retrieval on the gold table.
        self.retriever.init_retriever(table_id, df)

        column_prompt = self.get_prompt(
            "extract_column_prompt",
            table_caption=table_caption,
            query=query,
        )
        (
            schema_retrieval_result,
            column_queries,
            retrieved_columns,
        ) = self.retrieve_schema_by_prompt(column_prompt)

        cell_prompt = self.get_prompt(
            "extract_cell_prompt",
            table_caption=table_caption,
            query=query,
        )
        (
            cell_retrieval_result,
            cell_queries,
            retrieved_cells,
        ) = self.retrieve_cell_by_prompt(cell_prompt)

        prompt = solve_prompt_template.format(
            query=query,
            schema_retrieval_result=schema_retrieval_result,
            cell_retrieval_result=cell_retrieval_result,
        )

        init_prompt_token_count = self.model.get_token_count(
            prompt
        )

        answer, n_iter, solution = self.solver_loop(
            df,
            prompt,
        )

        result = {
            "id": data["id"],
            "sc_id": sc_id,
            "table_id": table_id,
            "table_caption": table_caption,
            "table_path": str(table_path),
            "query": query,
            "solution": solution,
            "answer": answer,
            "label": data["label"],
            "n_iter": n_iter,
            "init_prompt_token_count": init_prompt_token_count,
            "input_token_count": self.total_input_token_count,
            "output_token_count": self.total_output_token_count,
            "total_token_count": self.total_token_count,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "column_queries": column_queries,
            "cell_queries": cell_queries,
            "retrieved_columns": retrieved_columns,
            "retrieved_cells": retrieved_cells,
            "oracle": True,
        }

        if "databench_qa_index" in data:
            result["databench_qa_index"] = data[
                "databench_qa_index"
            ]

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                result,
                fp,
                ensure_ascii=False,
                indent=4,
            )

        with open(
            log_path.replace(".json", ".txt"),
            "w",
            encoding="utf-8",
        ) as fp:
            fp.write(prompt + solution)

        return result
