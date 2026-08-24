# coding=utf-8

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_oracle_examples(
    oracle_path: Path,
) -> dict[int, dict]:
    examples = {}

    with oracle_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if "databench_qa_index" in item:
                idx = int(item["databench_qa_index"])
            else:
                idx = int(
                    str(item["id"]).rsplit("-", 1)[-1]
                )

            examples[idx] = item

    return examples


def load_retrieval_file(
    retrieval_path: Path,
) -> list[Any]:
    """
    Supports:
    1. JSON:
       [
         [... ranked candidates for q0 ...],
         [... ranked candidates for q1 ...],
         ...
       ]

    2. JSON:
       [
         {"query_index": 0, "top_tables": [...]},
         ...
       ]

    3. JSONL:
       one dict per line
    """

    if retrieval_path.suffix.lower() == ".json":
        with retrieval_path.open(encoding="utf-8") as fp:
            obj = json.load(fp)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            if "results" in obj:
                return obj["results"]

            if "data" in obj:
                return obj["data"]

            raise ValueError(
                "Unsupported JSON dict format. "
                f"Keys: {list(obj.keys())[:20]}"
            )

        raise TypeError(
            f"Unsupported JSON top-level type: "
            f"{type(obj).__name__}"
        )

    records = []

    with retrieval_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def normalize_candidate(
    candidate: Any,
    rank: int,
) -> dict:
    """
    Normalize several common retrieval result formats.

    Supported examples:

    "066_IBM_HR"

    ["066_IBM_HR", 0.85]

    {
        "table_id": "066_IBM_HR",
        "score": 0.85
    }
    """

    if isinstance(candidate, str):
        return {
            "rank": rank,
            "table_id": candidate,
            "title": candidate,
            "score": None,
            "bm25_score": None,
            "dense_score": None,
        }

    if isinstance(candidate, dict):
        table_id = (
            candidate.get("table_id")
            or candidate.get("table_name")
            or candidate.get("dataset")
            or candidate.get("table")
            or candidate.get("id")
        )

        if table_id is None:
            raise ValueError(
                "Cannot find table ID in candidate dict: "
                f"{candidate}"
            )

        return {
            "rank": int(
                candidate.get("rank", rank)
            ),
            "table_id": str(table_id),
            "title": str(
                candidate.get(
                    "title",
                    candidate.get("caption", table_id),
                )
            ),
            "score": candidate.get("score"),
            "bm25_score": candidate.get(
                "bm25_score"
            ),
            "dense_score": candidate.get(
                "dense_score"
            ),
        }

    if isinstance(candidate, (list, tuple)):
        if len(candidate) == 0:
            raise ValueError(
                "Encountered empty candidate list."
            )

        table_id = candidate[0]

        score = (
            candidate[1]
            if len(candidate) >= 2
            and isinstance(
                candidate[1],
                (int, float),
            )
            else None
        )

        return {
            "rank": rank,
            "table_id": str(table_id),
            "title": str(table_id),
            "score": score,
            "bm25_score": None,
            "dense_score": None,
        }

    raise TypeError(
        "Unsupported candidate type: "
        f"{type(candidate).__name__}: {candidate}"
    )


def prepare_multi_data(
    retrieval_path: Path,
    oracle_path: Path,
    data_root: Path,
    output_path: Path,
    candidate_k: int,
) -> None:

    oracle_examples = load_oracle_examples(
        oracle_path
    )

    retrieval_records = load_retrieval_file(
        retrieval_path
    )

    print(
        "Retrieval records:",
        len(retrieval_records),
    )

    if len(retrieval_records) != len(
        oracle_examples
    ):
        raise ValueError(
            "Retrieval/oracle question count mismatch: "
            f"{len(retrieval_records)} vs "
            f"{len(oracle_examples)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_questions = 0
    gold_in_top_k_count = 0
    gold_found_total = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_fp:

        for position, retrieval in enumerate(
            retrieval_records
        ):
            # -----------------------------------------
            # Old details format:
            #
            # {
            #   "query_index": 0,
            #   "question": "...",
            #   "gold_table_id": "...",
            #   "gold_rank": 2,
            #   "top_tables": [...]
            # }
            # -----------------------------------------
            if isinstance(retrieval, dict):
                query_index = int(
                    retrieval.get(
                        "query_index",
                        position,
                    )
                )

                raw_candidates = (
                    retrieval.get("top_tables")
                    or retrieval.get("tables")
                    or retrieval.get("results")
                )

                if raw_candidates is None:
                    raise ValueError(
                        "Cannot find candidate list "
                        f"for query {query_index}"
                    )

            # -----------------------------------------
            # New opentab.json format:
            #
            # [
            #   candidate1,
            #   candidate2,
            #   ...
            # ]
            #
            # Outer list index = query index.
            # -----------------------------------------
            elif isinstance(retrieval, list):
                query_index = position
                raw_candidates = retrieval

            else:
                raise TypeError(
                    f"Unexpected retrieval record "
                    f"type at index {position}: "
                    f"{type(retrieval).__name__}"
                )

            if query_index not in oracle_examples:
                raise KeyError(
                    f"Missing oracle question "
                    f"{query_index}"
                )

            oracle = oracle_examples[
                query_index
            ]

            gold_table_id = str(
                oracle["table_id"]
            )

            normalized_candidates = []

            for rank, candidate in enumerate(
                raw_candidates,
                start=1,
            ):
                normalized_candidates.append(
                    normalize_candidate(
                        candidate,
                        rank,
                    )
                )

            # Remove duplicate tables while preserving
            # retrieval order.
            seen = set()
            unique_candidates = []

            for candidate in normalized_candidates:
                table_id = candidate["table_id"]

                if table_id in seen:
                    continue

                seen.add(table_id)
                unique_candidates.append(candidate)

            normalized_candidates = (
                unique_candidates
            )

            # Compute gold rank directly from retrieval.
            gold_rank = None

            for candidate in normalized_candidates:
                if (
                    candidate["table_id"]
                    == gold_table_id
                ):
                    gold_rank = candidate["rank"]
                    gold_found_total += 1
                    break

            candidate_tables = []

            for candidate in (
                normalized_candidates[
                    :candidate_k
                ]
            ):
                table_id = candidate[
                    "table_id"
                ]

                table_path = (
                    data_root
                    / "data"
                    / table_id
                    / "all.parquet"
                )

                if not table_path.exists():
                    raise FileNotFoundError(
                        f"Table not found: "
                        f"{table_path}"
                    )

                candidate_tables.append(
                    {
                        **candidate,
                        "table_path": str(
                            table_path
                        ),
                    }
                )

            candidate_ids = [
                candidate["table_id"]
                for candidate
                in candidate_tables
            ]

            gold_in_candidates = (
                gold_table_id
                in candidate_ids
            )

            if gold_in_candidates:
                gold_in_top_k_count += 1

            output_item = {
                "id": (
                    f"databench-"
                    f"{query_index}"
                ),
                "databench_qa_index":
                    query_index,
                "question":
                    oracle["question"],
                "label":
                    oracle["label"],

                "gold_table_id":
                    gold_table_id,
                "gold_rank":
                    gold_rank,
                "gold_in_candidates":
                    gold_in_candidates,

                "candidate_k":
                    candidate_k,
                "candidate_tables":
                    candidate_tables,
            }

            output_fp.write(
                json.dumps(
                    output_item,
                    ensure_ascii=False,
                )
                + "\n"
            )

            num_questions += 1

    recall = (
        gold_in_top_k_count
        / num_questions
    )

    print()
    print("=" * 60)
    print(
        f"Questions:          "
        f"{num_questions}"
    )
    print(
        f"Candidate K:        "
        f"{candidate_k}"
    )
    print(
        f"Gold found total:   "
        f"{gold_found_total}/"
        f"{num_questions}"
    )
    print(
        f"Gold in top-{candidate_k}:  "
        f"{gold_in_top_k_count}/"
        f"{num_questions}"
    )
    print(
        f"Retrieval recall:   "
        f"{recall:.6f}"
    )
    print(
        f"Output:             "
        f"{output_path}"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieval-path",
        default=(
            "data/databench_test80/"
            "retrieval/"
            "hybrid_title_columns/"
            "opentab.json"
        ),
    )

    parser.add_argument(
        "--oracle-path",
        default=(
            "data/databench_test80/"
            "wtq_databench_oracle_all.jsonl"
        ),
    )

    parser.add_argument(
        "--data-root",
        default="data/databench_test80",
    )

    parser.add_argument(
        "--output-path",
        default=(
            "data/"
            "wtq_databench_multi_"
            "opentab_top10.jsonl"
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    prepare_multi_data(
        retrieval_path=Path(
            args.retrieval_path
        ),
        oracle_path=Path(
            args.oracle_path
        ),
        data_root=Path(
            args.data_root
        ),
        output_path=Path(
            args.output_path
        ),
        candidate_k=args.candidate_k,
    )


if __name__ == "__main__":
    main()
