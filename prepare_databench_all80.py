# coding=utf-8

import json
from pathlib import Path


ORACLE_PATH = Path(
    "data/wtq_databench_oracle_all.jsonl"
)

DATA_ROOT = Path(
    "data/databench_test80/data"
)

OUTPUT_PATH = Path(
    "data/wtq_databench_multi_all80.jsonl"
)


def load_all_tables():
    tables = []

    for table_dir in sorted(DATA_ROOT.iterdir()):
        if not table_dir.is_dir():
            continue

        parquet_path = table_dir / "all.parquet"

        if not parquet_path.exists():
            continue

        table_id = table_dir.name

        tables.append({
            "table_id": table_id,
            "title": table_id,
            "table_path": str(parquet_path),
        })

    return tables


def main():
    all_tables = load_all_tables()

    print(f"Found tables: {len(all_tables)}")

    if len(all_tables) != 80:
        raise RuntimeError(
            f"Expected 80 tables, but found {len(all_tables)}"
        )

    oracle_items = []

    with ORACLE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                oracle_items.append(
                    json.loads(line)
                )

    print(f"Questions: {len(oracle_items)}")

    if len(oracle_items) != 522:
        raise RuntimeError(
            f"Expected 522 questions, "
            f"but found {len(oracle_items)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as out:

        for idx, oracle in enumerate(oracle_items):

            gold_table_id = str(
                oracle["table_id"]
            )

            candidate_tables = []

            gold_position = None

            for position, table in enumerate(
                all_tables,
                start=1,
            ):
                candidate = {
                    # IMPORTANT:
                    # This is only a deterministic dataframe index,
                    # NOT a retrieval rank.
                    "rank": position,
                    "table_id": table["table_id"],
                    "title": table["title"],
                    "score": None,
                    "bm25_score": None,
                    "dense_score": None,
                    "table_path": table["table_path"],
                }

                candidate_tables.append(candidate)

                if table["table_id"] == gold_table_id:
                    gold_position = position

            if gold_position is None:
                raise RuntimeError(
                    f"Gold table not found for q{idx}: "
                    f"{gold_table_id}"
                )

            item = {
                "id": f"databench-{idx}",
                "databench_qa_index": idx,
                "question": oracle["question"],
                "label": oracle["label"],

                "gold_table_id": gold_table_id,

                # For all80 this is NOT retrieval rank.
                # It is only the table's deterministic position
                # in the df1...df80 mapping.
                "gold_rank": gold_position,

                # All 80 tables are always supplied.
                "gold_in_candidates": True,

                "candidate_k": 80,
                "candidate_tables": candidate_tables,
            }

            out.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 60)
    print(f"Questions:       {len(oracle_items)}")
    print(f"Candidate K:     80")
    print(f"Gold available:  {len(oracle_items)}/{len(oracle_items)}")
    print(f"Output:          {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
