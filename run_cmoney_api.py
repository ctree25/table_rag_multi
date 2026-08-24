"""Run CMoney TableRAG from a precomputed table-retrieval JSONL.

Example:
python run_cmoney_api.py \
  --retrieval_path=data/cmoney/retrieval_eval.jsonl \
  --schema_path=data/cmoney/table_schema_20250122.csv \
  --candidate_k=5 \
  --schema_top_k=5 \
  --model_name=gpt-4.1-mini \
  --log_dir=output/cmoney_api_top5 \
  --verbose=True
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import fire
from tqdm import tqdm

from agent.rag_agent_multi_cmoney_api import TableRAGMultiCMoneyAPIAgent


def load_retrieval_jsonl(
    retrieval_path: str | Path,
    *,
    candidate_k: int,
) -> list[dict]:
    """Convert retrieval_eval.jsonl rows into agent input records."""
    path = Path(retrieval_path)
    records: list[dict] = []

    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc

            question_id = obj.get("question_id")
            query = obj.get("query")
            retrieved_tables = obj.get("retrieved_tables") or []
            gt_tables = obj.get("gt_tables") or []

            if question_id is None or query is None:
                raise ValueError(
                    f"Missing question_id/query at {path}:{line_no}"
                )
            if not isinstance(retrieved_tables, list):
                raise ValueError(
                    f"retrieved_tables must be a list at {path}:{line_no}"
                )

            candidates = [
                str(name).strip()
                for name in retrieved_tables[:candidate_k]
                if str(name).strip()
            ]
            records.append(
                {
                    "id": question_id,
                    "question_id": question_id,
                    "question": str(query),
                    "gt_tables": [str(x) for x in gt_tables],
                    "candidate_tables": candidates,
                    "retrieval_metadata": {
                        "first_gt_rank": obj.get("first_gt_rank"),
                        "hit_at_k": obj.get("hit_at_k"),
                        "retrieval_mode": obj.get("retrieval_mode"),
                        "timestamp": obj.get("timestamp"),
                    },
                }
            )

    return records


def write_predictions(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for result in results:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(
    retrieval_path: str = "data/cmoney/retrieval_eval.jsonl",
    schema_path: str = "data/cmoney/table_schema_20250122.csv",
    candidate_k: int = 5,
    schema_top_k: int = 5,
    model_name: str = "gpt-4.1-mini",
    retrieve_mode: str = "hybrid",
    embed_model_name: str = "jinaai/jina-embeddings-v3",
    log_dir: str = "output/cmoney_api_top5",
    db_dir: str = "db/cmoney_api",
    sc: int = 1,
    max_depth: int = 6,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    api_timeout: float = 30.0,
    observation_max_rows: int = 50,
    observation_max_chars: int = 12000,
    schema_embed_device: str = "cpu",
    schema_embed_batch_size: int = 64,
    schema_encode_dim: int = 1024,
    stop_at: int = -1,
    resume_from: int = 0,
    load_exist: bool = True,
    verbose: bool = False,
):
    """Run schema-only retrieval and SQL-API reasoning for CMoney."""
    os.makedirs(Path(log_dir) / "log", exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    dataset = load_retrieval_jsonl(
        retrieval_path,
        candidate_k=candidate_k,
    )
    if stop_at < 0 or stop_at > len(dataset):
        stop_at = len(dataset)

    config = {
        "retrieval_path": retrieval_path,
        "schema_path": schema_path,
        "candidate_k": candidate_k,
        "schema_top_k": schema_top_k,
        "model_name": model_name,
        "retrieve_mode": retrieve_mode,
        "embed_model_name": embed_model_name,
        "log_dir": log_dir,
        "db_dir": db_dir,
        "sc": sc,
        "max_depth": max_depth,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "api_timeout": api_timeout,
        "observation_max_rows": observation_max_rows,
        "observation_max_chars": observation_max_chars,
        "schema_embed_device": schema_embed_device,
        "schema_embed_batch_size": schema_embed_batch_size,
        "schema_encode_dim": schema_encode_dim,
        "stop_at": stop_at,
        "resume_from": resume_from,
        "load_exist": load_exist,
    }
    with (Path(log_dir) / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)

    agent = TableRAGMultiCMoneyAPIAgent(
        schema_path=schema_path,
        candidate_k=candidate_k,
        schema_top_k=schema_top_k,
        api_timeout=api_timeout,
        observation_max_rows=observation_max_rows,
        observation_max_chars=observation_max_chars,
        schema_embed_device=schema_embed_device,
        schema_embed_batch_size=schema_embed_batch_size,
        schema_encode_dim=schema_encode_dim,
        model_name=model_name,
        retrieve_mode=retrieve_mode,
        embed_model_name=embed_model_name,
        # Keep task=wtq only for compatibility with the inherited base class.
        # This agent imports its own CMoney solve prompt directly.
        task="wtq",
        agent_type="TableRAGMulti",
        top_k=schema_top_k,
        sr=0,
        max_encode_cell=1,
        temperature=temperature,
        top_p=top_p,
        stop_tokens=["Observation:"],
        max_tokens=max_tokens,
        max_depth=max_depth,
        load_exist=load_exist,
        log_dir=log_dir,
        db_dir=db_dir,
        verbose=verbose,
    )

    results: list[dict] = []
    subset = dataset[resume_from:stop_at]
    for data in tqdm(subset):
        for sc_id in range(sc):
            result = agent.run(data, sc_id=sc_id)
            # Preserve retrieval-file metadata in the aggregate JSONL.
            result["retrieval_metadata"] = data.get("retrieval_metadata", {})
            results.append(result)

    output_path = Path(log_dir) / "predictions.jsonl"
    write_predictions(results, output_path)

    summary = {
        "num_results": len(results),
        "num_questions": len(subset),
        "candidate_k": candidate_k,
        "schema_top_k": schema_top_k,
        "selected_gold_count": sum(
            1 for r in results if r.get("selected_is_gold")
        ),
        "gold_in_candidates_count": sum(
            1 for r in results if r.get("gold_in_candidates")
        ),
        "predictions_path": str(output_path),
    }
    with (Path(log_dir) / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    fire.Fire(main)
