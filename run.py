# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import pandas as pd
from tqdm import tqdm

from agent import (
    TableAgent,
    TableRAGAgent,
    TableRAGMultiAgent,
)
from agent.rag_agent_oracle import TableRAGOracleAgent
from evaluate import evaluate
from utils.load_data import load_dataset


def solve(args):
    agent_args, data, sc_id = args

    if agent_args["agent_type"] == "TableRAGMulti":
        agent = TableRAGMultiAgent(**agent_args)

    elif agent_args["agent_type"] == "TableRAGOracle":
        agent = TableRAGOracleAgent(**agent_args)

    elif "TableRAG" in agent_args["agent_type"]:
        agent = TableRAGAgent(**agent_args)

    elif agent_args["agent_type"] in [
        "PyReAct",
        "ReadSchema",
        "RandSampling",
        "TableSampling",
    ]:
        agent = TableAgent(**agent_args)

    else:
        raise NotImplementedError(
            f"Agent type {agent_args['agent_type']} not supported."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return agent.run(data, sc_id=sc_id)


def write_token_usage(log_dir):
    """Aggregate all completed per-question logs."""
    log_path = Path(log_dir) / "log"
    records = []

    if log_path.exists():
        for result_path in sorted(log_path.glob("*.json")):
            try:
                with result_path.open(
                    encoding="utf-8"
                ) as fp:
                    records.append(json.load(fp))
            except (OSError, json.JSONDecodeError):
                # Ignore incomplete files if a process is writing.
                continue

    # ------------------------------------------------------------------
    # Actual usage
    #
    # New Multi-table logs explicitly store actual_* fields.
    # For old logs / non-Multi agents, fall back to the original fields.
    # ------------------------------------------------------------------
    actual_input_token_count = sum(
        int(
            record.get(
                "actual_input_token_count",
                record.get("input_token_count", 0),
            )
            or 0
        )
        for record in records
    )

    actual_output_token_count = sum(
        int(
            record.get(
                "actual_output_token_count",
                record.get("output_token_count", 0),
            )
            or 0
        )
        for record in records
    )

    actual_total_token_count = (
        actual_input_token_count
        + actual_output_token_count
    )

    # ------------------------------------------------------------------
    # Cached expansion usage
    #
    # Only Multi-table logs with query-expansion cache accounting have
    # these fields. Old/non-Multi logs naturally contribute zero.
    # ------------------------------------------------------------------
    cached_input_token_count = sum(
        int(record.get("cached_input_token_count", 0) or 0)
        for record in records
    )

    cached_output_token_count = sum(
        int(record.get("cached_output_token_count", 0) or 0)
        for record in records
    )

    cached_total_token_count = (
        cached_input_token_count
        + cached_output_token_count
    )

    # ------------------------------------------------------------------
    # No-cache-equivalent usage
    #
    # Prefer explicitly stored per-question values when available.
    # For old logs, actual usage is the best backward-compatible fallback.
    # ------------------------------------------------------------------
    no_cache_equivalent_input_token_count = sum(
        int(
            record.get(
                "no_cache_equivalent_input_token_count",
                record.get(
                    "actual_input_token_count",
                    record.get("input_token_count", 0),
                ),
            )
            or 0
        )
        for record in records
    )

    no_cache_equivalent_output_token_count = sum(
        int(
            record.get(
                "no_cache_equivalent_output_token_count",
                record.get(
                    "actual_output_token_count",
                    record.get("output_token_count", 0),
                ),
            )
            or 0
        )
        for record in records
    )

    no_cache_equivalent_total_token_count = (
        no_cache_equivalent_input_token_count
        + no_cache_equivalent_output_token_count
    )

    usage = {
        "num_results": len(records),

        # Backward-compatible names: actual tokens consumed by this run.
        "input_token_count": actual_input_token_count,
        "output_token_count": actual_output_token_count,
        "total_token_count": actual_total_token_count,

        # Explicit actual usage.
        "actual_input_token_count": actual_input_token_count,
        "actual_output_token_count": actual_output_token_count,
        "actual_total_token_count": actual_total_token_count,

        # Tokens skipped because query expansion came from cache.
        "cached_input_token_count": cached_input_token_count,
        "cached_output_token_count": cached_output_token_count,
        "cached_total_token_count": cached_total_token_count,

        # Comparable theoretical usage with query-expansion cache disabled.
        "no_cache_equivalent_input_token_count":
            no_cache_equivalent_input_token_count,
        "no_cache_equivalent_output_token_count":
            no_cache_equivalent_output_token_count,
        "no_cache_equivalent_total_token_count":
            no_cache_equivalent_total_token_count,
    }

    usage_path = Path(log_dir) / "token_usage.json"
    temporary_path = Path(
        str(usage_path) + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            usage,
            fp,
            ensure_ascii=False,
            indent=4,
        )

    # Atomic replacement prevents a half-written JSON file.
    os.replace(temporary_path, usage_path)


def write_databench_responses(
    results,
    log_dir,
    target_sc_id=0,
):
    """Write official DataBench/SemEval one-answer-per-line file."""

    selected = [
        result
        for result in results
        if int(result.get("sc_id", 0)) == target_sc_id
    ]

    def get_index(result):
        result_id = str(result.get("id", ""))

        if not result_id.startswith("databench-"):
            raise ValueError(
                f"Unexpected DataBench id: {result_id}"
            )

        return int(result_id.removeprefix("databench-"))

    selected.sort(key=get_index)

    indices = [
        get_index(result)
        for result in selected
    ]

    expected_indices = list(range(len(selected)))

    if indices != expected_indices:
        missing = sorted(
            set(expected_indices) - set(indices)
        )
        raise RuntimeError(
            "Cannot export responses.txt because result IDs "
            f"are incomplete or out of range. Missing: {missing[:20]}"
        )

    output_path = Path(log_dir) / "responses.txt"
    temporary_path = Path(str(output_path) + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as fp:
        for result in selected:
            answer = result.get("answer", "")

            if answer is None:
                answer = ""

            answer = (
                str(answer)
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

            fp.write(answer + "\n")

    os.replace(temporary_path, output_path)

    return output_path


def main(
    dataset_path='data/tabfact/test_sub_nosynth.jsonl',
    model_name='gpt-3.5-turbo-0125',
    agent_type='PyReAct',
    retrieve_mode='embed',
    embed_model_name='text-embedding-3-large',
    log_dir='output/test',
    db_dir='db/',
    top_k=5,
    candidate_k=10,
    sr=0,  # self-refine, deprecated
    sc=1,  # self-consistency
    max_encode_cell=10000,
    stop_at=-1,
    resume_from=0,
    load_exist=True,
    n_worker=1,
    max_depth=5,
    max_tokens=128,
    temperature=0.8,
    verbose=False,
):
    os.makedirs(os.path.join(log_dir, 'log'), exist_ok=True)

    # store the config
    task = [
        task_name
        for task_name in ['tabfact', 'wtq', 'arcade', 'bird']
        if task_name in dataset_path
    ][0]

    db_dir = os.path.join(
        db_dir,
        task + '_' + Path(dataset_path).stem,
    )

    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump(
            {
                key: value
                for key, value in locals().items()
                if key != 'fp'
            },
            fp,
            indent=4,
        )

    dataset = load_dataset(task, dataset_path, stop_at)
    if stop_at < 0:
        stop_at = len(dataset)

    agent_args = {
        'model_name': model_name,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': agent_type,
        'top_k': top_k,
        'sr': sr,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'load_exist': load_exist,
        'max_depth': max_depth,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'verbose': verbose,
    }

    if agent_type == "TableRAGMulti":
        agent_args["candidate_k"] = candidate_k

    results = []

    if n_worker == 1:
        for data in tqdm(dataset[resume_from:stop_at]):
            for sc_id in tqdm(
                range(sc),
                position=1,
                leave=False,
            ):
                result = solve(
                    (agent_args, data, sc_id)
                )
                results.append(result)
                write_token_usage(log_dir)

    else:
        with tqdm(
            total=(stop_at - resume_from) * sc
        ) as pbar:
            with ProcessPoolExecutor(
                max_workers=n_worker
            ) as executor:
                futures = [
                    executor.submit(
                        solve,
                        (agent_args, data, sc_id),
                    )
                    for data in dataset[resume_from:stop_at]
                    for sc_id in range(sc)
                ]

                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    results.append(result)
                    write_token_usage(log_dir)

    write_token_usage(log_dir)

    if "databench" in dataset_path.lower():
        if sc != 1:
            raise ValueError(
                "Automatic DataBench export currently requires "
                "sc=1. Self-consistency results must be voted "
                "before exporting."
            )

        responses_path = write_databench_responses(
            results=results,
            log_dir=log_dir,
            target_sc_id=0,
        )

        print(f"DataBench responses: {responses_path}")

    acc = evaluate(task, results)
    print(f'Accuracy: {acc}')

    # Report both actual and no-cache-equivalent token usage.
    stats_keys = [
        'n_iter',
        'init_prompt_token_count',

        'input_token_count',
        'output_token_count',
        'total_token_count',

        'actual_input_token_count',
        'actual_output_token_count',
        'actual_total_token_count',

        'cached_input_token_count',
        'cached_output_token_count',
        'cached_total_token_count',

        'no_cache_equivalent_input_token_count',
        'no_cache_equivalent_output_token_count',
        'no_cache_equivalent_total_token_count',
    ]

    stats_df = pd.DataFrame.from_records(results)

    # Backward compatibility for agents/logs that do not expose the
    # explicit token-accounting fields.
    stats_df['actual_input_token_count'] = stats_df.get(
        'actual_input_token_count',
        stats_df.get('input_token_count', 0),
    )
    stats_df['actual_output_token_count'] = stats_df.get(
        'actual_output_token_count',
        stats_df.get('output_token_count', 0),
    )
    stats_df['actual_total_token_count'] = (
        stats_df['actual_input_token_count']
        + stats_df['actual_output_token_count']
    )

    if 'cached_input_token_count' not in stats_df.columns:
        stats_df['cached_input_token_count'] = 0
    if 'cached_output_token_count' not in stats_df.columns:
        stats_df['cached_output_token_count'] = 0

    stats_df['cached_total_token_count'] = (
        stats_df['cached_input_token_count']
        + stats_df['cached_output_token_count']
    )

    if 'no_cache_equivalent_input_token_count' not in stats_df.columns:
        stats_df['no_cache_equivalent_input_token_count'] = (
            stats_df['actual_input_token_count']
            + stats_df['cached_input_token_count']
        )

    if 'no_cache_equivalent_output_token_count' not in stats_df.columns:
        stats_df['no_cache_equivalent_output_token_count'] = (
            stats_df['actual_output_token_count']
            + stats_df['cached_output_token_count']
        )

    stats_df['no_cache_equivalent_total_token_count'] = (
        stats_df['no_cache_equivalent_input_token_count']
        + stats_df['no_cache_equivalent_output_token_count']
    )

    # Keep the old names explicitly aligned with actual usage.
    stats_df['input_token_count'] = (
        stats_df['actual_input_token_count']
    )
    stats_df['output_token_count'] = (
        stats_df['actual_output_token_count']
    )
    stats_df['total_token_count'] = (
        stats_df['actual_total_token_count']
    )

    print(
        stats_df[stats_keys]
        .describe()
        .to_string()
    )

    # store the result
    result_dict = (
        stats_df[stats_keys]
        .mean()
        .to_dict()
    )
    result_dict['accuracy'] = acc

    for key in [
        'model_name',
        'retrieve_mode',
        'embed_model_name',
        'task',
        'agent_type',
        'top_k',
        'max_encode_cell',
        'sr',
    ]:
        result_dict[key] = agent_args[key]

    result_dict['sc'] = sc
    result_dict['data'] = Path(dataset_path).stem

    result_path = os.path.join(
        log_dir,
        'result.json',
    )

    with open(result_path, 'w') as fp:
        json.dump(
            result_dict,
            fp,
            indent=4,
        )


if __name__ == '__main__':
    fire.Fire(main)
