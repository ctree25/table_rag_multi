# coding=utf-8

import json
from pathlib import Path

from tqdm import tqdm


def load_dataset(task, dataset_path, stop_at=-1):
    dataset = []
    tag = Path(dataset_path).stem

    with open(dataset_path, encoding="utf-8") as fp:
        all_lines = fp.readlines()

    # 原版在 stop_at=-1 時會漏掉最後一題。
    # 指定正整數時才進行截斷。
    if stop_at is not None and stop_at >= 0:
        all_lines = all_lines[:stop_at]

    for i, line in tqdm(
        enumerate(all_lines),
        total=len(all_lines),
        desc=f"Loading {task}-{tag} dataset",
    ):
        info = json.loads(line)

        # DataBench 已經有 databench-0、databench-1 等穩定 ID，
        # 不要再改成檔名加流水號。
        if "id" not in info:
            info["id"] = f"{tag}-{i}"

        # 單表資料才需要由 table_caption 建立 table_id。
        # Multi-table 資料則使用 candidate_tables。
        if (
            "table_id" not in info
            and "table_caption" in info
        ):
            info["table_id"] = (
                info["table_caption"].replace(" ", "_")
            )

        dataset.append(info)

    return dataset
