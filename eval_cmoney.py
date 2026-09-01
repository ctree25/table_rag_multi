
#!/usr/bin/env python3
# coding: utf-8

import argparse
import csv
import json
import re
from pathlib import Path


def normalize_whitespace(text):
    """Only remove whitespace. Do NOT change values/units/precision."""
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))


def load_gt(path):
    gt = {}

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            qid = int(row["problem_id"])
            gt[qid] = {
                "question": row["problem"].strip(),
                "answer": row["answer_text"].strip(),
            }

    return gt


def load_predictions(log_dir, sc_id=0):
    preds = {}

    for path in Path(log_dir).glob(f"*-{sc_id}.json"):
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)

            qid = int(
                obj.get(
                    "question_id",
                    obj.get("id"),
                )
            )

            answer = obj.get("answer", "")

            if answer is None:
                answer = ""
            elif isinstance(answer, bool):
                answer = "True" if answer else "False"
            elif isinstance(answer, (list, dict)):
                answer = json.dumps(
                    answer,
                    ensure_ascii=False,
                    separators=(", ", ": "),
                )
            else:
                answer = str(answer)

            preds[qid] = {
                "answer": answer.strip(),
                "table_source": obj.get("table_source"),
                "selected_is_gold": obj.get(
                    "selected_is_gold"
                ),
            }

        except Exception as e:
            print(
                f"[WARN] {path}: "
                f"{type(e).__name__}: {e}"
            )

    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-dir",
        default="output/cmoney_api_oracle_qwen35_9b/log",
    )

    parser.add_argument(
        "--answer-csv",
        default=(
            "data/cmoney/"
            "basic_problems_20260116_answer.csv"
        ),
    )

    parser.add_argument(
        "--sc-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--show-wrong",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    gt = load_gt(args.answer_csv)
    preds = load_predictions(
        args.log_dir,
        args.sc_id,
    )

    eval_ids = sorted(set(preds) & set(gt))
    total = len(eval_ids)

    strict_correct = 0
    normalized_correct = 0

    wrong = []

    for qid in eval_ids:
        gt_answer = gt[qid]["answer"]

        pred = preds[qid]
        pred_answer = pred["answer"]

        # 1. Strict exact match
        strict_ok = (
            pred_answer == gt_answer
        )

        # 2. Exact match after ONLY removing whitespace
        normalized_ok = (
            normalize_whitespace(pred_answer)
            ==
            normalize_whitespace(gt_answer)
        )

        if strict_ok:
            strict_correct += 1

        if normalized_ok:
            normalized_correct += 1

        if not normalized_ok:
            wrong.append({
                "qid": qid,
                "question": gt[qid]["question"],
                "gt": gt_answer,
                "pred": pred_answer,
                "table_source": (
                    pred.get("table_source")
                    if pred
                    else None
                ),
                "selected_is_gold": (
                    pred.get("selected_is_gold")
                    if pred
                    else None
                ),
            })

    strict_acc = (
        strict_correct / total
        if total
        else 0
    )

    normalized_acc = (
        normalized_correct / total
        if total
        else 0
    )

    print("=" * 72)
    print("CMoney Evaluation")
    print("=" * 72)

    print(f"GT questions       : {len(gt)}")
    print(f"Prediction logs    : {len(preds)}")
    print(f"Evaluated questions: {total}")

    print()

    print(
        f"Strict Exact Match : "
        f"{strict_acc:.6f} "
        f"({strict_correct}/{total})"
    )

    print(
        f"Whitespace Norm EM : "
        f"{normalized_acc:.6f} "
        f"({normalized_correct}/{total})"
    )

    print(
        f"Recovered by norm  : "
        f"{normalized_correct - strict_correct}"
    )

    print()

    if args.show_wrong == 0:
        return

    if args.show_wrong < 0:
        show = wrong
    else:
        show = wrong[:args.show_wrong]

    # print(
    #     f"Wrong after whitespace normalization: "
    #     f"{len(wrong)}"
    # )
    # print("-" * 72)

    # for x in show:
    #     print(
    #         f"[{x['qid']}] "
    #         f"{x['question']}"
    #     )
    #     print(f"  GT   : {x['gt']!r}")
    #     print(f"  Pred : {x['pred']!r}")

    #     if x["table_source"] is not None:
    #         print(
    #             f"  Table: "
    #             f"{x['table_source']}"
    #         )

    #     if x["selected_is_gold"] is not None:
    #         print(
    #             f"  selected_is_gold: "
    #             f"{x['selected_is_gold']}"
    #         )

    #     print()


if __name__ == "__main__":
    main()
