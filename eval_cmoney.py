#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import random
import re
import statistics
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


def build_random_folds(
    qids,
    n_splits=5,
    seed=42,
):
    """Match the previous 5-fold construction exactly.

    Important:
    - qids are sorted first, so assignment is independent of input order.
    - random.Random(seed).shuffle(...) is the same logic as the previous
      five_fold_retrieval_eval.py.
    """
    qids = sorted(int(qid) for qid in qids)

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    if len(qids) < n_splits:
        raise ValueError(
            f"Not enough questions ({len(qids)}) "
            f"for {n_splits} folds"
        )

    indices = list(range(len(qids)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    base = len(qids) // n_splits
    remainder = len(qids) % n_splits

    fold_sizes = [
        base + (1 if i < remainder else 0)
        for i in range(n_splits)
    ]

    fold_qids = []
    start = 0

    for size in fold_sizes:
        idxs = indices[start:start + size]
        fold_qids.append([qids[i] for i in idxs])
        start += size

    return fold_qids


def load_fold_assignments(
    path,
    eval_ids,
    n_splits=5,
):
    """Load the exact historical fold_assignments.csv if supplied."""
    eval_ids = set(int(qid) for qid in eval_ids)

    assignments = {
        fold: []
        for fold in range(1, n_splits + 1)
    }

    seen = set()

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"question_id", "fold"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Fold assignment file is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            qid = int(row["question_id"])

            # Historical file may have a split column.
            # Use only test rows if it exists.
            if (
                "split" in row
                and str(row.get("split", "")).strip()
                and str(row["split"]).strip().lower() != "test"
            ):
                continue

            if qid not in eval_ids:
                continue

            fold = int(row["fold"])

            if fold not in assignments:
                raise ValueError(
                    f"Unexpected fold={fold} for qid={qid}"
                )

            if qid in seen:
                raise ValueError(
                    f"Duplicate fold assignment for qid={qid}"
                )

            assignments[fold].append(qid)
            seen.add(qid)

    missing_eval_ids = sorted(eval_ids - seen)

    if missing_eval_ids:
        raise ValueError(
            "The supplied fold assignment file does not cover all "
            f"evaluated questions. Missing IDs: "
            f"{missing_eval_ids[:30]}"
        )

    return [
        assignments[fold]
        for fold in range(1, n_splits + 1)
    ]


def evaluate_ids(
    qids,
    gt,
    preds,
):
    strict_correct = 0
    normalized_correct = 0

    for qid in qids:
        gt_answer = gt[qid]["answer"]
        pred_answer = preds[qid]["answer"]

        if pred_answer == gt_answer:
            strict_correct += 1

        if (
            normalize_whitespace(pred_answer)
            == normalize_whitespace(gt_answer)
        ):
            normalized_correct += 1

    total = len(qids)

    return {
        "n": total,
        "strict_correct": strict_correct,
        "strict_acc": (
            strict_correct / total
            if total else 0.0
        ),
        "normalized_correct": normalized_correct,
        "normalized_acc": (
            normalized_correct / total
            if total else 0.0
        ),
    }


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

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Used only when --fold-assignments is not supplied. "
            "Matches the previous random 5-fold script."
        ),
    )

    parser.add_argument(
        "--fold-assignments",
        default=None,
        help=(
            "Optional path to the previous fold_assignments.csv. "
            "Use this to guarantee exactly the same historical folds."
        ),
    )

    args = parser.parse_args()

    gt = load_gt(args.answer_csv)

    preds = load_predictions(
        args.log_dir,
        args.sc_id,
    )

    # Same denominator as before:
    # only questions having BOTH a prediction log and GT.
    eval_ids = sorted(set(preds) & set(gt))

    total = len(eval_ids)

    # ================================================================
    # Full / pooled evaluation
    # ================================================================
    full_metrics = evaluate_ids(
        eval_ids,
        gt,
        preds,
    )

    strict_correct = full_metrics[
        "strict_correct"
    ]
    normalized_correct = full_metrics[
        "normalized_correct"
    ]

    strict_acc = full_metrics["strict_acc"]
    normalized_acc = full_metrics[
        "normalized_acc"
    ]

    wrong = []

    for qid in eval_ids:
        gt_answer = gt[qid]["answer"]
        pred = preds[qid]
        pred_answer = pred["answer"]

        normalized_ok = (
            normalize_whitespace(pred_answer)
            == normalize_whitespace(gt_answer)
        )

        if not normalized_ok:
            wrong.append(
                {
                    "qid": qid,
                    "question": gt[qid]["question"],
                    "gt": gt_answer,
                    "pred": pred_answer,
                    "table_source": (
                        pred.get("table_source")
                        if pred else None
                    ),
                    "selected_is_gold": (
                        pred.get("selected_is_gold")
                        if pred else None
                    ),
                }
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

    # ================================================================
    # 5-fold stability evaluation
    # ================================================================
    if args.fold_assignments:
        fold_qids = load_fold_assignments(
            args.fold_assignments,
            eval_ids,
            n_splits=args.n_splits,
        )

        fold_source = (
            f"historical assignments: "
            f"{args.fold_assignments}"
        )

    else:
        fold_qids = build_random_folds(
            eval_ids,
            n_splits=args.n_splits,
            seed=args.seed,
        )

        fold_source = (
            f"reconstructed with seed={args.seed}"
        )

    fold_metrics = []

    for fold_idx, qids in enumerate(
        fold_qids,
        start=1,
    ):
        metrics = evaluate_ids(
            qids,
            gt,
            preds,
        )

        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    strict_values = [
        x["strict_acc"]
        for x in fold_metrics
    ]

    normalized_values = [
        x["normalized_acc"]
        for x in fold_metrics
    ]

    strict_mean = statistics.mean(
        strict_values
    )
    strict_std = (
        statistics.stdev(strict_values)
        if len(strict_values) > 1
        else 0.0
    )

    normalized_mean = statistics.mean(
        normalized_values
    )
    normalized_std = (
        statistics.stdev(normalized_values)
        if len(normalized_values) > 1
        else 0.0
    )

    print()
    print("=" * 72)
    print(
        f"{args.n_splits}-Fold Test Stability "
        f"({fold_source})"
    )
    print("=" * 72)

    for metrics in fold_metrics:
        print(
            f"Fold {metrics['fold']}: "
            f"n={metrics['n']:3d} | "
            f"Strict={metrics['strict_acc']:.6f} "
            f"({metrics['strict_correct']}/{metrics['n']}) | "
            f"Whitespace={metrics['normalized_acc']:.6f} "
            f"({metrics['normalized_correct']}/{metrics['n']})"
        )

    print()
    print("Mean ± sample std across test folds")

    print(
        f"Strict Exact Match : "
        f"{strict_mean:.6f} ± {strict_std:.6f}"
    )

    print(
        f"Whitespace Norm EM : "
        f"{normalized_mean:.6f} ± "
        f"{normalized_std:.6f}"
    )

    print()
    print("Pooled/full-set accuracy")

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

    if args.show_wrong == 0:
        return

    if args.show_wrong < 0:
        show = wrong
    else:
        show = wrong[:args.show_wrong]

    # Wrong-case printing remains disabled to preserve current behavior.
    _ = show


if __name__ == "__main__":
    main()
