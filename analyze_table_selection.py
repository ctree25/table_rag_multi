# coding=utf-8

import argparse
import json
from pathlib import Path


def normalize_answer(x):
    if x is None:
        return ""
    return str(x).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_dir",
        help=(
            "Experiment folder, e.g. "
            "output/databench_multi_opentab_top10_gpt56sol_bm25"
        ),
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    log_dir = experiment_dir / "log"

    if not log_dir.exists():
        raise FileNotFoundError(
            f"Log directory not found: {log_dir}"
        )

    json_files = sorted(log_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(
            f"No JSON logs found in {log_dir}"
        )

    total = 0

    selected_gold = 0
    selected_wrong = 0

    answer_correct = 0
    answer_wrong = 0

    selected_gold_answer_correct = 0
    selected_gold_answer_wrong = 0

    selected_wrong_answer_correct = 0
    selected_wrong_answer_wrong = 0

    bad_files = []

    for path in json_files:
        try:
            with path.open(
                encoding="utf-8"
            ) as f:
                result = json.load(f)
        except Exception as e:
            bad_files.append(
                (str(path), str(e))
            )
            continue

        if "selected_is_gold" not in result:
            print(
                f"Skip: selected_is_gold missing: "
                f"{path.name}"
            )
            continue

        total += 1

        is_gold = bool(
            result["selected_is_gold"]
        )

        prediction = normalize_answer(
            result.get("answer")
        )
        label = normalize_answer(
            result.get("label")
        )

        # Simple exact comparison of the answer stored
        # in the TableRAG log against the stored label.
        is_correct = prediction == label

        if is_gold:
            selected_gold += 1

            if is_correct:
                selected_gold_answer_correct += 1
            else:
                selected_gold_answer_wrong += 1

        else:
            selected_wrong += 1

            if is_correct:
                selected_wrong_answer_correct += 1
            else:
                selected_wrong_answer_wrong += 1

        if is_correct:
            answer_correct += 1
        else:
            answer_wrong += 1

    if total == 0:
        raise RuntimeError(
            "No valid result logs found."
        )

    selection_accuracy = (
        selected_gold / total
    )

    answer_accuracy = (
        answer_correct / total
    )

    correct_given_gold = (
        selected_gold_answer_correct
        / selected_gold
        if selected_gold
        else 0.0
    )

    correct_given_wrong = (
        selected_wrong_answer_correct
        / selected_wrong
        if selected_wrong
        else 0.0
    )

    print()
    print("=" * 64)
    print(f"Experiment: {experiment_dir}")
    print(f"Valid results: {total}")
    print("=" * 64)

    print()
    print("[Table Selection]")
    print(
        f"Selected gold table: "
        f"{selected_gold}/{total} "
        f"= {selection_accuracy:.4f}"
    )
    print(
        f"Selected wrong table: "
        f"{selected_wrong}/{total} "
        f"= {selected_wrong / total:.4f}"
    )

    print()
    print("[Answer]")
    print(
        f"Correct answer: "
        f"{answer_correct}/{total} "
        f"= {answer_accuracy:.4f}"
    )
    print(
        f"Wrong answer: "
        f"{answer_wrong}/{total} "
        f"= {answer_wrong / total:.4f}"
    )

    print()
    print("[Answer | Selected Gold Table]")
    print(
        f"Correct: "
        f"{selected_gold_answer_correct}/"
        f"{selected_gold} "
        f"= {correct_given_gold:.4f}"
    )
    print(
        f"Wrong: "
        f"{selected_gold_answer_wrong}/"
        f"{selected_gold} "
        f"= "
        f"{selected_gold_answer_wrong / selected_gold:.4f}"
        if selected_gold
        else "Wrong: N/A"
    )

    print()
    print("[Answer | Selected Wrong Table]")
    print(
        f"Correct: "
        f"{selected_wrong_answer_correct}/"
        f"{selected_wrong} "
        f"= {correct_given_wrong:.4f}"
    )
    print(
        f"Wrong: "
        f"{selected_wrong_answer_wrong}/"
        f"{selected_wrong} "
        f"= "
        f"{selected_wrong_answer_wrong / selected_wrong:.4f}"
        if selected_wrong
        else "Wrong: N/A"
    )

    print()
    print("=" * 64)

    if bad_files:
        print(
            f"\nWarning: {len(bad_files)} "
            "broken JSON file(s) skipped:"
        )

        for path, error in bad_files:
            print(f"  {path}: {error}")


if __name__ == "__main__":
    main()
