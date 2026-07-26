"""Benchmark HELIOS visual verifier against LIBERO oracle labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

from libero.libero import benchmark

from verifier.visual_verifier import VisualOutcomeVerifier


EVAL_FILE = Path("results/baseline_object/eval_info.json")
OUTPUT_FILE = Path("results/verifier_results_v2.csv")


def sample_frames(
    video_path: Path,
) -> tuple[Image.Image, Image.Image, Image.Image]:

    frames = [frame for frame in iio.imiter(video_path)]

    if len(frames) < 3:
        raise RuntimeError(f"Too few frames in {video_path}")

    first = frames[0]
    middle = frames[len(frames) // 2]
    final = frames[-1]

    return (
        Image.fromarray(first).convert("RGB"),
        Image.fromarray(middle).convert("RGB"),
        Image.fromarray(final).convert("RGB"),
    )


def main() -> None:
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        eval_info = json.load(f)

    suite = benchmark.get_benchmark_dict()["libero_object"]()

    verifier = VisualOutcomeVerifier(device=0)

    rows: list[dict] = []

    print("\n" + "=" * 70)
    print("HELIOS VISUAL VERIFIER V2")
    print("=" * 70)

    for task_entry in eval_info["per_task"]:
        task_id = int(task_entry["task_id"])
        metrics = task_entry["metrics"]

        instruction = suite.get_task(task_id).language

        for episode_id, (oracle_success, video_path) in enumerate(
            zip(
                metrics["successes"],
                metrics["video_paths"],
                strict=True,
            )
        ):
            before, middle, after = sample_frames(Path(video_path))

            result = verifier.verify(
                instruction,
                before,
                middle,
                after,
            )

            oracle = "SUCCESS" if oracle_success else "FAILURE"
            correct = result.verdict == oracle

            print(
                f"Task {task_id} | Ep {episode_id} | "
                f"oracle={oracle:<7} | "
                f"visual={result.verdict:<9} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

            rows.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "oracle": oracle,
                    "visual_verdict": result.verdict,
                    "correct": correct,
                    "raw_response": result.raw_response.replace("\n", " "),
                    "video_path": video_path,
                }
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    success_rows = [r for r in rows if r["oracle"] == "SUCCESS"]
    failure_rows = [r for r in rows if r["oracle"] == "FAILURE"]

    tp = sum(
        r["visual_verdict"] == "SUCCESS"
        for r in success_rows
    )

    tn = sum(
        r["visual_verdict"] == "FAILURE"
        for r in failure_rows
    )

    total = len(rows)
    correct = sum(r["correct"] for r in rows)

    success_recall = tp / len(success_rows)
    failure_recall = tn / len(failure_rows)
    balanced_accuracy = (success_recall + failure_recall) / 2

    uncertain = sum(
        r["visual_verdict"] == "UNCERTAIN"
        for r in rows
    )

    print("\n" + "=" * 70)
    print("VERIFIER V2 RESULTS")
    print("=" * 70)

    print(f"Samples             : {total}")
    print(f"Accuracy            : {100 * correct / total:.1f}%")
    print(f"Success recall      : {100 * success_recall:.1f}%")
    print(f"Failure recall      : {100 * failure_recall:.1f}%")
    print(f"Balanced accuracy   : {100 * balanced_accuracy:.1f}%")
    print(f"Uncertain           : {uncertain}/{total}")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
