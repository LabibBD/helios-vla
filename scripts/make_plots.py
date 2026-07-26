"""Generate HELIOS benchmark figures."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "results/helios_recovery_v4_30ep/"
    "checkpoint_recovery_episodes.csv"
)

OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    df = pd.read_csv(INPUT)

    df["first_attempt_success"] = as_bool(
        df["first_attempt_success"]
    )
    df["final_success"] = as_bool(
        df["final_success"]
    )
    df["recovered"] = as_bool(
        df["recovered"]
    )

    # ---------------------------------------------------------
    # Figure 1: overall success improvement
    # ---------------------------------------------------------

    first = 100 * df["first_attempt_success"].mean()
    final = 100 * df["final_success"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["One-shot SmolVLA", "HELIOS"]
    values = [first, final]

    bars = ax.bar(labels, values)

    ax.set_ylabel("Task Success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("HELIOS Improves Final Manipulation Success")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "overall_success.png",
        dpi=200,
    )
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 2: per-task first vs final success
    # ---------------------------------------------------------

    grouped = (
        df.groupby("task_id")[
            ["first_attempt_success", "final_success"]
        ]
        .mean()
        * 100
    )

    ax = grouped.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_xlabel("LIBERO Object Task ID")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("First-Attempt vs Final Success by Task")
    ax.legend(
        ["First attempt", "After HELIOS recovery"]
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "per_task_success.png",
        dpi=200,
    )
    plt.close()

    # ---------------------------------------------------------
    # Figure 3: failure recovery
    # ---------------------------------------------------------

    failed_first = df[
        ~df["first_attempt_success"]
    ]

    recovered = int(failed_first["final_success"].sum())
    unrecovered = len(failed_first) - recovered

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Recovered", "Still failed"],
        [recovered, unrecovered],
    )

    ax.set_ylabel("Episodes")
    ax.set_title(
        "Outcome of Initially Failed Episodes"
    )

    for bar, value in zip(
        bars,
        [recovered, unrecovered],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            str(value),
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "recovery_outcomes.png",
        dpi=200,
    )
    plt.close(fig)

    print("=" * 60)
    print("HELIOS FIGURES GENERATED")
    print("=" * 60)
    print(f"One-shot success : {first:.1f}%")
    print(f"HELIOS success   : {final:.1f}%")
    print(f"Recovered        : {recovered}/{len(failed_first)}")

    for path in sorted(OUTPUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
