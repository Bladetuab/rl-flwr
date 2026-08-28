"""Select one globally balanced tau and plot per-round ASR/FDR for each dataset.

Data source:
    outputs/2026-07-09 .. outputs/2026-07-11

Selection rule:
    For each tau, first average each metric over both datasets and the three
    attacks. A balanced score is then the equal-weight mean of four utilities:

        lower run-level ASR is better
        lower run-level FDR is better
        higher final test accuracy is better
        lower normalized ConvLSTM forwards is better

    Utilities are min-max normalized across tau values, so the selected tau is
    one global value used for MNIST and Fashion-MNIST.

Outputs:
    sweep_figures/best_global_tau_balance_scores_20260709_20260711.csv
    sweep_figures/best_global_tau_selected_rounds_asr_fdr_20260709_20260711.csv
    sweep_figures/best_tau_rounds_asr_fdr_mnist_tau0.8.{png,pdf}
    sweep_figures/best_tau_rounds_asr_fdr_fashion_mnist_tau0.8.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from plot_rounds_asr_fdr_mnist_fashion import (
    ASR_CEILING,
    ATTACKS,
    ATTACK_COLORS,
    ATTACK_LABELS,
    DATASETS,
    DATASET_LABELS,
    DEFAULT_END_DATE,
    DEFAULT_OUT_DIR,
    DEFAULT_START_DATE,
    IEEE_FONT,
    IEEE_WIDTH_IN,
    ONSET_ROUND,
    check_geometry,
    collect_runs,
)
from plot_tau_accuracy_normalized_forwards import (
    read_convlstm_forwards,
    read_final_accuracy,
)


OUT_SCORE_STEM = "best_global_tau_balance_scores_20260709_20260711"
OUT_ROUNDS_STEM = "best_global_tau_selected_rounds_asr_fdr_20260709_20260711"

ATTACK_LINESTYLES = {
    "gaussian": "-",
    "lie": "--",
    "min_max": "-.",
}
ATTACK_MARKERS = {
    "gaussian": "o",
    "lie": "s",
    "min_max": "^",
}


def run_level_error_rates(row: dict) -> tuple[float, float]:
    totals = {"fn": 0, "fp": 0, "n": 0}
    for a in row["curve"].values():
        totals["fn"] += a["fn"]
        totals["fp"] += a["fp"]
        totals["n"] += a["n"]
    return totals["fn"] / totals["n"], totals["fp"] / totals["n"]


def attach_run_level_metrics(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        n_rounds, final_accuracy = read_final_accuracy(Path(row["run_dir"]))
        _, normalized_forwards = read_convlstm_forwards(Path(row["run_dir"]))
        if final_accuracy is None or normalized_forwards is None:
            continue
        asr, fdr = run_level_error_rates(row)
        enriched.append(
            {
                **row,
                "run_asr": asr,
                "run_fdr": fdr,
                "final_accuracy": final_accuracy,
                "normalized_convlstm_forwards": normalized_forwards,
                "n_result_rounds": n_rounds,
            }
        )
    return enriched


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_by_tau(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["tau"]].append(row)

    agg_rows = []
    expected_combos = {(dataset, attack) for dataset in DATASETS for attack in ATTACKS}
    for tau, group in grouped.items():
        combos = {(row["dataset"], row["attack"]) for row in group}
        if combos != expected_combos:
            continue
        agg_rows.append(
            {
                "tau": tau,
                "mean_asr": mean([row["run_asr"] for row in group]),
                "mean_fdr": mean([row["run_fdr"] for row in group]),
                "mean_final_accuracy": mean([row["final_accuracy"] for row in group]),
                "mean_normalized_convlstm_forwards": mean(
                    [row["normalized_convlstm_forwards"] for row in group]
                ),
            }
        )

    return sorted(agg_rows, key=lambda row: row["tau"])


def score_rows(agg_rows: list[dict]) -> list[dict]:
    ranges = {}
    for metric in (
        "mean_asr",
        "mean_fdr",
        "mean_final_accuracy",
        "mean_normalized_convlstm_forwards",
    ):
        vals = [row[metric] for row in agg_rows]
        ranges[metric] = (min(vals), max(vals))

    def low_utility(row: dict, metric: str) -> float:
        lo, hi = ranges[metric]
        if hi == lo:
            return 1.0
        return (hi - row[metric]) / (hi - lo)

    def high_utility(row: dict, metric: str) -> float:
        lo, hi = ranges[metric]
        if hi == lo:
            return 1.0
        return (row[metric] - lo) / (hi - lo)

    scored = []
    for row in agg_rows:
        utility_asr = low_utility(row, "mean_asr")
        utility_fdr = low_utility(row, "mean_fdr")
        utility_accuracy = high_utility(row, "mean_final_accuracy")
        utility_forwards = low_utility(row, "mean_normalized_convlstm_forwards")
        score = mean([utility_asr, utility_fdr, utility_accuracy, utility_forwards])
        scored.append(
            {
                **row,
                "utility_asr": utility_asr,
                "utility_fdr": utility_fdr,
                "utility_final_accuracy": utility_accuracy,
                "utility_normalized_convlstm_forwards": utility_forwards,
                "balanced_score": score,
            }
        )

    best_tau = max(scored, key=lambda row: (row["balanced_score"], -row["tau"]))["tau"]
    return sorted(
        ({**row, "selected": row["tau"] == best_tau} for row in scored),
        key=lambda row: row["tau"],
    )


def selected_tau(scored_rows: list[dict]) -> float:
    return next(row["tau"] for row in scored_rows if row["selected"])


def write_score_csv(scored_rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUT_SCORE_STEM}.csv"
    fieldnames = [
        "tau",
        "mean_asr",
        "mean_fdr",
        "mean_final_accuracy",
        "mean_normalized_convlstm_forwards",
        "utility_asr",
        "utility_fdr",
        "utility_final_accuracy",
        "utility_normalized_convlstm_forwards",
        "balanced_score",
        "selected",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored_rows:
            writer.writerow(
                {
                    **{
                        key: (
                            f"{row[key]:.6f}" if isinstance(row[key], float) else row[key]
                        )
                        for key in fieldnames
                    },
                    "selected": int(row["selected"]),
                }
            )
    return out_path


def write_selected_rounds_csv(rows: list[dict], selected: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUT_ROUNDS_STEM}.csv"
    fieldnames = [
        "date",
        "dataset",
        "selected_tau",
        "attack",
        "round",
        "n",
        "n_malicious",
        "fn",
        "fp",
        "tp",
        "tn",
        "asr",
        "fdr",
        "final_accuracy",
        "normalized_convlstm_forwards",
        "run_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if abs(row["tau"] - selected) > 1e-9:
                continue
            for rnd, a in row["curve"].items():
                writer.writerow(
                    {
                        "date": row["date"],
                        "dataset": row["dataset"],
                        "selected_tau": f"{row['tau']:.1f}",
                        "attack": row["attack"],
                        "round": rnd,
                        "n": a["n"],
                        "n_malicious": a["n_malicious"],
                        "fn": a["fn"],
                        "fp": a["fp"],
                        "tp": a["tp"],
                        "tn": a["tn"],
                        "asr": f"{a['asr']:.5f}",
                        "fdr": f"{a['fdr']:.5f}",
                        "final_accuracy": f"{row['final_accuracy']:.5f}",
                        "normalized_convlstm_forwards": f"{row['normalized_convlstm_forwards']:.5f}",
                        "run_dir": row["run_dir"],
                    }
                )
    return out_path


def metric_ylim(rows: list[dict], dataset: str, tau: float, metric: str) -> tuple[float, float]:
    vals = [
        a[metric]
        for row in rows
        if row["dataset"] == dataset and abs(row["tau"] - tau) < 1e-9
        for a in row["curve"].values()
    ]
    top = max(vals) * 1.12 if vals and max(vals) > 0 else 1.0
    if metric == "asr":
        top = max(top, ASR_CEILING * 1.10)
    return -0.02 * top, top


def draw_panel(ax, rows: list[dict], dataset: str, tau: float, metric: str) -> None:
    ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0, zorder=0)
    ax.axvline(ONSET_ROUND, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    if metric == "asr":
        ax.axhline(
            ASR_CEILING,
            color="0.45",
            linestyle=(0, (1, 1.6)),
            linewidth=0.7,
            zorder=1,
        )

    for attack in ATTACKS:
        attack_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["attack"] == attack
            and abs(row["tau"] - tau) < 1e-9
        ]
        if not attack_rows:
            continue
        row = attack_rows[0]
        rounds = sorted(row["curve"])
        ax.plot(
            rounds,
            [row["curve"][rnd][metric] for rnd in rounds],
            color=ATTACK_COLORS[attack],
            linestyle=ATTACK_LINESTYLES[attack],
            marker=ATTACK_MARKERS[attack],
            markevery=5,
            markersize=2.8,
            linewidth=1.25,
            label=ATTACK_LABELS[attack],
            zorder=3,
        )

    ax.set_xlim(0.5, 50.5)
    ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
    ax.set_ylim(*metric_ylim(rows, dataset, tau, metric))
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_dataset(rows: list[dict], dataset: str, tau: float, out_dir: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_WIDTH_IN, 2.7),
        gridspec_kw={"wspace": 0.22},
    )
    panels = (
        ("asr", "Attack Success Rate"),
        ("fdr", "False Detection Rate"),
    )
    for ax, (metric, label) in zip(axes, panels):
        draw_panel(ax, rows, dataset, tau, metric)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(label)

    axes[0].annotate(
        "all attackers evade",
        xy=(9.0, ASR_CEILING - 0.012),
        ha="left",
        va="top",
        fontsize=IEEE_FONT - 2,
        color="0.35",
    )
    axes[0].annotate(
        "no attack",
        xy=(3.0, metric_ylim(rows, dataset, tau, "asr")[1] * 0.55),
        ha="center",
        va="center",
        rotation=90,
        fontsize=IEEE_FONT - 2,
        color="0.35",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=ATTACK_COLORS[attack],
            linestyle=ATTACK_LINESTYLES[attack],
            marker=ATTACK_MARKERS[attack],
            markersize=2.8,
            linewidth=1.25,
            label=ATTACK_LABELS[attack],
        )
        for attack in ATTACKS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            label=f"attack onset r={ONSET_ROUND}",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.suptitle(
        rf"{DATASET_LABELS[dataset]} at selected $\tau={tau:.1f}$",
        y=0.985,
        fontsize=IEEE_FONT,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.26)

    out_stem = f"best_tau_rounds_asr_fdr_{dataset}_tau{tau:.1f}"
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one global balanced tau and plot per-round ASR/FDR for each dataset."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = attach_run_level_metrics(collect_runs(args.start_date, args.end_date))
    if not rows:
        raise ValueError("No valid MNIST/Fashion-MNIST runs found.")

    check_geometry(rows)
    scored_rows = score_rows(aggregate_by_tau(rows))
    global_tau = selected_tau(scored_rows)

    score_path = write_score_csv(scored_rows, args.out_dir)
    rounds_path = write_selected_rounds_csv(rows, global_tau, args.out_dir)
    print(f"Loaded {len(rows)} runs.")
    print(f"Saved {score_path}")
    print(f"Saved {rounds_path}")
    best = next(row for row in scored_rows if row["selected"])
    print(
        f"Selected global tau={global_tau:.1f}: "
        f"score={best['balanced_score']:.4f}, "
        f"ASR={best['mean_asr']:.4f}, FDR={best['mean_fdr']:.4f}, "
        f"Acc={best['mean_final_accuracy']:.4f}, "
        f"NormFwd={best['mean_normalized_convlstm_forwards']:.4f}"
    )
    for dataset in DATASETS:
        plot_dataset(rows, dataset, global_tau, args.out_dir)


if __name__ == "__main__":
    main()
