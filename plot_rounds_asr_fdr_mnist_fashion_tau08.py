"""Plot per-round ASR/FDR for MNIST and Fashion-MNIST at tau = 0.8.

Data source:
    outputs/2026-07-09 .. outputs/2026-07-11

Metrics, computed per communication round from client_states_rounds.csv:
    ASR = FN / N = malicious clients missed by the detector / all clients
    FDR = FP / N = benign clients wrongly flagged by the detector / all clients

Usage:
    python plot_rounds_asr_fdr_mnist_fashion_tau08.py

Outputs:
    sweep_figures/rounds_asr_fdr_mnist_fashion_tau08_20260709_20260711.{png,pdf}
    sweep_figures/rounds_asr_fdr_mnist_fashion_tau08_20260709_20260711.csv
"""

from __future__ import annotations

import argparse
import csv
import os
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
    CLIENTS_PER_ROUND,
    DATASETS,
    DATASET_LABELS,
    DEFAULT_END_DATE,
    DEFAULT_OUT_DIR,
    DEFAULT_START_DATE,
    IEEE_FONT,
    IEEE_WIDTH_IN,
    MALICIOUS_PER_ROUND,
    ONSET_ROUND,
    check_geometry,
    collect_runs,
    metric_ylim,
)


TAU_VALUE = 0.8
OUT_STEM = "rounds_asr_fdr_mnist_fashion_tau08_20260709_20260711"

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


def filter_tau(rows: list[dict], tau: float) -> list[dict]:
    return [row for row in rows if abs(row["tau"] - tau) < 1e-9]


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUT_STEM}.csv"
    fieldnames = [
        "date",
        "dataset",
        "attack",
        "tau",
        "round",
        "n",
        "n_malicious",
        "fn",
        "fp",
        "tp",
        "tn",
        "asr",
        "fdr",
        "run_dir",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for rnd, a in row["curve"].items():
                writer.writerow(
                    {
                        "date": row["date"],
                        "dataset": row["dataset"],
                        "attack": row["attack"],
                        "tau": f"{row['tau']:.1f}",
                        "round": rnd,
                        "n": a["n"],
                        "n_malicious": a["n_malicious"],
                        "fn": a["fn"],
                        "fp": a["fp"],
                        "tp": a["tp"],
                        "tn": a["tn"],
                        "asr": f"{a['asr']:.5f}",
                        "fdr": f"{a['fdr']:.5f}",
                        "run_dir": row["run_dir"],
                    }
                )

    return out_path


def draw_panel(
    ax,
    rows: list[dict],
    dataset: str,
    metric: str,
    ylim: tuple[float, float],
) -> None:
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
        attack_rows = [row for row in rows if row["dataset"] == dataset and row["attack"] == attack]
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
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(IEEE_WIDTH_IN, 4.15),
        sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.18},
    )

    metrics = (
        ("asr", "Attack Success Rate"),
        ("fdr", "False Detection Rate"),
    )
    ylims = {metric: metric_ylim(rows, metric) for metric, _ in metrics}

    for col, dataset in enumerate(DATASETS):
        axes[0][col].set_title(DATASET_LABELS[dataset])
        for row_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx][col]
            draw_panel(ax, rows, dataset, metric, ylims[metric])
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.set_yticklabels([])
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Communication round")

    axes[0][0].annotate(
        "all attackers evade",
        xy=(9.0, ASR_CEILING - 0.012),
        ha="left",
        va="top",
        fontsize=IEEE_FONT - 2,
        color="0.35",
    )
    axes[0][0].annotate(
        "no attack",
        xy=(3.0, ylims["asr"][1] * 0.55),
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
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.suptitle(rf"Per-round ASR and FDR at $\tau={TAU_VALUE:.1f}$", y=0.985, fontsize=IEEE_FONT)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.18)

    for ext in ("png", "pdf"):
        out_path = out_dir / f"{OUT_STEM}.{ext}"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ASR/FDR over communication round for tau = 0.8."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tau", type=float, default=TAU_VALUE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_tau(collect_runs(args.start_date, args.end_date), args.tau)
    if not rows:
        raise ValueError(f"No matching MNIST/Fashion-MNIST tau={args.tau:.1f} runs found.")

    check_geometry(rows)
    csv_path = write_csv(rows, args.out_dir)
    print(
        f"Loaded {len(rows)} tau={args.tau:.1f} runs "
        f"({CLIENTS_PER_ROUND} clients/round, {MALICIOUS_PER_ROUND} malicious from r={ONSET_ROUND})."
    )
    print(f"Saved {csv_path}")
    plot(rows, args.out_dir)


if __name__ == "__main__":
    main()
