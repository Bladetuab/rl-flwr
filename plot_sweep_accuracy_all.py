"""Plot final accuracy vs quantile for every dataset and attack.

Reads the July 2026-07-06..07-11 sweep (current 5-channel no-valdelta detector) via
``plot_sweep_asr_fdr.collect()`` -- the same deduped 50-round run set used by the
ASR/FDR figures (``plot_sweep_asr_fdr_all.py``), so accuracy / ASR / FDR form a
matched trio.  ``final_accuracy`` is each run's last-round server accuracy from
results.json, identical in definition to the ``final_accuracy`` column of the older
sweep-summary CSV this script used to read.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede the pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Single source of the deduped July run set + per-run metrics (accuracy, forwards).
from plot_sweep_asr_fdr import collect

OUT_DIR = Path("sweep_figures")
OUT_NAME_ACCURACY = "sweep_accuracy_all_datasets_attacks"
OUT_NAME_DUAL = "sweep_accuracy_forwards_all_datasets_attacks"

DATASET_COLORS = {
    "mnist": "#1f77b4",
    "fashion_mnist": "#2ca02c",
    "cifar10": "#d62728",
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}

ATTACK_STYLES = {
    "gaussian": {"label": "Gaussian", "linestyle": "-", "marker": "o"},
    "lie": {"label": "LIE", "linestyle": "--", "marker": "s"},
    "min_max": {"label": "Min-Max", "linestyle": "-.", "marker": "^"},
}

IEEE_WIDTH_IN = 7.16
IEEE_FONT_SIZE = 8

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": IEEE_FONT_SIZE,
        "axes.labelsize": IEEE_FONT_SIZE,
        "axes.titlesize": IEEE_FONT_SIZE,
        "xtick.labelsize": IEEE_FONT_SIZE - 1,
        "ytick.labelsize": IEEE_FONT_SIZE - 1,
        "legend.fontsize": IEEE_FONT_SIZE - 1,
        "legend.title_fontsize": IEEE_FONT_SIZE - 1,
        "lines.linewidth": 1.15,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


def load_data() -> pd.DataFrame:
    """Build the accuracy table from the July sweep (same runs as the ASR/FDR plots).

    ``collect()`` returns {(dataset, attack): [{tau, final_accuracy, forwards, ...}]}
    for the deduped 50-round runs; flatten it to the columns the plotters expect
    (``dataset``, ``attack_mode``, ``tau_s_quantile``, ``final_accuracy``,
    ``total_convlstm_forwards``).
    """
    # CIFAR-10 is excluded: it is not an evaluated benchmark in the paper.
    KEEP_DATASETS = {"mnist", "fashion_mnist"}

    # collect() supplies (tau, forwards) but no longer carries final_accuracy.
    # Join it against final_accuracy_by_dataset.csv -- the same per-run table the
    # paper's Table IV is sourced from -- keyed on (dataset, attack, tau).
    import csv as _csv, re as _re
    _DS = {"MNIST": "mnist", "Fashion-MNIST": "fashion_mnist"}
    _AT = {"Gaussian": "gaussian", "LIE": "lie", "Min-Max": "min_max"}
    acc_lookup = {}
    with open(Path(__file__).parent / "final_accuracy_by_dataset.csv") as fh:
        for row in _csv.DictReader(fh):
            m = _re.match(r"Our Method \((.+?), tau=([0-9.]+)\)$", row["label"])
            if not m:
                continue
            ds, tau = _DS.get(row["dataset"]), float(m.group(2))
            at = _AT.get(m.group(1))
            if ds and at:
                acc_lookup[(ds, at, round(tau, 2))] = float(row["final_accuracy"])

    rows = []
    for (dataset, attack), recs in collect().items():
        if dataset not in KEEP_DATASETS:
            continue
        for r in recs:
            acc = acc_lookup.get((dataset, attack, round(float(r["tau"]), 2)))
            if acc is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "attack_mode": attack,
                    "tau_s_quantile": float(r["tau"]),
                    "final_accuracy": acc,
                    "total_convlstm_forwards": r["forwards"],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "attack_mode", "tau_s_quantile"])


def _dataset_handles(df: pd.DataFrame) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.6,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset, color in DATASET_COLORS.items()
        if dataset in set(df["dataset"])
    ]


def _attack_handles(df: pd.DataFrame, with_lines: bool = False) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=style["linestyle"] if with_lines else "None",
            marker=style["marker"],
            linewidth=1.15,
            markersize=3.8,
            label=style["label"],
        )
        for attack, style in ATTACK_STYLES.items()
        if attack in set(df["attack_mode"])
    ]


def _single_row_legend(ax, handles: list[Line2D], anchor_y: float = -0.2) -> None:
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=len(handles),
        frameon=False,
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def plot_accuracy(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_IN, 3.15))

    for (dataset, attack), grp in df.groupby(["dataset", "attack_mode"]):
        grp = grp.sort_values("tau_s_quantile")
        style = ATTACK_STYLES.get(
            attack, {"label": attack, "linestyle": "-", "marker": "o"}
        )
        ax.plot(
            grp["tau_s_quantile"],
            grp["final_accuracy"],
            color=DATASET_COLORS.get(dataset, "black"),
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.15,
            markersize=3.4,
            alpha=0.95,
        )

    ax.set_xlabel("Quantile ($q_s$)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Final Accuracy vs Quantile Across Datasets and Attacks")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _single_row_legend(ax, _dataset_handles(df) + _attack_handles(df, with_lines=True))
    fig.subplots_adjust(bottom=0.22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{OUT_NAME_ACCURACY}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_accuracy_and_forwards(df: pd.DataFrame) -> None:
    fig, ax_acc = plt.subplots(figsize=(IEEE_WIDTH_IN, 3.25))
    ax_fwd = ax_acc.twinx()

    for (dataset, attack), grp in df.groupby(["dataset", "attack_mode"]):
        grp = grp.sort_values("tau_s_quantile")
        color = DATASET_COLORS.get(dataset, "black")
        style = ATTACK_STYLES.get(
            attack, {"label": attack, "linestyle": "-", "marker": "o"}
        )

        ax_acc.plot(
            grp["tau_s_quantile"],
            grp["final_accuracy"],
            color=color,
            marker=style["marker"],
            linestyle="-",
            linewidth=1.15,
            markersize=3.4,
            alpha=0.95,
        )
        ax_fwd.plot(
            grp["tau_s_quantile"],
            grp["total_convlstm_forwards"],
            color=color,
            marker=style["marker"],
            linestyle=":",
            linewidth=1.0,
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=0.8,
            alpha=0.8,
        )

    ax_acc.set_xlabel("Quantile ($q_s$)")
    ax_acc.set_ylabel("Final Accuracy")
    ax_fwd.set_ylabel("Total ConvLSTM Forwards")
    ax_acc.set_title(
        "Final Accuracy and Total ConvLSTM Forwards vs Quantile"
    )

    ax_acc.set_xlim(-0.03, 1.03)
    ax_acc.set_ylim(0.0, 1.0)
    ax_fwd.set_ylim(0.0, df["total_convlstm_forwards"].max() * 1.1)
    ax_acc.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax_acc.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax_fwd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            marker="o",
            linewidth=1.15,
            markersize=3.4,
            label="Acc.",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=":",
            marker="o",
            markerfacecolor="white",
            linewidth=1.0,
            markersize=3.4,
            label="Fwd.",
        ),
    ]
    handles = _dataset_handles(df) + _attack_handles(df) + metric_handles
    _single_row_legend(ax_acc, handles)
    fig.subplots_adjust(bottom=0.22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{OUT_NAME_DUAL}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    plot_accuracy(data)
    plot_accuracy_and_forwards(data)
