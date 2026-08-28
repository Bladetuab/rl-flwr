"""Plot accuracy and ConvLSTM forwards vs tau_s from sweep_summary_perfect.json."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "sweep_figures"
SUMMARY_JSON = OUT_DIR / "sweep_summary_perfect.json"

ATTACK_COLORS = {
    "gaussian": "#d62728",
    "lie": "#2ca02c",
    "min_max": "#1f77b4",
}
ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie": "LIE",
    "min_max": "Min-Max",
}
MARKERS = {
    "gaussian": "o",
    "lie": "s",
    "min_max": "^",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.35,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_data() -> list[dict]:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"{SUMMARY_JSON} not found.")

    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    return sorted(records, key=lambda r: (r["attack_mode"], r["tau_s_quantile"]))


def plot_accuracy_vs_tau(records: list[dict]) -> None:
    fig, ax_acc = plt.subplots(figsize=(7.16, 3.15))
    ax_fwd = ax_acc.twinx()
    optimal_q = 0.6

    handles = []
    labels = []

    attack_modes = sorted({record["attack_mode"] for record in records})
    for attack in attack_modes:
        grp = [r for r in records if r["attack_mode"] == attack]
        x = [r["tau_s_quantile"] for r in grp]
        accuracy = [r["final_accuracy"] for r in grp]
        forwards = [r["total_convlstm_forwards"] for r in grp]
        attack_label = ATTACK_LABELS.get(attack, attack)

        acc_line, = ax_acc.plot(
            x,
            accuracy,
            color=ATTACK_COLORS[attack],
            marker=MARKERS[attack],
            linestyle="-",
        )
        fwd_line, = ax_fwd.plot(
            x,
            forwards,
            color=ATTACK_COLORS[attack],
            marker=MARKERS[attack],
            markerfacecolor="none",
            linestyle="--",
        )

        handles.extend([acc_line, fwd_line])
        labels.extend([f"{attack_label} acc.", f"{attack_label} fwd."])

    ax_acc.axvline(
        optimal_q,
        color="black",
        linestyle=":",
        linewidth=1.25,
        zorder=0,
    )
    ax_acc.annotate(
        r"Selected $q_s=0.6$",
        xy=(optimal_q, 0.92),
        xycoords=("data", "axes fraction"),
        xytext=(7, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )

    ax_acc.set_xlabel(r"Quantile threshold, $q_s$")
    ax_acc.set_ylabel("Final test accuracy")
    ax_fwd.set_ylabel("ConvLSTM forward passes")
    ax_acc.set_xlim(-0.05, 1.05)
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax_fwd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_acc.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax_acc.legend(
        handles,
        labels,
        loc="lower left",
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="none",
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.45,
    )

    fig.tight_layout()
    save(fig, "sweep_accuracy_vs_tau_perfect")


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=600)
        print(f"Saved -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_accuracy_vs_tau(load_data())
