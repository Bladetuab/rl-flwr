"""
Plot sweep_tau_results.json — same style as sweep_anom_decay.py.

  - Left  y-axis : Final-round server accuracy
  - Right y-axis : Total ConvLSTM forward passes
  - q = 1.0 highlighted as the "EWMA-trigger OFF" baseline

Usage (from inside rl-flwr/):
    python plot_sweep_tau.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "sweep_tau_results.json"
PLOT_FILE    = Path(__file__).parent / "sweep_tau_quantile.png"
FIXED_ANOM_DECAY = 0.8


def make_plot(records: list[dict]) -> None:
    records  = sorted(records, key=lambda r: r["tau_s_quantile"])
    x        = [r["tau_s_quantile"] for r in records]
    acc      = [r["accuracy"] for r in records]
    fwd      = [r["total_convlstm_forwards"] for r in records]

    color_acc = "#2563EB"   # blue
    color_fwd = "#DC2626"   # red
    color_bl  = "#16A34A"   # green  — baseline marker

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ---- accuracy (left axis) ----
    ax1.plot(x, acc, marker="o", linewidth=2, color=color_acc, label="Final Accuracy", zorder=3)
    ax1.set_xlabel("q", fontsize=13)
    ax1.set_ylabel("Final-Round Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # ---- forwards (right axis) ----
    ax2 = ax1.twinx()
    ax2.plot(x, fwd, marker="s", linewidth=2, linestyle="--",
             color=color_fwd, label="Total ConvLSTM Forwards", zorder=3)
    ax2.set_ylabel("Total ConvLSTM Forward Passes", fontsize=12, color=color_fwd)
    ax2.tick_params(axis="y", labelcolor=color_fwd)

    # ---- highlight q = 1.0 baseline ----
    # baseline_idx = next((i for i, r in enumerate(records) if r["tau_s_quantile"] == 1.0), None)
    # if baseline_idx is not None:
    #     bx   = x[baseline_idx]
    #     bfwd = fwd[baseline_idx]
    #     bacc = acc[baseline_idx]

    #     ax1.axvline(bx, color=color_bl, linestyle=":", linewidth=1.6, alpha=0.7)

        # ax2.annotate(
        #     f"EWMA trigger OFF\n(q=1.0 baseline)\n{bfwd} forwards",
        #     xy=(bx, bfwd),
        #     xytext=(-90, -40),
        #     textcoords="offset points",
        #     fontsize=8,
        #     color=color_bl,
        #     arrowprops=dict(arrowstyle="->", color=color_bl, lw=1.2),
        # )

        # ax1.scatter([bx], [bacc], color=color_bl, s=80, zorder=5)
        # ax2.scatter([bx], [bfwd], color=color_bl, s=80, zorder=5)

    # ---- accuracy value labels ----
    for xi, yi in zip(x, acc):
        if not np.isnan(yi):
            ax1.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=color_acc,
            )

    # ---- EWMA marginal-cost annotation (cost above baseline) ----
    # if baseline_idx is not None:
    #     base_fwd = fwd[baseline_idx]
    #     for xi, fi in zip(x, fwd):
    #         delta = fi - base_fwd
    #         if delta > 0:
    #             ax2.annotate(
    #                 f"+{delta}",
    #                 xy=(xi, fi),
    #                 xytext=(0, 9),
    #                 textcoords="offset points",
    #                 ha="center",
    #                 fontsize=7,
    #                 color=color_fwd,
    #                 alpha=0.75,
    #             )

    # ---- legend ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    # baseline_handle = Line2D([0], [0], color=color_bl, linestyle=":", linewidth=1.6,
    #                          label="q=1.0 baseline (trigger OFF)")
    # ax1.legend(lines1 + lines2 + [baseline_handle],
    #            labels1 + labels2 + ["q=1.0 baseline (trigger OFF)"],
    #            loc="lower left", fontsize=9)

    # plt.title(
    #     f"Impact of EWMA Trigger (tau_s_quantile) on Accuracy & Compute Cost\n"
    #     f"(anom_decay fixed = {FIXED_ANOM_DECAY})",
    #     fontsize=13, fontweight="bold",
    # )
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved -> {PLOT_FILE}")


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {RESULTS_FILE}")
    make_plot(records)


if __name__ == "__main__":
    main()
