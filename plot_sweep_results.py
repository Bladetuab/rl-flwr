"""plot_sweep_results.py
Reads outputs/sweep_summary.csv and generates publication-quality figures
showing the accuracy–computation trade-off of the adaptive re-evaluation
policy across all (attack, tau) combinations.

Figures produced (saved to outputs/sweep_figures/):
  1. sweep_accuracy_vs_tau.pdf/.png
       – Accuracy vs tau_s for each attack type (single plot, three lines)
  2. sweep_forwards_vs_tau.pdf/.png
       – Total ConvLSTM forward passes vs tau_s for each attack type
  3. sweep_combined_{attack}.pdf/.png  (one per attack)
       – Dual-axis: accuracy (left) + total forwards (right) vs tau_s
  4. sweep_ratio_scored_vs_tau.pdf/.png
       – Mean ratio of clients scored per round vs tau_s for each attack
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV  = Path("/home/tuan/get-local-model-bt-id/rl-flwr/outputs/sweep_summary_20260826_191104.csv")
OUT_DIR      = Path("/home/tuan/get-local-model-bt-id/rl-flwr/") / "sweep_figures"
SUMMARY_JSON = OUT_DIR / "sweep_summary.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
ATTACK_COLORS = {
    "gaussian": "#e6194b",   # red
    "lie":      "#3cb44b",   # green
    "min_max":  "#4363d8",   # blue
}
ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie":      "LIE",
    "min_max":  "Min-Max",
}
MARKERS = {
    "gaussian": "o",
    "lie":      "s",
    "min_max":  "^",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi":      150,
})


def load_data() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run sweep_tau_attack.py first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(["attack_mode", "tau_s_quantile"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Accuracy vs tau_s  (three attack lines, single axis)
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_vs_tau(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for attack, grp in df.groupby("attack_mode"):
        ax.plot(
            grp["tau_s_quantile"], grp["final_accuracy"],
            color=ATTACK_COLORS[attack],
            marker=MARKERS[attack],
            label=ATTACK_LABELS.get(attack, attack),
        )
        for _, row in grp.iterrows():
            ax.annotate(
                f"{row['final_accuracy']:.3f}",
                (row["tau_s_quantile"], row["final_accuracy"]),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7.5,
                color=ATTACK_COLORS[attack],
            )

    ax.set_xlabel(r"Re-evaluation trigger quantile $q_s$")
    ax.set_ylabel("Final-Round Accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Attack", loc="lower left")
    ax.set_title(r"Accuracy vs. $q_s$ for Each Attack Type")

    fig.tight_layout()
    _save(fig, "sweep_accuracy_vs_tau")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Total ConvLSTM forwards vs tau_s
# ─────────────────────────────────────────────────────────────────────────────
def plot_forwards_vs_tau(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for attack, grp in df.groupby("attack_mode"):
        ax.plot(
            grp["tau_s_quantile"], grp["total_convlstm_forwards"],
            color=ATTACK_COLORS[attack],
            marker=MARKERS[attack],
            linestyle="--",
            label=ATTACK_LABELS.get(attack, attack),
        )

    ax.set_xlabel(r"Re-evaluation trigger quantile $q_s$")
    ax.set_ylabel("Total ConvLSTM Forward Passes")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Attack", loc="upper right")
    ax.set_title(r"ConvLSTM Inference Cost vs. $q_s$")

    fig.tight_layout()
    _save(fig, "sweep_forwards_vs_tau")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Dual-axis per attack (accuracy + forwards on same plot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined_per_attack(df: pd.DataFrame):
    attacks = df["attack_mode"].unique()

    for attack in attacks:
        grp = df[df["attack_mode"] == attack].copy()
        color_acc  = "#1f77b4"   # blue for accuracy
        color_fwd  = "#d62728"   # red for forwards

        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        ax2 = ax1.twinx()

        # Accuracy (left axis)
        l1, = ax1.plot(
            grp["tau_s_quantile"], grp["final_accuracy"],
            color=color_acc, marker="o", label="Final Accuracy",
        )
        for _, row in grp.iterrows():
            ax1.annotate(
                f"{row['final_accuracy']:.3f}",
                (row["tau_s_quantile"], row["final_accuracy"]),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5, color=color_acc,
            )

        # Total forwards (right axis)
        l2, = ax2.plot(
            grp["tau_s_quantile"], grp["total_convlstm_forwards"],
            color=color_fwd, marker="s", linestyle="--",
            label="Total ConvLSTM Forwards",
        )

        ax1.set_xlabel(r"Re-evaluation trigger quantile $q_s$")
        ax1.set_ylabel("Final-Round Accuracy", color=color_acc)
        ax2.set_ylabel("Total ConvLSTM Forward Passes", color=color_fwd)
        ax1.tick_params(axis="y", labelcolor=color_acc)
        ax2.tick_params(axis="y", labelcolor=color_fwd)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))

        ax1.set_xlim(-0.05, 1.05)
        ax1.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        label = ATTACK_LABELS.get(attack, attack)
        ax1.set_title(
            rf"Accuracy–Computation Trade-off ({label} Attack, $q_s$ Sweep)"
        )

        lines = [l1, l2]
        ax1.legend(lines, [l.get_label() for l in lines], loc="center left")

        fig.tight_layout()
        _save(fig, f"sweep_combined_{attack}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Mean ratio scored per round vs tau_s
# ─────────────────────────────────────────────────────────────────────────────
def plot_ratio_scored_vs_tau(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for attack, grp in df.groupby("attack_mode"):
        ax.plot(
            grp["tau_s_quantile"], grp["mean_ratio_scored"] * 100,
            color=ATTACK_COLORS[attack],
            marker=MARKERS[attack],
            label=ATTACK_LABELS.get(attack, attack),
        )

    ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="Full scoring (100%)")
    ax.set_xlabel(r"Re-evaluation trigger quantile $q_s$")
    ax.set_ylabel("Mean \% of Clients Scored per Round")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 110)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Attack", loc="upper right")
    ax.set_title(r"Fraction of Clients Scored per Round vs. $q_s$")

    fig.tight_layout()
    _save(fig, "sweep_ratio_scored_vs_tau")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – q on Y-axis, twin X (accuracy bottom, forwards top)
# ─────────────────────────────────────────────────────────────────────────────
def plot_q_vs_acc_and_forwards():
    """q on x-axis; accuracy on left y (solid); ConvLSTM forwards on right y (dashed).
    One curve per attack on each axis (6 lines total).
    Reads from SUMMARY_JSON, not the CSV."""
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(
            f"{SUMMARY_JSON} not found. Run export_summary_to_json() first."
        )
    with open(SUMMARY_JSON) as f:
        records = json.load(f)
    df = pd.DataFrame(records).sort_values(["attack_mode", "tau_s_quantile"])

    fig, ax_acc = plt.subplots(figsize=(11, 5))
    ax_fwd = ax_acc.twinx()

    handles, labels = [], []
    for attack_key, grp in df.groupby("attack_mode"):
        attack = str(attack_key)
        grp = grp.sort_values("tau_s_quantile")
        color = ATTACK_COLORS[attack]
        atk_label = ATTACK_LABELS.get(attack, attack)

        l_acc, = ax_acc.plot(
            grp["tau_s_quantile"], grp["final_accuracy"],
            color=color, marker="o", linestyle="-", linewidth=2,
        )
        l_fwd, = ax_fwd.plot(
            grp["tau_s_quantile"], grp["total_convlstm_forwards"],
            color=color, marker="s", linestyle="--", linewidth=2,
            markerfacecolor="none",
        )
        handles += [l_acc, l_fwd]
        labels  += [f"{atk_label} – Acc", f"{atk_label} – Fwd"]

    ax_acc.set_xlabel(r"$q$")
    ax_acc.set_ylabel("Final-Round Accuracy")
    ax_fwd.set_ylabel("Total ConvLSTM Forward Passes")
    ax_fwd.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    ax_acc.set_xlim(-0.05, 1.05)
    ax_acc.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4)

    ax_acc.legend(handles, labels, loc="lower left", ncol=3, fontsize=9,
                  framealpha=0.9)

    fig.tight_layout()
    _save(fig, "sweep_q_vs_acc_and_forwards")


# ─────────────────────────────────────────────────────────────────────────────
# JSON export of the sweep summary
# ─────────────────────────────────────────────────────────────────────────────
def export_summary_to_json(df: pd.DataFrame):
    out = OUT_DIR / "sweep_summary.json"
    records = df.to_dict(orient="records")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name: str):
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


def print_summary_table(df: pd.DataFrame):
    print("\n── Sweep Summary ──────────────────────────────────────────────")
    print(df[[
        "attack_mode", "tau_s_quantile",
        "final_accuracy", "total_convlstm_forwards",
        "mean_ratio_scored",
    ]].to_string(index=False, float_format="{:.4f}".format))
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print_summary_table(df)

    print("\nExporting JSON …")
    export_summary_to_json(df)

    print("\nGenerating figures …")
    plot_accuracy_vs_tau(df)
    plot_forwards_vs_tau(df)
    plot_combined_per_attack(df)
    plot_ratio_scored_vs_tau(df)
    plot_q_vs_acc_and_forwards()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")