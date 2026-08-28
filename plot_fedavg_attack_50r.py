"""plot_fedavg_attack_50r.py  --  undefended FedAvg under attack, matched 50-round protocol.

Regenerates the paper's Fig. 1. The original was drawn from the 30-round / 5-local-epoch
campaign; undefended FedAvg has since been run at 50 rounds / 10 local epochs for all three
attacks plus an adversary-free control, so this figure is protocol-matched to Table IV and
to Figs. 2-4.

  x-axis : communication round (1..50)
  y-axis : global test accuracy (evaluate_metrics_serverapp.accuracy in results.json)
  colour : attack, with the adversary-free control in neutral grey -- it is a reference,
           not a fourth attack, so it does not take a categorical hue.
  cols   : dataset

Provenance is the run-DIR name, reached through the sweep summary's `save_path` column, so
a curve cannot disagree with its table entry.

Figure (sweep_figures/):
    fedavg_attack_50r.{png,pdf}

Usage (from inside rl-flwr/):
    python plot_fedavg_attack_50r.py
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sweep_figures"
SUMMARY = ROOT / "outputs" / "sweep_summary_20260817_174855.csv"

DATASETS = ["mnist", "fashion_mnist"]
SERIES = ["gaussian", "lie", "min_max", "no_attack"]

DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}
SERIES_LABELS = {
    "gaussian": "Gaussian", "lie": "LIE", "min_max": "Min-Max",
    "no_attack": "No attack (reference)",
}
# Okabe-Ito for the three attacks; the control is neutral because it encodes "absence of
# attack" rather than another category.
SERIES_COLORS = {
    "gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2",
    "no_attack": "#666666",
}
SERIES_STYLE = {"gaussian": "-", "lie": "-", "min_max": "-", "no_attack": (0, (4, 2))}

ONSET_ROUND = 6
N_ROUNDS = 50
IEEE_WIDTH_IN = 7.16
IEEE_FONT = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": IEEE_FONT, "axes.labelsize": IEEE_FONT, "axes.titlesize": IEEE_FONT,
    "xtick.labelsize": IEEE_FONT - 1, "ytick.labelsize": IEEE_FONT - 1,
    "legend.fontsize": IEEE_FONT - 1,
    "figure.dpi": 300, "savefig.dpi": 600,
})


def accuracy_curve(run_dir: Path) -> dict:
    f = run_dir / "results.json"
    if not f.exists():
        return {}
    return {
        int(e["round"]): float(e["evaluate_metrics_serverapp"]["accuracy"])
        for e in json.load(open(f))
        if e.get("evaluate_metrics_serverapp", {}).get("accuracy") is not None
    }


def collect() -> dict:
    out = {}
    assert SUMMARY.exists(), f"missing summary: {SUMMARY}"
    for row in csv.DictReader(open(SUMMARY)):
        if row.get("error") or row["dataset"] not in DATASETS:
            continue
        if row["attack_mode"] not in SERIES:
            continue
        curve = accuracy_curve(Path(row["save_path"]))
        if curve:
            out[(row["dataset"], row["attack_mode"])] = curve
    missing = [(d, a) for d in DATASETS for a in SERIES if (d, a) not in out]
    assert not missing, f"no run for: {missing}"
    for key, c in out.items():
        assert len(c) == N_ROUNDS, f"{key}: {len(c)} rounds, expected {N_ROUNDS}"
    print(f"  geometry OK on all {len(out)} curves: rounds 1..{N_ROUNDS}")
    return out


def plot(data: dict) -> None:
    rounds = list(range(1, N_ROUNDS + 1))
    fig, axes = plt.subplots(
        1, len(DATASETS), figsize=(IEEE_WIDTH_IN, 2.35), sharey=True, layout="constrained"
    )
    for ax, dataset in zip(axes, DATASETS):
        ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0, zorder=0)
        ax.axvline(ONSET_ROUND, color="0.35", linewidth=0.7, zorder=1)
        ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for s in SERIES:
            c = data[(dataset, s)]
            ax.plot(rounds, [c[r] for r in rounds], color=SERIES_COLORS[s],
                    linestyle=SERIES_STYLE[s], linewidth=1.2, zorder=3,
                    solid_capstyle="round")
        ax.set_xlim(0.5, N_ROUNDS + 0.5)
        ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy")

    handles = [
        Line2D([0], [0], color=SERIES_COLORS[s], linestyle=SERIES_STYLE[s],
               linewidth=1.5, label=SERIES_LABELS[s])
        for s in SERIES
    ] + [Patch(facecolor="0.5", alpha=0.10, label="Warm-up rounds")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               columnspacing=1.3, handlelength=2.0, handletextpad=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"fedavg_attack_50r.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    data = collect()
    print("\n  Final-round accuracy (must match Table IV):")
    for dataset in DATASETS:
        vals = {s: data[(dataset, s)][N_ROUNDS] for s in SERIES}
        print(f"  {dataset:14s} " + "  ".join(f"{s}={v:.4f}" for s, v in vals.items()))
    plot(data)
    print(f"\nFigure saved under: {OUT_DIR.resolve()}")
