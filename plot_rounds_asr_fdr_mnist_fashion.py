"""Plot per-round attack success rate and false detection rate for MNIST/Fashion-MNIST.

Data source:
    outputs/2026-07-09 .. outputs/2026-07-11

Metrics, computed per communication round from client_states_rounds.csv:
    ASR = FN / N = malicious clients missed by the detector / all clients
    FDR = FP / N = benign clients wrongly flagged by the detector / all clients

For each dataset/attack/round, the solid curve is the median over the tau sweep and
the translucent band is the min-max range over tau.

Usage:
    python plot_rounds_asr_fdr_mnist_fashion.py

Outputs:
    sweep_figures/rounds_asr_fdr_mnist_fashion_20260709_20260711.{png,pdf}
    sweep_figures/rounds_asr_fdr_mnist_fashion_20260709_20260711.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_OUT_DIR = ROOT / "sweep_figures"

DEFAULT_START_DATE = "2026-07-09"
DEFAULT_END_DATE = "2026-07-11"
OUT_STEM = "rounds_asr_fdr_mnist_fashion_20260709_20260711"

DATASETS = ("mnist", "fashion_mnist")
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
}
DATASET_ORDER = {dataset: idx for idx, dataset in enumerate(DATASETS)}

ATTACKS = ("gaussian", "lie", "min_max")
ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie": "LIE",
    "min_max": "Min-Max",
}
ATTACK_COLORS = {
    "gaussian": "#D55E00",
    "lie": "#009E73",
    "min_max": "#0072B2",
}
ATTACK_ORDER = {attack: idx for idx, attack in enumerate(ATTACKS)}

RUN_RE = re.compile(
    r"^\d\d-\d\d-\d\d_(mnist|fashion_mnist)_"
    r"(gaussian|lie|min_max|no_attack)_tau([0-9.]+)$"
)

ONSET_ROUND = 6
CLIENTS_PER_ROUND = 100
MALICIOUS_PER_ROUND = 40
ASR_CEILING = MALICIOUS_PER_ROUND / CLIENTS_PER_ROUND

IEEE_WIDTH_IN = 7.16
IEEE_FONT = 8

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": IEEE_FONT,
        "axes.labelsize": IEEE_FONT,
        "axes.titlesize": IEEE_FONT,
        "xtick.labelsize": IEEE_FONT - 1,
        "ytick.labelsize": IEEE_FONT - 1,
        "legend.fontsize": IEEE_FONT - 1,
        "legend.title_fontsize": IEEE_FONT - 1,
        "lines.linewidth": 1.15,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


def iter_dates(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError(f"start date {start_date} is after end date {end_date}")

    days = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def per_round_metrics(run_dir: Path) -> dict[int, dict]:
    """Return per-round counts and ASR/FDR for one run."""
    path = run_dir / "client_states_rounds.csv"
    if not path.exists():
        return {}

    agg = defaultdict(lambda: {"fn": 0, "fp": 0, "tp": 0, "tn": 0, "n": 0, "n_malicious": 0})
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rnd = int(row["round"])
            is_malicious = int(row["is_malicious"])
            predicted_malicious = int(row["predict_state"])
            a = agg[rnd]
            a["n"] += 1
            a["n_malicious"] += is_malicious
            if is_malicious:
                if predicted_malicious:
                    a["tp"] += 1
                else:
                    a["fn"] += 1
            else:
                if predicted_malicious:
                    a["fp"] += 1
                else:
                    a["tn"] += 1

    return {
        rnd: {
            **a,
            "asr": a["fn"] / a["n"],
            "fdr": a["fp"] / a["n"],
        }
        for rnd, a in sorted(agg.items())
        if a["n"] > 0
    }


def collect_runs(start_date: str, end_date: str) -> list[dict]:
    rows = []
    for day in iter_dates(start_date, end_date):
        day_dir = OUTPUTS_DIR / day
        if not day_dir.exists():
            continue

        for run_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            match = RUN_RE.match(run_dir.name)
            if not match:
                continue
            dataset, attack, tau_text = match.groups()
            if attack == "no_attack":
                continue
            curve = per_round_metrics(run_dir)
            if not curve:
                continue
            rows.append(
                {
                    "date": day,
                    "dataset": dataset,
                    "attack": attack,
                    "tau": float(tau_text),
                    "n_rounds": len(curve),
                    "curve": curve,
                    "run_dir": str(run_dir),
                }
            )

    return dedupe_runs(rows)


def dedupe_runs(rows: list[dict]) -> list[dict]:
    """Prefer 50-round runs, then most rounds, then latest date for duplicate keys."""
    best: dict[tuple[str, str, float], dict] = {}

    def rank(row: dict) -> tuple[int, int, str, str]:
        return (
            1 if row["n_rounds"] == 50 else 0,
            row["n_rounds"],
            row["date"],
            row["run_dir"],
        )

    for row in rows:
        key = (row["dataset"], row["attack"], row["tau"])
        if key not in best or rank(row) > rank(best[key]):
            best[key] = row

    return sorted(
        best.values(),
        key=lambda row: (
            DATASET_ORDER[row["dataset"]],
            ATTACK_ORDER[row["attack"]],
            row["tau"],
        ),
    )


def check_geometry(rows: list[dict]) -> None:
    for row in rows:
        where = f"{row['dataset']}/{row['attack']}/tau={row['tau']:.1f}"
        for rnd, a in row["curve"].items():
            if a["n"] != CLIENTS_PER_ROUND:
                raise ValueError(f"{where} round {rnd}: expected 100 clients, got {a['n']}")
            expected_malicious = MALICIOUS_PER_ROUND if rnd >= ONSET_ROUND else 0
            if a["n_malicious"] != expected_malicious:
                raise ValueError(
                    f"{where} round {rnd}: expected {expected_malicious} malicious, "
                    f"got {a['n_malicious']}"
                )


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


def grouped_curves(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["attack"])].append(row)
    return grouped


def round_summary(group_rows: list[dict], metric: str) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    rounds = sorted({rnd for row in group_rows for rnd in row["curve"]})
    values = np.array(
        [
            [row["curve"][rnd][metric] for rnd in rounds]
            for row in group_rows
            if all(rnd in row["curve"] for rnd in rounds)
        ],
        dtype=float,
    )
    return rounds, np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0)


def metric_ylim(rows: list[dict], metric: str) -> tuple[float, float]:
    vals = [a[metric] for row in rows for a in row["curve"].values()]
    top = max(vals) * 1.12 if vals and max(vals) > 0 else 1.0
    if metric == "asr":
        top = max(top, ASR_CEILING * 1.10)
    return -0.02 * top, top


def draw_panel(ax, rows: list[dict], dataset: str, metric: str, ylim: tuple[float, float]) -> None:
    ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0, zorder=0)
    ax.axvline(ONSET_ROUND, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    if metric == "asr":
        ax.axhline(ASR_CEILING, color="0.45", linestyle=(0, (1, 1.6)), linewidth=0.7, zorder=1)

    grouped = grouped_curves(rows)
    for attack in ATTACKS:
        group_rows = grouped.get((dataset, attack), [])
        if not group_rows:
            continue
        rounds, low, mid, high = round_summary(group_rows, metric)
        color = ATTACK_COLORS[attack]
        ax.fill_between(rounds, low, high, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(rounds, mid, color=color, linewidth=1.4, zorder=3)

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
        figsize=(IEEE_WIDTH_IN, 4.35),
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
        Line2D([0], [0], color=ATTACK_COLORS[attack], linewidth=1.4, label=ATTACK_LABELS[attack])
        for attack in ATTACKS
    ]
    handles.extend(
        [
            Patch(facecolor="0.35", alpha=0.12, label=r"min-max over $\tau$"),
            Line2D([0], [0], color="0.35", linestyle="--", linewidth=0.8, label=f"attack onset r={ONSET_ROUND}"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.18)

    for ext in ("png", "pdf"):
        out_path = out_dir / f"{OUT_STEM}.{ext}"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ASR/FDR over communication round for MNIST and Fashion-MNIST."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_runs(args.start_date, args.end_date)
    if not rows:
        raise ValueError("No matching MNIST/Fashion-MNIST runs found.")

    check_geometry(rows)
    csv_path = write_csv(rows, args.out_dir)
    print(f"Loaded {len(rows)} runs.")
    print(f"Saved {csv_path}")
    plot(rows, args.out_dir)


if __name__ == "__main__":
    main()
