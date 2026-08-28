"""Plot tau sweeps with normalized ConvLSTM forwards and final accuracy.

Data source:
    outputs/2026-07-09 .. outputs/2026-07-11

Datasets:
    MNIST and Fashion-MNIST

Metrics:
    - Final test accuracy: last-round server accuracy from results.json.
    - Normalized Total ConvLSTM forwards: sum(n_scored) / sum(n_clients) from
      convlstm_cost.csv for the same run.

The default plot excludes no_attack runs because they do not form a tau sweep.

Usage:
    python plot_tau_accuracy_normalized_forwards.py

Output:
    sweep_figures/tau_accuracy_normalized_forwards_mnist.{png,pdf}
    sweep_figures/tau_accuracy_normalized_forwards_fashion_mnist.{png,pdf}
    sweep_figures/tau_accuracy_normalized_forwards_mnist_fashion_mnist.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_OUT_DIR = ROOT / "sweep_figures"

DEFAULT_START_DATE = "2026-07-09"
DEFAULT_END_DATE = "2026-07-11"

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
    "no_attack": "No Attack",
}
ATTACK_STYLES = {
    "gaussian": {"color": "#D55E00", "linestyle": "-", "marker": "o"},
    "lie": {"color": "#009E73", "linestyle": "--", "marker": "s"},
    "min_max": {"color": "#0072B2", "linestyle": "-.", "marker": "^"},
    "no_attack": {"color": "#666666", "linestyle": ":", "marker": "D"},
}
ATTACK_ORDER = {attack: idx for idx, attack in enumerate((*ATTACKS, "no_attack"))}

RUN_RE = re.compile(
    r"^\d\d-\d\d-\d\d_(mnist|fashion_mnist)_"
    r"(gaussian|lie|min_max|no_attack)_tau([0-9.]+)$"
)

IEEE_WIDTH_IN = 7.16
IEEE_SINGLE_COL_IN = 3.5
IEEE_SINGLE_COL_HEIGHT_IN = 2.8
IEEE_FONT = 8
OUT_STEM = "tau_accuracy_normalized_forwards_mnist_fashion_mnist"

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
    """Return inclusive ISO date strings from start_date to end_date."""
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


def read_final_accuracy(run_dir: Path) -> tuple[int, float | None]:
    """Return (round count, final server accuracy) from results.json."""
    path = run_dir / "results.json"
    if not path.exists():
        return 0, None

    try:
        with path.open() as f:
            results = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return 0, None

    if not isinstance(results, list) or not results:
        return 0, None

    final_metrics = results[-1].get("evaluate_metrics_serverapp", {})
    accuracy = final_metrics.get("accuracy")
    return len(results), float(accuracy) if accuracy is not None else None


def read_convlstm_forwards(run_dir: Path) -> tuple[int | None, float | None]:
    """Return (total forwards, normalized forwards) from convlstm_cost.csv."""
    path = run_dir / "convlstm_cost.csv"
    if not path.exists():
        return None, None

    total_scored = 0
    total_clients = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            total_scored += int(row["n_scored"])
            total_clients += int(row["n_clients"])

    if total_clients == 0:
        return total_scored, None
    return total_scored, total_scored / total_clients


def collect_runs(
    start_date: str,
    end_date: str,
    include_no_attack: bool = False,
) -> list[dict]:
    """Collect one row per valid run in the requested date range."""
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
            if attack == "no_attack" and not include_no_attack:
                continue

            n_rounds, final_accuracy = read_final_accuracy(run_dir)
            total_forwards, normalized_forwards = read_convlstm_forwards(run_dir)
            if (
                n_rounds == 0
                or final_accuracy is None
                or total_forwards is None
                or normalized_forwards is None
            ):
                continue

            rows.append(
                {
                    "date": day,
                    "dataset": dataset,
                    "attack": attack,
                    "tau": float(tau_text),
                    "final_accuracy": final_accuracy,
                    "total_convlstm_forwards": total_forwards,
                    "normalized_total_convlstm_forwards": normalized_forwards,
                    "n_rounds": n_rounds,
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
        key=lambda r: (
            DATASET_ORDER.get(r["dataset"], len(DATASET_ORDER)),
            ATTACK_ORDER.get(r["attack"], len(ATTACK_ORDER)),
            r["tau"],
        ),
    )


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUT_STEM}.csv"
    fieldnames = [
        "date",
        "dataset",
        "attack",
        "tau",
        "final_accuracy",
        "total_convlstm_forwards",
        "normalized_total_convlstm_forwards",
        "n_rounds",
        "run_dir",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def metric_limits(
    values: list[float],
    min_width: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Return padded limits, preserving a useful minimum visible width."""
    v_min = min(values)
    v_max = max(values)
    pad = max(min_width * 0.15, (v_max - v_min) * 0.16)
    low = max(lower, v_min - pad)
    high = min(upper, v_max + pad)

    if high - low < min_width:
        center = (high + low) / 2.0
        low = max(lower, center - min_width / 2.0)
        high = min(upper, center + min_width / 2.0)
        if high - low < min_width:
            if low == lower:
                high = min(upper, low + min_width)
            elif high == upper:
                low = max(lower, high - min_width)

    return low, high


def legend_handles(rows: list[dict]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=ATTACK_STYLES[attack]["color"],
            linestyle=ATTACK_STYLES[attack]["linestyle"],
            marker=ATTACK_STYLES[attack]["marker"],
            markersize=3.2,
            linewidth=1.15,
            label=ATTACK_LABELS[attack],
        )
        for attack in ATTACKS
        if any(row["attack"] == attack for row in rows)
    ]


def plot_rows(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        raise ValueError("No valid MNIST/Fashion-MNIST runs found for plotting.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        plot_dataset_rows(
            rows=[row for row in rows if row["dataset"] == dataset],
            dataset=dataset,
            out_dir=out_dir,
        )


def plot_dataset_rows(rows: list[dict], dataset: str, out_dir: Path) -> None:
    if not rows:
        return

    fig, ax_fwd = plt.subplots(
        figsize=(IEEE_SINGLE_COL_IN, IEEE_SINGLE_COL_HEIGHT_IN),
        constrained_layout=False,
    )
    ax_acc = ax_fwd.twinx()

    for attack in ATTACKS:
        attack_rows = sorted(
            [row for row in rows if row["attack"] == attack],
            key=lambda row: row["tau"],
        )
        if not attack_rows:
            continue

        style = ATTACK_STYLES[attack]
        tau = [r["tau"] for r in attack_rows]
        ax_fwd.plot(
            tau,
            [r["normalized_total_convlstm_forwards"] for r in attack_rows],
            color=style["color"],
            linestyle=":",
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=2.8,
            linewidth=1.0,
            alpha=0.95,
            zorder=3,
        )
        ax_acc.plot(
            tau,
            [r["final_accuracy"] for r in attack_rows],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            linewidth=1.15,
            zorder=4,
        )

    fwd_low, fwd_high = metric_limits(
        [row["normalized_total_convlstm_forwards"] for row in rows],
        min_width=0.1,
        lower=0.0,
        upper=1.0,
    )
    acc_low, acc_high = metric_limits(
        [row["final_accuracy"] for row in rows],
        min_width=0.05,
        lower=0.0,
        upper=1.0,
    )

    ax_fwd.set_xlim(-0.03, 1.03)
    ax_fwd.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax_fwd.set_ylim(fwd_low, fwd_high)
    ax_acc.set_ylim(acc_low, acc_high)
    ax_fwd.set_xlabel(r"Sweep $\tau$")
    ax_fwd.set_ylabel("Normalized ConvLSTM forwards")
    ax_acc.set_ylabel("Final test accuracy")
    ax_acc.set_title(DATASET_LABELS[dataset], pad=5)

    ax_fwd.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_fwd.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_acc.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_fwd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_fwd.tick_params(axis="both", length=2.2, width=0.6)
    ax_acc.tick_params(axis="y", length=2.2, width=0.6)
    ax_fwd.grid(axis="both", linestyle="--", linewidth=0.45, alpha=0.3)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            label="Accuracy",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=":",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=2.8,
            linewidth=1.0,
            label="Norm. forwards",
        ),
    ]
    handles = legend_handles(rows) + metric_handles
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.15, right=0.86, top=0.88, bottom=0.34)

    out_stem = f"tau_accuracy_normalized_forwards_{dataset}"
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot tau on the x-axis, normalized ConvLSTM forwards on the left y-axis, "
            "and final test accuracy on the right y-axis."
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-no-attack",
        action="store_true",
        help="Include no_attack runs if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_runs(
        start_date=args.start_date,
        end_date=args.end_date,
        include_no_attack=args.include_no_attack,
    )
    csv_path = write_csv(rows, args.out_dir)
    print(f"Loaded {len(rows)} runs.")
    print(f"Saved {csv_path}")
    plot_rows(rows, args.out_dir)


if __name__ == "__main__":
    main()
