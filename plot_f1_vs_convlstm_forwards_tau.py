"""Plot detector F1-score vs total ConvLSTM forward passes, with tau as labels.

Data source:
    outputs/2026-07-09 .. outputs/2026-07-11

Datasets:
    mnist, fashion_mnist

Metrics:
    F1-score is the binary detector F1 for the malicious class, computed from
    client_states_rounds.csv:

        TP = is_malicious == 1 and predict_state == 1
        FP = is_malicious == 0 and predict_state == 1
        FN = is_malicious == 1 and predict_state == 0
        F1 = 2 * TP / (2 * TP + FP + FN)

    Total ConvLSTM forward passes is sum(n_scored) from convlstm_cost.csv.

The no_attack runs are excluded by default because they have no malicious positives,
so detector F1 is not meaningful for them.

Usage:
    python plot_f1_vs_convlstm_forwards_tau.py

Output:
    One IEEE-style double-column figure per dataset:
        sweep_figures/f1_vs_convlstm_forwards_tau_mnist.{png,pdf}
        sweep_figures/f1_vs_convlstm_forwards_tau_fashion_mnist.{png,pdf}
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

ATTACKS = ("gaussian", "lie", "min_max")
ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie": "LIE",
    "min_max": "Min-Max",
}
ATTACK_MARKERS = {
    "gaussian": "o",
    "lie": "s",
    "min_max": "^",
}

RUN_RE = re.compile(
    r"^\d\d-\d\d-\d\d_(mnist|fashion_mnist)_"
    r"(gaussian|lie|min_max|no_attack)_tau([0-9.]+)$"
)

IEEE_WIDTH_IN = 7.16
IEEE_DATASET_HEIGHT_IN = 2.45
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


def read_detector_counts(run_dir: Path) -> dict[str, int]:
    """Aggregate detector confusion counts across all logged client-round rows."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
    path = run_dir / "client_states_rounds.csv"
    if not path.exists():
        return counts

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            is_malicious = int(row["is_malicious"])
            predicted_malicious = int(row["predict_state"])
            counts["n"] += 1

            if is_malicious == 1 and predicted_malicious == 1:
                counts["tp"] += 1
            elif is_malicious == 0 and predicted_malicious == 1:
                counts["fp"] += 1
            elif is_malicious == 1 and predicted_malicious == 0:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

    return counts


def read_total_convlstm_forwards(run_dir: Path) -> int | None:
    """Return sum(n_scored) from convlstm_cost.csv."""
    path = run_dir / "convlstm_cost.csv"
    if not path.exists():
        return None

    with path.open(newline="") as f:
        return sum(int(row["n_scored"]) for row in csv.DictReader(f))


def read_run_summary(run_dir: Path) -> tuple[int, float | None]:
    """Return (n_rounds, final server accuracy), if results.json is present."""
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
    return len(results), final_metrics.get("accuracy")


def f1_from_counts(counts: dict[str, int]) -> float | None:
    """Binary F1 for the malicious class."""
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    if denom == 0:
        return None
    return 2 * counts["tp"] / denom


def precision_from_counts(counts: dict[str, int]) -> float | None:
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else None


def recall_from_counts(counts: dict[str, int]) -> float | None:
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else None


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

            counts = read_detector_counts(run_dir)
            total_forwards = read_total_convlstm_forwards(run_dir)
            f1 = f1_from_counts(counts)
            if counts["n"] == 0 or total_forwards is None or f1 is None:
                continue

            n_rounds, final_accuracy = read_run_summary(run_dir)
            rows.append(
                {
                    "date": day,
                    "dataset": dataset,
                    "attack": attack,
                    "tau": float(tau_text),
                    "f1_score": f1,
                    "precision": precision_from_counts(counts),
                    "recall": recall_from_counts(counts),
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "tn": counts["tn"],
                    "total_client_rows": counts["n"],
                    "total_convlstm_forwards": total_forwards,
                    "n_rounds": n_rounds,
                    "final_accuracy": final_accuracy,
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
        key=lambda r: (r["dataset"], r["attack"], r["tau"]),
    )


def write_metrics_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "f1_vs_convlstm_forwards_tau_mnist_fashion_mnist.csv"
    fieldnames = [
        "date",
        "dataset",
        "attack",
        "tau",
        "f1_score",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "tn",
        "total_client_rows",
        "total_convlstm_forwards",
        "n_rounds",
        "final_accuracy",
        "run_dir",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def plot_f1_vs_forwards(rows: list[dict], out_dir: Path, annotate_tau: bool = True) -> None:
    if not rows:
        raise ValueError("No valid rows found for plotting.")

    out_dir.mkdir(parents=True, exist_ok=True)
    limits = axis_limits(rows)
    for dataset in DATASETS:
        plot_dataset_f1_vs_forwards(
            rows=rows,
            dataset=dataset,
            out_dir=out_dir,
            limits=limits,
            annotate_tau=annotate_tau,
        )


def axis_limits(rows: list[dict]) -> dict[str, float]:
    """Common axis limits so MNIST and Fashion-MNIST figures are comparable."""
    all_f1 = [row["f1_score"] for row in rows]
    f1_pad = max(0.01, (max(all_f1) - min(all_f1)) * 0.08)
    x_min = max(0.0, np.floor((min(all_f1) - f1_pad) * 20.0) / 20.0)
    x_max = min(1.0, np.ceil((max(all_f1) + f1_pad) * 20.0) / 20.0)

    all_forwards = [row["total_convlstm_forwards"] for row in rows]
    y_pad = max(100, int((max(all_forwards) - min(all_forwards)) * 0.06))
    y_min = max(0, min(all_forwards) - y_pad)
    y_max = max(all_forwards) + y_pad

    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def plot_dataset_f1_vs_forwards(
    rows: list[dict],
    dataset: str,
    out_dir: Path,
    limits: dict[str, float],
    annotate_tau: bool = True,
) -> None:
    """Save one IEEE double-column figure for one dataset."""
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    if not dataset_rows:
        return

    fig, axes = plt.subplots(
        1,
        len(ATTACKS),
        figsize=(IEEE_WIDTH_IN, IEEE_DATASET_HEIGHT_IN),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    tau_ticks = np.arange(0.0, 1.01, 0.2)
    text_offsets = ((3, 3), (3, -8), (-12, 3), (-12, -8))

    for col_idx, attack in enumerate(ATTACKS):
        ax = axes[col_idx]
        attack_rows = sorted(
            [row for row in dataset_rows if row["attack"] == attack],
            key=lambda row: row["tau"],
        )
        if attack_rows:
            ax.plot(
                [row["f1_score"] for row in attack_rows],
                [row["total_convlstm_forwards"] for row in attack_rows],
                color="0.65",
                linewidth=0.55,
                alpha=0.75,
                zorder=1,
            )
            ax.scatter(
                [row["f1_score"] for row in attack_rows],
                [row["total_convlstm_forwards"] for row in attack_rows],
                c=[row["tau"] for row in attack_rows],
                cmap=cmap,
                norm=norm,
                marker=ATTACK_MARKERS[attack],
                s=24,
                linewidth=0.45,
                edgecolors="0.20",
                zorder=3,
            )

            if annotate_tau:
                for row in attack_rows:
                    idx = int(round(row["tau"] * 10)) % len(text_offsets)
                    dx, dy = text_offsets[idx]
                    ax.annotate(
                        f"{row['tau']:.1f}",
                        (row["f1_score"], row["total_convlstm_forwards"]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=IEEE_FONT - 2,
                        color="0.18",
                    )

        ax.set_title(ATTACK_LABELS[attack])
        ax.set_xlim(limits["x_min"], limits["x_max"])
        ax.set_ylim(limits["y_min"], limits["y_max"])
        ax.set_xticks(np.arange(limits["x_min"], limits["x_max"] + 1e-9, 0.1))
        ax.grid(axis="both", linestyle="--", linewidth=0.45, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if col_idx == 0:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

    fig.suptitle(DATASET_LABELS[dataset], fontsize=IEEE_FONT)
    fig.supxlabel("F1-score")
    fig.supylabel("Total ConvLSTM forward passes")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, ticks=tau_ticks, pad=0.02)
    cbar.set_label(r"$\tau$")

    for ext in ("png", "pdf"):
        out_path = out_dir / f"f1_vs_convlstm_forwards_tau_{dataset}.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot F1-score vs total ConvLSTM forwards with tau values."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-no-attack",
        action="store_true",
        help="Include no_attack runs if their F1-score is defined.",
    )
    parser.add_argument(
        "--no-annotate-tau",
        action="store_true",
        help="Use only the colorbar for tau values instead of text labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_runs(
        start_date=args.start_date,
        end_date=args.end_date,
        include_no_attack=args.include_no_attack,
    )
    csv_path = write_metrics_csv(rows, args.out_dir)
    print(f"Loaded {len(rows)} runs.")
    print(f"Saved {csv_path}")
    plot_f1_vs_forwards(rows, args.out_dir, annotate_tau=not args.no_annotate_tau)


if __name__ == "__main__":
    main()
