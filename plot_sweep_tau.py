"""Twin-axis q_s sweep plot, driven by sweep_summary CSVs.

    x axis      scoring quantile  q_s
    left  y     final accuracy
    right y     ConvLSTM forward passes

Usage
-----
    # default: the 66-run ustad_af sweep, both datasets pooled
    python plot_sweep_tau.py

    # one dataset out of one campaign
    python plot_sweep_tau.py --csv outputs/sweep_summary_20260826_191104.csv \
                             --dataset mnist

    # several CSVs together (e.g. the MNIST and Fashion halves of one campaign)
    python plot_sweep_tau.py --csv outputs/sweep_summary_20260826_191104.csv \
                                   outputs/sweep_summary_20260826_213024.csv \
                             --dataset all

    # narrow to a single attack as well
    python plot_sweep_tau.py --dataset fashion_mnist --attack min_max

Notes on what the figure does and does not show
-----------------------------------------------
The two series sit on independently chosen scales, so where the curves cross is
an artifact of those scales and means nothing. Accuracy is plotted on a full
[0, 1] axis by default (--zoom-accuracy overrides): across these sweeps the
per-q_s accuracy means span well under one point, which is inside single-run
noise, and auto-scaling would magnify that noise into an apparent swing.

Accuracy therefore cannot select an operating point. The quantity that can --
worst-cell false-discovery rate over the attacked rounds -- is not on this
figure; see plot_sweep_tau_panels.py for the three-measure version.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ieee_style

ROOT = Path(__file__).parent
DEFAULT_CSV = ["outputs/sweep_summary_20260814_152111.csv"]

COLOR_ACC = "#2a78d6"   # blue    - left axis, accuracy
COLOR_FWD = "#eb6834"   # orange  - right axis, forwards
INK, INK_2, GRID = ieee_style.INK, ieee_style.INK_2, ieee_style.GRID

DATASET_LABEL = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST",
                 "cifar10": "CIFAR-10", "all": "both datasets"}


# ---------------------------------------------------------------------------
def load_rows(csv_paths, dataset, attack):
    """Read the summary CSVs and keep the rows the user asked for."""
    rows = []
    for c in csv_paths:
        path = c if os.path.isabs(c) else os.path.join(ROOT, c)
        if not os.path.exists(path):
            sys.exit(f"error: no such CSV: {c}")
        with open(path) as fh:
            for r in csv.DictReader(fh):
                r["_src"] = os.path.basename(path)
                rows.append(r)
    if not rows:
        sys.exit("error: the CSVs contained no rows")

    # A run that errored has no usable metrics; the sweep driver records the
    # row anyway so the failure is visible. Drop those rather than plot zeros.
    rows = [r for r in rows if not (r.get("error") or "").strip()]

    avail_ds = sorted({r["dataset"] for r in rows})
    avail_atk = sorted({r["attack_mode"] for r in rows})
    if dataset != "all":
        rows = [r for r in rows if r["dataset"] == dataset]
        if not rows:
            sys.exit(f"error: no rows for dataset {dataset!r}. "
                     f"These CSVs contain: {', '.join(avail_ds)}")
    if attack != "all":
        rows = [r for r in rows if r["attack_mode"] == attack]
        if not rows:
            sys.exit(f"error: no rows for attack {attack!r}. "
                     f"These CSVs contain: {', '.join(avail_atk)}")

    missing = [r for r in rows if not (r.get("total_convlstm_forwards") or "").strip()]
    if len(missing) == len(rows):
        sys.exit("error: these CSVs carry no total_convlstm_forwards column data, "
                 "so there is no detector cost to plot")
    return rows, avail_ds, avail_atk


def infer_scale(rows):
    """Recover (N, T) from a run directory named by save_path.

    The summary CSV records neither the federation size nor the horizon, and
    they differ between campaigns (N=100/T=50 for the ustad_af sweep, N=50/T=30
    for the ustad_cal ones), so the exhaustive-scoring reference N*T cannot be
    hardcoded. Read it from the first run whose directory still exists.
    """
    for r in rows:
        d = (r.get("save_path") or "").strip()
        if not d or not os.path.isdir(d):
            continue
        cs, rj = os.path.join(d, "client_states_rounds.csv"), os.path.join(d, "results.json")
        if not (os.path.exists(cs) and os.path.exists(rj)):
            continue
        try:
            T = max(x["round"] for x in json.load(open(rj)))
        except Exception:
            continue
        clients = {row["client_id"] for row in csv.DictReader(open(cs)) if row["round"] == "1"}
        if clients:
            return len(clients), T
    return None, None


def aggregate(rows):
    """Mean over the selected cells at each q_s."""
    by = collections.defaultdict(list)
    for r in rows:
        by[float(r["tau_s_quantile"])].append(r)
    out = []
    for t in sorted(by):
        g = by[t]
        num = lambda k: [float(r[k]) for r in g if (r.get(k) or "").strip()]
        out.append(dict(tau=t, n=len(g),
                        acc=statistics.mean(num("final_accuracy")),
                        fwd=statistics.mean(num("total_convlstm_forwards"))))
    return out


# ---------------------------------------------------------------------------
def make_plot(pts, exhaustive, caption, out_path, zoom_accuracy, sel):
    """Draw the figure. No embedded title: D.01 puts the caption below the
    figure via \\caption{} in LaTeX, so a title inside the axes would duplicate
    it in the wrong font at the wrong size. The caption text is printed for the
    author to paste instead."""
    ieee_style.apply(base=8.0)

    x = [p["tau"] for p in pts]
    acc = [p["acc"] for p in pts]
    fwd = [p["fwd"] for p in pts]

    fig, ax1 = plt.subplots(figsize=(ieee_style.COL_SINGLE, ieee_style.COL_SINGLE * 0.68))

    l1, = ax1.plot(x, acc, marker="o", color=COLOR_ACC,
                   label="Final accuracy (left)", zorder=3)
    ax1.set_xlabel("Scoring quantile $q_s$")
    ax1.set_ylabel("Final-round accuracy", color=COLOR_ACC)
    ax1.tick_params(axis="y", colors=COLOR_ACC)
    ieee_style.tighten_x(ax1, x)                       # H.06
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    if zoom_accuracy:
        ieee_style.tighten_y(ax1, acc, bottom=min(acc) * 0.999)
    else:
        ax1.set_ylim(0, 1.0, auto=False)               # H.07

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    l2, = ax2.plot(x, fwd, marker="s", linestyle=(0, (4, 2)), color=COLOR_FWD,
                   label="ConvLSTM forwards (right)", zorder=3)
    ax2.set_ylabel("ConvLSTM forward passes", color=COLOR_FWD)
    ax2.tick_params(axis="y", colors=COLOR_FWD)
    top = exhaustive if exhaustive else max(fwd)
    ax2.set_ylim(0, top, auto=False)                   # H.07
    if exhaustive:
        ax2.axhline(exhaustive, color=INK_2, linewidth=0.6,
                    linestyle=(0, (1.5, 2)), zorder=1)

    if sel is not None and sel in x:
        ax1.axvline(sel, color=INK_2, linewidth=0.6, zorder=1)

    # H.03: legend inside the axes, in the least dense corner. Accuracy runs
    # along the top and forwards descends across the middle, so the lower-left
    # quadrant is the only region no series enters.
    ax1.legend(handles=[l1, l2], loc="lower left", fontsize=8,
               handlelength=2.0, borderaxespad=0.4, labelspacing=0.3)

    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print("\nLaTeX (caption below the figure, per D.01):\n")
    print(caption)


# ---------------------------------------------------------------------------
def out_name(args):
    if args.out:
        return args.out
    tag = args.dataset + ("" if args.attack == "all" else "_" + args.attack)
    return str(ROOT / f"sweep_tau_quantile_{tag}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", default=DEFAULT_CSV,
                    help="one or more sweep_summary_*.csv (default: the 66-run sweep)")
    ap.add_argument("--dataset", default="all",
                    help="mnist | fashion_mnist | cifar10 | all  (default: all)")
    ap.add_argument("--attack", default="all",
                    help="gaussian | lie | min_max | fang | no_attack | all  (default: all)")
    ap.add_argument("--out", default=None,
                    help="output file (default: derived from the selection). "
                         "Prefer .pdf -- IEEE wants vector figures with embedded fonts.")
    ap.add_argument("--clients", type=int, default=None,
                    help="federation size N, if it cannot be inferred from save_path")
    ap.add_argument("--rounds", type=int, default=None,
                    help="horizon T, if it cannot be inferred from save_path")
    ap.add_argument("--select", type=float, default=None,
                    help="mark this q_s as the chosen operating point")
    ap.add_argument("--zoom-accuracy", action="store_true",
                    help="auto-scale the accuracy axis instead of using [0,1]. "
                         "Magnifies single-run noise; off by default for that reason.")
    args = ap.parse_args()

    rows, avail_ds, avail_atk = load_rows(args.csv, args.dataset, args.attack)
    pts = aggregate(rows)
    if len(pts) < 2:
        sys.exit(f"error: only {len(pts)} q_s value(s) after filtering; nothing to sweep")

    N, T = infer_scale(rows)
    if args.clients: N = args.clients
    if args.rounds:  T = args.rounds
    exhaustive = N * T if (N and T) else None

    cells = len({(r["dataset"], r["attack_mode"]) for r in rows})
    scope = DATASET_LABEL.get(args.dataset, args.dataset)
    setup = f"{len(pts)} settings over {cells} configuration" + ("s" if cells != 1 else "")
    if exhaustive:
        setup += f", $N={N}$, $T={T}$"
    spread = max(p["acc"] for p in pts) - min(p["acc"] for p in pts)
    cut = 100 * (1 - min(p["fwd"] for p in pts) / exhaustive) if exhaustive else None
    # D.03: the caption states the takeaway, not just the axes.
    caption = (
        "\\begin{figure}[t]\n  \\centering\n"
        f"  \\includegraphics[width=\\columnwidth]{{{Path(out_name(args)).name}}}\n"
        f"  \\caption{{Detector cost and accuracy versus the scoring quantile "
        f"$q_s$ on {scope} ({setup}). Invocations fall "
        + (f"by {cut:.1f}\\% against exhaustive scoring" if cut else "monotonically")
        + f", while accuracy varies by only {spread:.4f} across the sweep, so the "
          "quantile trades compute without trading detection quality.}\n"
        "  \\label{fig:qs_sweep}\n\\end{figure}")

    make_plot(pts, exhaustive, caption, out_name(args), args.zoom_accuracy, args.select)

    for p in pts:
        print(f"  q_s={p['tau']:.1f}  acc={p['acc']:.4f}  fwd={p['fwd']:7.0f}  (n={p['n']})")


if __name__ == "__main__":
    main()
