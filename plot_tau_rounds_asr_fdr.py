"""plot_tau_rounds_asr_fdr.py  --  ASR/FDR over the (tau_s x round) plane.

The project already has two views of this sweep, and neither shows what this figure
shows:

  * ``plot_sweep_asr_fdr_<dataset>.py``  collapses each run to ONE scalar ASR/FDR and
    puts tau_s on the x-axis.  The round dimension is gone.
  * ``plot_rounds_asr_fdr.py``           keeps the round dimension but draws all 11
    tau_s runs as overlapping thin lines.  tau_s is spaghetti, not an axis.

Here BOTH tau_s and round are real axes, and the run-level scalar is pinned next to
the per-round field so you can see how one becomes the other.

THE POINT.  The run-level headline is "ASR ~ 0".  That number is a 50-round average of
a metric whose entire mass sits in ONE round:

                rounds 1-5      round 6              rounds 7-50
      ASR       exactly 0       mean .114 max .300   max .010
      FDR       mean  .410      mean .250            mean .112

Averaging the round-6 spike over 50 rounds dilutes it to a run-level ASR of ~0.002.
The left panel reports that ~0; the heatmaps show WHY it is misleading -- the spike is
a single vertical stripe at the attack onset.

Sharper still, once the plane is drawn: ASR is 0 in 4355 of the 4356 post-onset cells
(the lone exception is fashion_mnist/min_max at 0.010).  The ENTIRE ASR signal in this
sweep is round 6.  And it is attack-dependent -- the round-6 peak over tau_s is

      Gaussian   0.000  0.000  0.000     (mnist / fashion_mnist / cifar10)
      LIE        0.230  0.210  0.020
      Min-Max    0.290  0.300  0.250

so Gaussian attackers are caught 100% of the time on every dataset, and the detector's
whole exposure is the LIE/Min-Max onset round.  That is why each ASR panel is directly
labelled with its round-6 peak: on the shared colour scale the spike is a faint stripe,
and Gaussian's panel is *entirely blank* -- which without a label reads as "no data"
rather than "perfect detection".

  left panel  : x = run-level ASR / FDR,  y = tau_s.  3 attacks overlaid.
  heatmaps    : x = communication round,  y = tau_s,  colour = the per-round rate.
                rows = attacks, cols = {ASR, FDR}.
  shared y    : the swept tau_s (11 rows, 0.0 .. 1.0), across every panel.

LEFT IS EXACTLY THE ROW-MEAN OF RIGHT, not a separate statistic.  Every round has
exactly 100 clients, so

    run-level ASR = sum(FN) / sum(n) = (1/50) * sum_r (FN_r / 100) = mean_r(ASR_r)

``check_row_mean_identity()`` asserts this rather than asking you to trust the caption.

Reads outputs/2026-07-06 .. 2026-07-11 (every run dir since 2026-07-06; none exist
after 07-11) -- the current 5-channel no-valdelta detector, the same run set the sweep
trio reads.  Duplicate (dataset, attack, tau) runs are de-duplicated to the 50-round
run, with the same ranking as the other scripts, so all of them read identical runs.

PROVENANCE COMES FROM THE RUN-DIR NAME, NEVER FROM run_config.json.  That file is
written once per outer Flower session and reports attack-mode=gaussian,
tau-s-quantile=0.8 for EVERY run, because server_sweep.py sweeps all combos inside a
single ``flwr run``.  Reading it would silently mislabel every point.

Two different things in this codebase are called a "quantile".  This figure uses the
first and not the second:
  * tau_s_quantile  -- swept 0.0..1.0, in the run-dir name, STATIC per run.  The EWMA
                       threshold above which a client is re-scored by the ConvLSTM
                       (strategy.py:3400): a re-scoring *frequency* knob.  Constant
                       within a run, which is exactly why it can be an AXIS here.
  * filter_quantile -- per-round, adaptive, clipped to [0.35, 0.90] (strategy.py:2636).
                       Aggregation filtering strictness.  Not used here.

Figures (sweep_figures/):
    sweep_tau_rounds_<dataset>.{png,pdf}   3x[left + ASR|FDR heatmaps], one per dataset
    sweep_tau_rounds_marginal.csv          the left panel's data + the identity check

Usage (from inside rl-flwr/):
    python plot_tau_rounds_asr_fdr.py
"""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sweep_figures"

# "Every run since 2026-07-06" is a START DATE, not a fixed list.  The sibling scripts
# hardcode 07-06..07-11, which was complete when they were written -- but it silently
# drops any run that lands later.  Discover the dirs instead, so a rerun after new runs
# appear picks them up.  (ISO dates compare correctly as strings.)
SINCE = "2026-07-06"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATASETS = ["mnist", "fashion_mnist", "cifar10"]
ATTACKS = ["gaussian", "lie", "min_max"]

RUN_RE = re.compile(
    r"^(\d\d-\d\d-\d\d)_(mnist|fashion_mnist|cifar10)_"
    r"(gaussian|lie|min_max|no_attack)_tau([0-9.]+)$"
)

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}
ATTACK_LABELS = {"gaussian": "Gaussian", "lie": "LIE", "min_max": "Min-Max"}
# Okabe-Ito categorical hues, identical to the other plot scripts.  Validated with
# the dataviz palette checker: all six checks PASS, worst adjacent CVD dE 37.2
# (protan) / 17.0 (tritan), both above the >=12 target.
ATTACK_COLORS = {"gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2"}

# Sequential ramp for the heatmaps: SINGLE HUE, light -> dark, so more ink = more
# error and a near-white panel reads as "nothing happened".  A multi-hue/rainbow ramp
# for magnitude is an anti-pattern.  Purple collides with none of the three categorical
# attack hues above, which leaves neutral grey free to mean "masked".
CMAP_NAME = "Purples"
MASK_COLOR = "#d9d9d9"  # structurally-undefined cells; NOT a value on the ramp

# The malicious clients start submitting poisoned updates at this round.
ONSET_ROUND = 6
CLIENTS_PER_ROUND = 100
MALICIOUS_PER_ROUND = 40

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
        "lines.linewidth": 1.15,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


# ── Data extraction ───────────────────────────────────────────────────────────
def per_round_metrics(run_dir: Path) -> dict:
    """{round: {fn, fp, tp, tn, n, n_malicious, asr, fdr}} for one run."""
    agg = defaultdict(lambda: dict(fn=0, fp=0, tp=0, tn=0, n=0, n_malicious=0))

    cs = run_dir / "client_states_rounds.csv"
    if not cs.exists():
        return {}
    with open(cs) as f:
        for row in csv.DictReader(f):
            rnd = int(row["round"])
            is_mal = int(row["is_malicious"])
            pred = int(row["predict_state"])
            a = agg[rnd]
            a["n"] += 1
            a["n_malicious"] += is_mal
            if is_mal == 1:
                a["tp"] += pred == 1
                a["fn"] += pred == 0
            else:
                a["fp"] += pred == 1
                a["tn"] += pred == 0

    return {
        rnd: dict(a, asr=a["fn"] / a["n"], fdr=a["fp"] / a["n"])
        for rnd, a in sorted(agg.items())
        if a["n"] > 0
    }


def date_dirs() -> list:
    """Every outputs/<date> dir on or after SINCE."""
    root = ROOT / "outputs"
    dates = sorted(
        d for d in os.listdir(root)
        if DATE_RE.match(d) and d >= SINCE and (root / d).is_dir()
    )
    assert dates, f"no outputs/<date> dirs on or after {SINCE}"
    return dates


def collect_rounds() -> dict:
    """{(dataset, attack): {tau: {round: metrics}}}, deduped to the 50-round run."""
    raw = []
    for date in date_dirs():
        dd = ROOT / "outputs" / date
        for name in sorted(os.listdir(dd)):
            m = RUN_RE.match(name)
            if not m:
                continue
            _, dataset, attack, tau = m.groups()
            if attack == "no_attack":
                continue
            curve = per_round_metrics(dd / name)
            if not curve:
                continue  # empty / incomplete run
            raw.append(
                dict(date=date, dataset=dataset, attack=attack, tau=float(tau),
                     nrounds=len(curve), curve=curve)
            )

    # prefer the 50-round run, then most rounds, then latest date -- the same ranking
    # as plot_sweep_asr_fdr.collect() and plot_rounds_asr_fdr.collect_rounds().
    def rank(r):
        return (1 if r["nrounds"] == 50 else 0, r["nrounds"], r["date"])

    best = {}
    for r in raw:
        key = (r["dataset"], r["attack"], r["tau"])
        if key not in best or rank(r) > rank(best[key]):
            best[key] = r

    grouped = defaultdict(dict)
    for (dataset, attack, tau), r in best.items():
        grouped[(dataset, attack)][tau] = r["curve"]
    return grouped


# ── Integrity guards ──────────────────────────────────────────────────────────
def check_geometry(grouped: dict) -> None:
    """Every run must have the client/malicious geometry the figure assumes.

    The pre-attack ASR mask is drawn identically on every ASR panel, so it is only
    correct if every run really has zero malicious clients before ONSET_ROUND.
    Checking one run would leave the claim unverified for the rest.
    """
    for (dataset, attack), taus in sorted(grouped.items()):
        for tau, curve in sorted(taus.items()):
            where = f"{dataset}/{attack}/tau={tau:.1f}"
            for rnd, a in curve.items():
                assert a["n"] == CLIENTS_PER_ROUND, (
                    f"{where} round {rnd}: {a['n']} clients, expected {CLIENTS_PER_ROUND}"
                )
                expected = MALICIOUS_PER_ROUND if rnd >= ONSET_ROUND else 0
                assert a["n_malicious"] == expected, (
                    f"{where} round {rnd}: {a['n_malicious']} malicious, expected {expected}"
                )
    print(
        f"  geometry OK on all {sum(len(t) for t in grouped.values())} runs: "
        f"{CLIENTS_PER_ROUND} clients/round, {MALICIOUS_PER_ROUND} malicious from "
        f"round {ONSET_ROUND}"
    )


def check_grid_complete(grouped: dict, taus: list, rounds: list) -> None:
    """Fail loudly on a missing cell.

    A silently absent run would render as a 0-valued cell -- i.e. it would look like a
    PERFECT result rather than a missing one.  That is the worst possible failure mode
    for this figure, so it is an error, not a warning.
    """
    for dataset in DATASETS:
        for attack in ATTACKS:
            cells = grouped.get((dataset, attack))
            assert cells is not None, f"missing entirely: {dataset}/{attack}"
            missing = [
                (t, r) for t in taus for r in rounds if r not in cells.get(t, {})
            ]
            assert not missing, (
                f"{dataset}/{attack}: {len(missing)} missing (tau, round) cells, "
                f"first few {missing[:5]}"
            )
    print(
        f"  grid complete: {len(DATASETS)}x{len(ATTACKS)} combos x {len(taus)} tau "
        f"x {len(rounds)} rounds = "
        f"{len(DATASETS) * len(ATTACKS) * len(taus) * len(rounds)} cells, no gaps"
    )


def check_row_mean_identity(grouped: dict, taus: list, rounds: list) -> None:
    """The figure CLAIMS left panel == row-mean of the heatmap.  Assert it.

    run-level ASR = sum(FN)/sum(n) over the run.  With n == 100 every round that is
    identically mean_r(FN_r/100) = mean_r(ASR_r) = the heatmap row mean.  The claim is
    load-bearing (it is what makes the two halves the same data), so it is checked
    rather than asserted in prose.
    """
    worst = 0.0
    for dataset in DATASETS:
        for attack in ATTACKS:
            cells = grouped[(dataset, attack)]
            for tau in taus:
                curve = cells[tau]
                for metric, num in (("asr", "fn"), ("fdr", "fp")):
                    run_level = (
                        sum(curve[r][num] for r in rounds)
                        / sum(curve[r]["n"] for r in rounds)
                    )
                    row_mean = float(np.mean([curve[r][metric] for r in rounds]))
                    worst = max(worst, abs(run_level - row_mean))
    assert worst < 1e-9, f"left panel is NOT the row-mean: max deviation {worst:.2e}"
    print(f"  row-mean identity holds: max |run-level - row-mean| = {worst:.2e}")


# ── Grid assembly ─────────────────────────────────────────────────────────────
def heat_grid(cells: dict, metric: str, taus: list, rounds: list) -> np.ma.MaskedArray:
    """[tau x round] array of the per-round rate, with structural zeros masked.

    ASR is *structurally* 0 before ONSET_ROUND: there are no malicious clients yet, so
    there is nothing to miss.  Painting those cells with the ramp's 0-colour would show
    a clean band exactly where the detector was never tested -- it would read as the
    detector succeeding.  They are masked instead.

    FDR is NOT masked there: a false positive is real and meaningful with no attacker
    present.  Indeed that is where FDR peaks (~0.41) -- the cold-start over-flagging.
    """
    M = np.array([[cells[t][r][metric] for r in rounds] for t in taus], dtype=float)
    if metric == "asr":
        pre = np.array(rounds) < ONSET_ROUND
        M = np.ma.masked_array(M, mask=np.broadcast_to(pre, M.shape).copy())
    else:
        M = np.ma.masked_array(M, mask=np.zeros_like(M, dtype=bool))
    return M


def run_level(cells: dict, metric: str, taus: list, rounds: list) -> list:
    """Run-level rate per tau = sum(errors)/sum(n) = the heatmap row mean."""
    num = {"asr": "fn", "fdr": "fp"}[metric]
    return [
        sum(cells[t][r][num] for r in rounds) / sum(cells[t][r]["n"] for r in rounds)
        for t in taus
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_dataset(grouped, dataset, taus, rounds, vmax):
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad(MASK_COLOR)

    # y extent: 11 rows centred on tau = 0.0 .. 1.0, so each row is 0.1 tall.
    dt = (taus[1] - taus[0]) / 2.0
    y0, y1 = taus[0] - dt, taus[-1] + dt
    x0, x1 = rounds[0] - 0.5, rounds[-1] + 0.5
    extent = (x0, x1, y0, y1)

    fig = plt.figure(figsize=(IEEE_WIDTH_IN, 4.1), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 2.05, 2.05], hspace=0.16, wspace=0.07)

    # ---- left: run-level rates vs tau_s, all 3 attacks (the row-mean of the right) ----
    ax_l = fig.add_subplot(gs[:, 0])
    for attack in ATTACKS:
        cells = grouped[(dataset, attack)]
        color = ATTACK_COLORS[attack]
        ax_l.plot(run_level(cells, "fdr", taus, rounds), taus, color=color,
                  linestyle="--", linewidth=1.2, marker="s", markersize=3.0,
                  alpha=0.95, zorder=3)
        ax_l.plot(run_level(cells, "asr", taus, rounds), taus, color=color,
                  linestyle="-", linewidth=1.6, marker="o", markersize=3.6, zorder=5)

    ax_l.set_ylim(y0, y1)
    ax_l.set_yticks(taus)
    ax_l.set_xlim(left=0.0)
    ax_l.set_ylabel(r"$q_s$  (swept $\tau_s$ quantile)")
    ax_l.set_xlabel("run-level rate")
    ax_l.set_title("Run-level\n(= row-mean $\\rightarrow$)", fontsize=IEEE_FONT)
    # solid hairline grid -- dashed grid reads as "threshold" when it is just a grid.
    ax_l.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.25, zorder=0)
    ax_l.set_axisbelow(True)
    # ASR collapses to ~0 once averaged over 50 rounds; say so, since it hugs the axis.
    ax_l.annotate("ASR $\\approx$ 0", xy=(0.02, 0.5), xycoords=("axes fraction", "axes fraction"),
                  fontsize=IEEE_FONT - 2, color="#555555", rotation=90, va="center")

    # ---- right: 3 attacks x {ASR, FDR} heatmaps over (round, tau_s) ----
    heat_axes = []
    im = None
    for i, attack in enumerate(ATTACKS):
        cells = grouped[(dataset, attack)]
        for j, metric in enumerate(("asr", "fdr")):
            ax = fig.add_subplot(gs[i, 1 + j])
            heat_axes.append(ax)
            im = ax.imshow(
                heat_grid(cells, metric, taus, rounds),
                aspect="auto", origin="lower", extent=extent,
                cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest",
            )
            # Hatch the masked band: a second, non-colour encoding of "undefined", so it
            # survives greyscale printing and CVD.
            if metric == "asr":
                ax.add_patch(Rectangle(
                    (x0, y0), ONSET_ROUND - 1, y1 - y0, facecolor="none",
                    edgecolor="#ffffff", hatch="///", linewidth=0.0, zorder=2,
                ))
            # attack onset, on the boundary between round 5 and 6 (solid hairline).
            ax.axvline(ONSET_ROUND - 0.5, color="#222222", linewidth=0.7, zorder=3)

            # Direct-label the round-6 peak on the ASR panels.  Two reasons, both real:
            # (a) on the shared [0, vmax] scale the spike is a legitimately faint stripe,
            #     so the reader cannot read its height off the colour; and
            # (b) Gaussian's panel is *entirely* empty (peak 0.000 -- every attacker
            #     caught, every round).  An unlabelled blank panel reads as "no data" or
            #     "broken", which is the opposite of what it means.  The label is what
            #     separates "perfect" from "missing".
            if metric == "asr":
                peak = max(cells[t][ONSET_ROUND]["asr"] for t in taus)
                note = (
                    "round-6 peak: 0 (all attackers caught)"
                    if peak == 0.0
                    else f"round-6 peak: {peak:.2f}"
                )
                ax.annotate(
                    note, xy=(ONSET_ROUND + 1.5, y1 - 0.04), fontsize=IEEE_FONT - 2,
                    color="#444444", ha="left", va="top", zorder=4,
                )

            ax.set_ylim(y0, y1)
            ax.set_yticks(taus)
            ax.set_yticklabels([])
            ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
            if i == 0:
                ax.set_title("ASR by round" if metric == "asr" else "FDR by round")
            if i == len(ATTACKS) - 1:
                ax.set_xlabel("Communication round")
            else:
                ax.set_xticklabels([])
            # The row label rides on the ASR column's y-axis (tick labels are off).
            if j == 0:
                ax.set_ylabel(ATTACK_LABELS[attack])

    cb = fig.colorbar(im, ax=heat_axes, location="right", fraction=0.030, pad=0.012)
    cb.set_label("per-round rate  (errors / 100 clients)")
    cb.outline.set_linewidth(0.4)

    attack_h = [
        Line2D([0], [0], color=ATTACK_COLORS[a], linewidth=1.6, label=ATTACK_LABELS[a])
        for a in ATTACKS
    ]
    metric_h = [
        Line2D([0], [0], color="0.25", linestyle="-", marker="o", markersize=3.6,
               linewidth=1.6, label="ASR (missed attackers)"),
        Line2D([0], [0], color="0.25", linestyle="--", marker="s", markersize=3.0,
               linewidth=1.2, label="FDR (flagged benign)"),
        Patch(facecolor=MASK_COLOR, edgecolor="#ffffff", hatch="///",
              label=f"no attackers yet (round < {ONSET_ROUND}): ASR undefined"),
    ]
    fig.legend(handles=attack_h + metric_h, loc="outside lower center", ncol=3,
               frameon=False, columnspacing=1.8, handlelength=1.8, handletextpad=0.5)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: ASR and FDR over the "
        r"($\tau_s$ quantile $\times$ round) plane",
        fontsize=IEEE_FONT,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"sweep_tau_rounds_{dataset}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


def export_marginal(grouped, taus, rounds):
    """The left panel's data, plus the row-mean identity, so it is auditable offline.

    The per-cell grid is deliberately NOT re-exported: plot_rounds_asr_fdr.py already
    writes those cells to rounds_asr_fdr_per_round.csv, and a second copy would be a
    divergence risk.
    """
    out = OUT_DIR / "sweep_tau_rounds_marginal.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "attack", "tau", "asr_run", "fdr_run",
                    "asr_rowmean", "fdr_rowmean"])
        for dataset in DATASETS:
            for attack in ATTACKS:
                cells = grouped[(dataset, attack)]
                asr_run = run_level(cells, "asr", taus, rounds)
                fdr_run = run_level(cells, "fdr", taus, rounds)
                for k, tau in enumerate(taus):
                    curve = cells[tau]
                    rm = {
                        m: float(np.mean([curve[r][m] for r in rounds]))
                        for m in ("asr", "fdr")
                    }
                    w.writerow([dataset, attack, f"{tau:.1f}",
                                f"{asr_run[k]:.6f}", f"{fdr_run[k]:.6f}",
                                f"{rm['asr']:.6f}", f"{rm['fdr']:.6f}"])
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    grouped = collect_rounds()
    taus = sorted({t for cells in grouped.values() for t in cells})
    rounds = sorted({r for cells in grouped.values() for c in cells.values() for r in c})
    print(
        f"Loaded {sum(len(c) for c in grouped.values())} runs across "
        f"{len(grouped)} (dataset, attack) combos x {len(taus)} tau values."
    )

    check_geometry(grouped)
    check_grid_complete(grouped, taus, rounds)
    check_row_mean_identity(grouped, taus, rounds)

    # ONE shared colour scale across both metrics, all attacks and all three datasets, so
    # the figures are comparable to each other AND ASR is comparable to FDR.  Per-panel
    # autoscaling would stretch ASR's post-attack noise (max 0.010) to full saturation and
    # render detector noise as if it were signal.
    vmax = max(
        c[r][m]
        for cells in grouped.values()
        for c in cells.values()
        for r in rounds
        for m in ("asr", "fdr")
    )
    print(f"  shared colour scale: [0, {vmax:.3f}] over both metrics and all datasets")

    export_marginal(grouped, taus, rounds)
    for dataset in DATASETS:
        plot_dataset(grouped, dataset, taus, rounds, vmax)
    print(f"\nAll figures saved under: {OUT_DIR.resolve()}")
