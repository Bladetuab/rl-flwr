"""plot_rounds_asr_fdr.py

Per-ROUND ASR / FDR dynamics for every dataset and attack, over the tau_s sweep
logged in outputs/2026-07-06 .. 2026-07-11 (the current 5-channel no-valdelta
detector -- the same run set the sweep trio reads).

This is the *time-resolved* companion to the sweep figures: where
``plot_sweep_asr_fdr*.py`` collapse each run to ONE scalar ASR/FDR and put tau_s
on the x-axis, this script keeps the round dimension and puts the training round
on the x-axis.  It is deliberately standalone (its own dedup + metric code) so it
cannot perturb the shared ``plot_sweep_asr_fdr.collect()`` that the sweep trio
depends on -- that helper must keep returning one scalar per run.

  x-axis : training round (1..50)
  y-axis : detection error rate, same definitions as the rest of the project
             ASR = FN / N = #(is_malicious==1 & predict_state==0) / N
             FDR = FP / N = #(is_malicious==0 & predict_state==1) / N
           but evaluated PER ROUND, so N = the client-rows in that round
           (100 clients/round), not the whole run.
  lines  : one thin line per tau_s in {0.0, 0.1, ..., 1.0} (11 per attack),
           colour = attack; a bold line marks the median over tau_s.

READ THE ONSET MARKER.  The attack starts at round 6: rounds 1-5 contain
0 malicious clients (of 100), rounds 6-50 contain 40.  Therefore:
  * ASR is *structurally* 0 in rounds 1-5 -- there is nothing to miss.  It is NOT
    evidence the detector caught anything.
  * The high warm-up FDR (~0.33-0.47) is a no-attack-signal regime, not a
    detector failure: with no attacker present the detector has no contrast to
    calibrate against and over-flags benign clients.
Both facts are drawn on the figure (shaded pre-attack span + round-6 rule) so the
curves cannot be misread.

Because ASR is normalised by ALL client rows (N=100), not by the malicious ones,
its ceiling is 0.40 (40 malicious / 100 clients).  An ASR of 0.11 at round 6 is
0.11*100 = 11 missed malicious clients = 11/40 = 27.5% of the attackers.

Duplicate (dataset, attack, tau) runs across dates are de-duplicated to the
50-round run, matching ``plot_sweep_asr_fdr.collect()`` (an 11-run 100-round
CIFAR-Gaussian sweep on 2026-07-06 is superseded by its 50-round rerun).

Figures (sweep_figures/):
    rounds_asr_fdr_combined.{png,pdf}      2x3: rows = {ASR, FDR}, cols = datasets
    rounds_asr_fdr_<dataset>.{png,pdf}     1x2 per dataset: ASR | FDR
    rounds_asr_fdr_per_round.csv           the underlying per-round counts

Usage (from inside rl-flwr/):
    python plot_rounds_asr_fdr.py
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
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sweep_figures"
DATES = [
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-11",
]
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
# Okabe-Ito colour-blind-safe categorical hues, identical to plot_sweep_asr_fdr.py
# (validated: all checks pass, worst adjacent CVD dE 17.0 tritan).
ATTACK_COLORS = {"gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2"}

# Round at which the malicious clients start submitting poisoned updates.
ONSET_ROUND = 6
# 40 of the 100 clients turn malicious at ONSET_ROUND, and ASR is normalised by ALL
# 100 client rows -- so ASR saturates at 0.40, which means every attacker evaded
# detection that round.  The saturation line is drawn on EVERY ASR panel, so this
# geometry is asserted for every run (not just one) by check_geometry().
CLIENTS_PER_ROUND = 100
MALICIOUS_PER_ROUND = 40
ASR_CEILING = MALICIOUS_PER_ROUND / CLIENTS_PER_ROUND

# The 11 tau_s runs are all drawn as thin lines (the raw view).  This additionally
# overlays a bold median-over-tau_s line per attack, purely to make 33 overlapping
# lines followable; set False for plain spaghetti with no summary curve.
SHOW_MEDIAN = True

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
    """{round: {asr, fdr, fn, fp, tp, tn, n, n_malicious}} for one run.

    ASR/FDR use the project-wide normalisation (denominator = ALL client rows in
    the round), just restricted to a single round instead of the whole run.
    """
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


def collect_rounds() -> dict:
    """{(dataset, attack): {tau: {round: metrics}}}, deduped to the 50-round run."""
    raw = []
    for date in DATES:
        dd = ROOT / "outputs" / date
        if not dd.exists():
            continue
        for name in sorted(os.listdir(dd)):
            m = RUN_RE.match(name)
            if not m:
                continue
            _, dataset, attack, tau = m.groups()
            if attack == "no_attack":
                continue
            curve = per_round_metrics(dd / name)
            if not curve:
                continue  # skip empty / incomplete runs
            raw.append(
                dict(
                    date=date,
                    dataset=dataset,
                    attack=attack,
                    tau=float(tau),
                    nrounds=len(curve),
                    curve=curve,
                )
            )

    # de-dup: prefer the 50-round run, then most rounds, then latest date
    # (same ranking as plot_sweep_asr_fdr.collect(), so both read the same runs).
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


def check_geometry(grouped: dict) -> None:
    """Assert EVERY run has the client/malicious geometry the figures assume.

    The ASR saturation line (ASR_CEILING) and its "all 40 attackers evade" label are
    drawn identically on every ASR panel, so they are only correct if every run really
    has CLIENTS_PER_ROUND clients with MALICIOUS_PER_ROUND of them turning malicious at
    ONSET_ROUND.  Checking a single run would leave that claim unverified for the rest.
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
        f"  geometry verified on all {sum(len(t) for t in grouped.values())} runs: "
        f"{CLIENTS_PER_ROUND} clients/round, {MALICIOUS_PER_ROUND} malicious from "
        f"round {ONSET_ROUND} => ASR ceiling {ASR_CEILING:.2f}"
    )


def median_curve(taus: dict, metric: str, rounds: list) -> list:
    """Median over the tau lines, per round."""
    return [
        float(np.median([taus[t][r][metric] for t in taus if r in taus[t]]))
        for r in rounds
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────
def _draw_panel(ax, grouped, dataset, metric, ymax):
    """One panel: all attacks x all tau, for one dataset and one metric."""
    # Pre-attack regime: no malicious clients exist at all (see module docstring).
    ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0, zorder=0)
    ax.axvline(ONSET_ROUND, color="0.35", linestyle="--", linewidth=0.8, zorder=1)

    # ASR is normalised by ALL client rows, so its ceiling is the malicious fraction:
    # ASR = 0.40 means every one of the 40 attackers evaded detection that round.
    if metric == "asr":
        ax.axhline(ASR_CEILING, color="0.45", linestyle=(0, (1, 1.6)), linewidth=0.7,
                   zorder=1)

    for attack in ATTACKS:
        taus = grouped.get((dataset, attack))
        if not taus:
            continue
        color = ATTACK_COLORS[attack]
        rounds = sorted({r for c in taus.values() for r in c})

        # One thin line per tau_s (11 of them).
        for tau in sorted(taus):
            curve = taus[tau]
            rr = sorted(curve)
            ax.plot(
                rr,
                [curve[r][metric] for r in rr],
                color=color,
                linewidth=0.55,
                alpha=0.28,
                zorder=2,
                solid_capstyle="round",
            )

        # Bold median over tau_s, so the central trend is followable.
        if SHOW_MEDIAN:
            ax.plot(
                rounds,
                median_curve(taus, metric, rounds),
                color=color,
                linewidth=1.5,
                alpha=1.0,
                zorder=4,
                solid_capstyle="round",
            )

    ax.set_xlim(0.5, 50.5)
    ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.02 * ymax, ymax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)


def _ymax(grouped, metric):
    vals = [
        c[r][metric]
        for taus in grouped.values()
        for c in taus.values()
        for r in c
    ]
    top = max(vals) * 1.12 if max(vals) > 0 else 1.0
    if metric == "asr":
        # Keep the saturation line (= total attack success) on-axis for reference.
        top = max(top, ASR_CEILING * 1.10)
    return top


def _legend_handles():
    attack_h = [
        Line2D([0], [0], color=ATTACK_COLORS[a], linewidth=1.5, label=ATTACK_LABELS[a])
        for a in ATTACKS
    ]
    semantic_h = [
        Line2D([0], [0], color="0.35", linewidth=0.55, alpha=0.5,
               label=r"one run ($\tau_s$ = 0.0 … 1.0)"),
        Line2D([0], [0], color="0.35", linewidth=1.5,
               label=r"median over $\tau_s$"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=0.8,
               label=f"attack onset (round {ONSET_ROUND})"),
        Patch(facecolor="0.5", alpha=0.10, label="no malicious clients present"),
    ]
    return attack_h, semantic_h


def plot_combined(grouped):
    """2x3 grid: rows = {ASR, FDR}, cols = datasets.  Each row shares a y-scale."""
    metrics = [
        ("asr", "ASR  =  FN / N"),
        ("fdr", "FDR  =  FP / N"),
    ]
    ymaxes = {m: _ymax(grouped, m) for m, _ in metrics}

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_WIDTH_IN, 4.5), sharex=True)
    for i, (metric, ylabel) in enumerate(metrics):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i][j]
            _draw_panel(ax, grouped, dataset, metric, ymaxes[metric])
            if i == 0:
                ax.set_title(DATASET_LABELS[dataset])
            if i == len(metrics) - 1:
                ax.set_xlabel("Training round")
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])

    # Spell out why ASR is pinned at 0 before the onset, on the panel it applies to.
    axes[0][0].annotate(
        "no attack yet",
        xy=(3.0, ymaxes["asr"] * 0.5),
        rotation=90,
        ha="center",
        va="center",
        fontsize=IEEE_FONT - 2,
        color="#555555",
    )
    # Direct-label the ASR saturation line rather than spending a legend slot on it.
    axes[0][0].annotate(
        "all 40 attackers evade",
        xy=(9.0, ASR_CEILING - 0.012),
        ha="left",
        va="top",
        fontsize=IEEE_FONT - 2,
        color="#555555",
    )

    attack_h, semantic_h = _legend_handles()
    fig.legend(
        handles=attack_h + semantic_h,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _save(fig, "rounds_asr_fdr_combined")


def plot_per_dataset(grouped):
    """One figure per dataset: ASR | FDR side by side, all attacks x all tau."""
    metrics = [("asr", "ASR  =  FN / N"), ("fdr", "FDR  =  FP / N")]
    ymaxes = {m: _ymax(grouped, m) for m, _ in metrics}
    attack_h, semantic_h = _legend_handles()

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDTH_IN, 2.9))
        for ax, (metric, ylabel) in zip(axes, metrics):
            _draw_panel(ax, grouped, dataset, metric, ymaxes[metric])
            ax.set_xlabel("Training round")
            ax.set_ylabel(ylabel)
        axes[0].set_title("Attack success rate")
        axes[1].set_title("False detection rate")
        axes[0].annotate(
            "all 40 attackers evade",
            xy=(9.0, ASR_CEILING - 0.012),
            ha="left",
            va="top",
            fontsize=IEEE_FONT - 2,
            color="#555555",
        )
        fig.suptitle(
            f"{DATASET_LABELS[dataset]}: per-round detection error over the "
            r"$\tau_s$ sweep",
            fontsize=IEEE_FONT,
        )

        fig.legend(
            handles=attack_h + semantic_h,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            columnspacing=1.4,
            handlelength=1.8,
            handletextpad=0.5,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 0.95))
        _save(fig, f"rounds_asr_fdr_{dataset}")


def export_csv(grouped):
    out = OUT_DIR / "rounds_asr_fdr_per_round.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "attack", "tau", "round", "n", "n_malicious",
                    "fn", "fp", "tp", "tn", "asr", "fdr"])
        for dataset in DATASETS:
            for attack in ATTACKS:
                taus = grouped.get((dataset, attack), {})
                for tau in sorted(taus):
                    curve = taus[tau]
                    for rnd in sorted(curve):
                        a = curve[rnd]
                        w.writerow([dataset, attack, f"{tau:.1f}", rnd, a["n"],
                                    a["n_malicious"], a["fn"], a["fp"], a["tp"],
                                    a["tn"], f"{a['asr']:.5f}", f"{a['fdr']:.5f}"])
    print(f"  Saved -> {out}")


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    grouped = collect_rounds()
    nruns = sum(len(t) for t in grouped.values())
    print(
        f"Loaded {nruns} runs across {len(grouped)} (dataset, attack) combos "
        f"x {len({t for taus in grouped.values() for t in taus})} tau values."
    )

    # The onset/ceiling annotations are drawn on every panel -- verify every run.
    check_geometry(grouped)
    # Read the data directly -- do NOT back it out of _ymax(), which clamps the ASR
    # axis up to the saturation line and would over-report the observed maximum.
    for metric in ("asr", "fdr"):
        vals = [c[r][metric] for taus in grouped.values() for c in taus.values() for r in c]
        peak_round = max(
            ((r, c[r][metric]) for taus in grouped.values() for c in taus.values() for r in c),
            key=lambda x: x[1],
        )[0]
        print(f"  {metric.upper()} range over all rounds/runs: "
              f"[{min(vals):.3f}, {max(vals):.3f}]  (peak at round {peak_round})")

    export_csv(grouped)
    plot_combined(grouped)
    plot_per_dataset(grouped)
    print(f"\nAll figures saved under: {OUT_DIR.resolve()}")
