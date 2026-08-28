"""plot_rounds_by_tau_asr_fdr.py  --  ASR/FDR VALUE vs round, one line per swept tau_s.

The point of this figure, versus the two that already exist:

  plot_rounds_asr_fdr.py    draws the same 11 tau_s lines per attack but colours them by
                            ATTACK -- so all 11 are the same hue and tau_s is invisible.
                            It is spaghetti in which the swept parameter cannot be seen.
  plot_tau_rounds_asr_fdr.py  puts tau_s on an axis and the rate in COLOUR.  Colour is a
                            weak encoder of value: ASR's round-6 spike (0.29 on a [0,0.48]
                            scale) washes out to near-white and the panel reads as blank.

Here the RATE IS ON A REAL Y-AXIS -- so you can read it -- with round on x and tau_s as an
ordered colour ramp.  Putting the round on x is what forces tau_s onto colour: both cannot
have an axis.

  x-axis : communication round (1..50)
  y-axis : the ASR / FDR VALUE, autoscaled PER METRIC (this is the fix -- sharing one
           scale across both metrics is exactly what made ASR invisible before)
  colour : the swept tau_s, light->dark over 0.0 .. 1.0, with a discrete colorbar
  facets : rows = {ASR, FDR}, cols = attacks.  One figure per dataset.

HOW MUCH TO TRUST THE COLOUR.  Eleven individually-identifiable colour steps DO NOT FIT in
one hue's lightness band -- checked, not guessed: the dataviz ordinal validator fails 11
steps of Blues on adjacent-dL (~0.05 per step, too close to tell apart).  So colour here
encodes tau_s's ORDER and SPREAD, not its identity: you can see that the lines stack
monotonically (or don't), and the colorbar plus the labelled tau_s = 0.0 / 1.0 extremes
anchor the direction.  You CANNOT read an individual tau_s curve off the hue.  If you need
that, this has to become small multiples (one panel per tau_s).
The ramp is Blues[0.50..1.0]; the light end is clipped there so the tau_s = 0.0 line clears
the 2:1 contrast floor against the surface (at the default light end it is 1.45:1 -- i.e.
invisible).

WHAT THE FIGURE SHOWS THAT THE OTHERS COULD NOT: for Gaussian, FDR is cleanly MONOTONE in
tau_s -- rank correlation +0.89 / +0.80 / +0.90 on MNIST / Fashion-MNIST / CIFAR-10, with a
tau_s range about 2x the round-to-round noise.  Three independent datasets, same sign, same
magnitude.  The lines simply stack in tau order.  LIE and Min-Max show no consistent
ordering (rank corr -0.25..+0.54).  CAVEAT: there are ZERO true replicates in this sweep
(no (dataset, attack, tau) has two 50-round runs) and the runs are unseeded, so for LIE and
Min-Max a weak tau_s effect cannot be separated from run-to-run variance.  Non-monotone is
NOT the same as "noise" -- it is "not resolvable with n=1 per cell".

READ THE ONSET.  40 of the 100 clients turn malicious at round 6; rounds 1-5 have none.  So
ASR is *structurally* 0 before round 6 (nothing to miss), not a detector success -- the
pre-attack span is shaded on every panel.  And because ASR = FN / N is normalised by ALL
100 clients, it SATURATES AT 0.40: an ASR of 0.30 is 30 missed attackers = 75% of the 40
evading.  The 0.40 ceiling is drawn on every ASR panel so the spike can be read that way,
and each ASR panel is labelled with its round-6 peak as a % of attackers.

Reads every outputs/<date> dir on or after SINCE (2026-07-06) -- a start date, not a fixed
list, so runs added later are picked up.  Provenance is the run-DIR NAME; run_config.json
reports attack-mode=gaussian, tau-s-quantile=0.8 for EVERY run (the sweep runs all combos
inside one `flwr run`) and would silently mislabel every point.

Figures (sweep_figures/):
    rounds_by_tau_<dataset>.{png,pdf}    2x3: rows = {ASR, FDR}, cols = attacks

Usage (from inside rl-flwr/):
    python plot_rounds_by_tau_asr_fdr.py
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sweep_figures"

# A START DATE, not a fixed list -- runs that land after today are picked up on a rerun.
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
# Attacks are FACETS here, not colours -- the hue channel is spent on tau_s.  The Okabe-Ito
# attack hues are kept only to tint the column titles, so this figure still speaks the same
# visual language as the rest of the project.
ATTACK_COLORS = {"gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2"}

# tau_s ramp: single hue, light -> dark (a multi-hue/rainbow ramp for an ordered quantity is
# an anti-pattern).  Clipped at 0.50 so the lightest step clears the 2:1 contrast floor
# against the surface -- unclipped, tau_s = 0.0 sits at 1.45:1 and is invisible on white.
TAU_CMAP = "Blues"
TAU_CMAP_LO = 0.50

ONSET_ROUND = 6
CLIENTS_PER_ROUND = 100
MALICIOUS_PER_ROUND = 40
# ASR = FN / N with N = ALL 100 clients, so it saturates at the malicious fraction.
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
        "lines.linewidth": 1.15,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


# ── Data extraction (standalone, per project convention) ──────────────────────
def per_round_metrics(run_dir: Path) -> dict:
    """{round: {fn, fp, n, n_malicious, asr, fdr}} for one run."""
    agg = defaultdict(lambda: dict(fn=0, fp=0, n=0, n_malicious=0))
    cs = run_dir / "client_states_rounds.csv"
    if not cs.exists():
        return {}
    with open(cs) as f:
        for row in csv.DictReader(f):
            a = agg[int(row["round"])]
            is_mal = int(row["is_malicious"])
            pred = int(row["predict_state"])
            a["n"] += 1
            a["n_malicious"] += is_mal
            if is_mal == 1:
                a["fn"] += pred == 0
            else:
                a["fp"] += pred == 1
    return {
        rnd: dict(a, asr=a["fn"] / a["n"], fdr=a["fp"] / a["n"])
        for rnd, a in sorted(agg.items())
        if a["n"] > 0
    }


def date_dirs() -> list:
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
                continue
            raw.append(dict(date=date, dataset=dataset, attack=attack, tau=float(tau),
                            nrounds=len(curve), curve=curve))

    # prefer the 50-round run, then most rounds, then latest date -- identical ranking to
    # the sibling scripts, so every figure in the project reads the same 99 runs.
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
    """The onset shading and the 0.40 ceiling are drawn on EVERY panel -- so verify the
    geometry they assert on every run, not just one."""
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
        f"{CLIENTS_PER_ROUND} clients/round, {MALICIOUS_PER_ROUND} malicious from round "
        f"{ONSET_ROUND} => ASR ceiling {ASR_CEILING:.2f}"
    )


# ── Plotting ──────────────────────────────────────────────────────────────────
def tau_colors(taus: list) -> list:
    base = plt.get_cmap(TAU_CMAP)
    return [base(x) for x in np.linspace(TAU_CMAP_LO, 1.0, len(taus))]


def plot_dataset(grouped, dataset, taus, rounds, ylims):
    colors = tau_colors(taus)
    # Keep the labels SHORT: at 8 pt a label like "ASR (missed attackers / 100)" is taller
    # than the panel and collides with the row below it.  These match the wording already
    # used by plot_rounds_asr_fdr.py, and the metric definitions live in the suptitle.
    metrics = [("asr", "ASR  =  FN / N"),
               ("fdr", "FDR  =  FP / N")]

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_WIDTH_IN, 3.9), sharex=True,
                             layout="constrained")

    for i, (metric, ylabel) in enumerate(metrics):
        for j, attack in enumerate(ATTACKS):
            ax = axes[i][j]
            cells = grouped[(dataset, attack)]

            # Pre-attack span: zero malicious clients exist, so ASR is structurally 0 here.
            ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0,
                       zorder=0)
            ax.axvline(ONSET_ROUND, color="0.35", linewidth=0.7, zorder=1)

            if metric == "asr":
                # ASR saturates at the malicious fraction; drawing it lets the spike be read
                # as "x% of attackers evaded" rather than as a bare rate.
                ax.axhline(ASR_CEILING, color="0.45", linewidth=0.7, zorder=1)

            for tau, color in zip(taus, colors):
                curve = cells[tau]
                ax.plot(rounds, [curve[r][metric] for r in rounds], color=color,
                        linewidth=0.9, alpha=0.95, zorder=3, solid_capstyle="round")

            ax.set_xlim(0.5, 50.5)
            ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
            ax.set_ylim(*ylims[metric])
            # solid hairline grid; dashed grid reads as "threshold" when it is just a grid.
            ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25, zorder=0)
            ax.set_axisbelow(True)

            if i == 0:
                ax.set_title(ATTACK_LABELS[attack], color=ATTACK_COLORS[attack])
            if i == len(metrics) - 1:
                ax.set_xlabel("Communication round")
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])

            # ASR is a delta function at the onset.  Give each panel its peak, translated
            # into the only unit anyone actually cares about: what fraction of the 40
            # attackers got through.  Gaussian's panel is flat zero, which without a label
            # reads as "no data" rather than "perfect detection".
            if metric == "asr":
                peak = max(cells[t][ONSET_ROUND]["asr"] for t in taus)
                note = (
                    "all attackers caught"
                    if peak == 0.0
                    else f"round-6 peak {peak:.2f}\n= {peak / ASR_CEILING * 100:.0f}% of attackers evade"
                )
                ax.annotate(note, xy=(0.97, 0.94), xycoords="axes fraction",
                            fontsize=IEEE_FONT - 2, color="#444444", ha="right", va="top")

    # Anchor the ramp's direction: the reader cannot identify an individual tau by hue
    # (11 steps do not fit in one hue's lightness band), but they can see which end is which.
    axes[1][0].annotate(r"$\tau_s\!=\!1.0$", xy=(0.30, 0.93), xycoords="axes fraction",
                        fontsize=IEEE_FONT - 2, color=colors[-1], ha="left", va="top")
    axes[1][0].annotate(r"$\tau_s\!=\!0.0$", xy=(0.30, 0.78), xycoords="axes fraction",
                        fontsize=IEEE_FONT - 2, color=colors[0], ha="left", va="top")

    # Discrete colorbar over the 11 swept values -- it carries tau_s's ORDER, which is what
    # the hue channel can honestly deliver.
    cmap = ListedColormap(colors)
    step = taus[1] - taus[0]
    norm = BoundaryNorm(np.array(taus + [taus[-1] + step]) - step / 2, ncolors=len(taus))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, location="right",
                      ticks=taus, fraction=0.028, pad=0.012)
    cb.set_label(r"swept re-scoring quantile  $\tau_s$")
    cb.ax.tick_params(labelsize=IEEE_FONT - 2)
    cb.outline.set_linewidth(0.4)

    fig.legend(
        handles=[
            Patch(facecolor="0.5", alpha=0.10, label="no malicious clients yet (rounds 1-5)"),
            Line2D([0], [0], color="0.35", linewidth=0.7,
                   label=f"attack onset (round {ONSET_ROUND})"),
            Line2D([0], [0], color="0.45", linewidth=0.7,
                   label=f"ASR ceiling {ASR_CEILING:.2f} = all {MALICIOUS_PER_ROUND} evade"),
        ],
        loc="outside lower center", ncol=3, frameon=False,
        columnspacing=2.0, handlelength=1.8, handletextpad=0.5,
    )
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: per-round ASR and FDR across the swept "
        r"$\tau_s$ quantile"
        "\n"
        r"$N=100$ clients/round;  ASR $=$ missed attackers$/N$,  "
        r"FDR $=$ wrongly flagged benign$/N$",
        fontsize=IEEE_FONT,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"rounds_by_tau_{dataset}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    grouped = collect_rounds()
    taus = sorted({t for cells in grouped.values() for t in cells})
    rounds = sorted({r for cells in grouped.values() for c in cells.values() for r in c})
    print(
        f"Loaded {sum(len(c) for c in grouped.values())} runs across {len(grouped)} "
        f"(dataset, attack) combos x {len(taus)} tau values, {len(rounds)} rounds."
    )
    check_geometry(grouped)

    # Autoscale PER METRIC, not across both -- one shared scale is what buried ASR before.
    # Global over datasets, so the three files stay comparable to each other.
    ylims = {}
    for metric in ("asr", "fdr"):
        top = max(c[r][metric] for cells in grouped.values() for c in cells.values()
                  for r in rounds)
        if metric == "asr":
            top = max(top, ASR_CEILING)  # keep the ceiling on-axis; it is the ASR yardstick
        ylims[metric] = (-0.02 * top * 1.08, top * 1.08)
        print(f"  {metric.upper()} y-axis: [0, {top * 1.08:.2f}]  (data max "
              f"{max(c[r][metric] for cells in grouped.values() for c in cells.values() for r in rounds):.3f})")

    for dataset in DATASETS:
        plot_dataset(grouped, dataset, taus, rounds, ylims)
    print(f"\nAll figures saved under: {OUT_DIR.resolve()}")
