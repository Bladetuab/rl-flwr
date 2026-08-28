"""plot_sweep_asr_fdr_fashion_mnist.py  --  Fashion-MNIST-only ASR / FDR / ConvLSTM-forwards figure.

Standalone, per-dataset split of ``plot_sweep_asr_fdr.py``.  Self-contained: no shared
imports, so this file alone reproduces the Fashion-MNIST panel.  The other two per-dataset
variants are byte-identical except for ``DATASET`` / ``DATASET_LABEL``.

  x-axis  : tau_s = re-evaluation trigger quantile q_s, read from the run-dir name
            ``*_tau<val>``.
  left  y : detection error rates in [0, 1]
              - ASR (solid)  = FN / N = malicious clients the detector MISSED / N
              - FDR (dashed) = FP / N = benign clients wrongly flagged / N
  right y : total ConvLSTM forward passes (dotted) = sum over rounds of n_scored.
N = total client participations (n_clients * n_rounds), aggregated over every round.

Reads the July 2026-07-06 .. 2026-07-11 sweep (current 5-channel no-valdelta
detector); duplicate (attack, tau) runs across dates are de-duplicated to the
50-round run (prefer 50 rounds, then most rounds, then latest date).

Output:  sweep_figures/sweep_asr_fdr_forwards_fashion_mnist.{png,pdf}
Usage:   python plot_sweep_asr_fdr_fashion_mnist.py
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
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

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

DATASET = "fashion_mnist"          # <-- the ONLY per-file difference (with DATASET_LABEL)
DATASET_LABEL = "Fashion-MNIST"

ATTACKS = ["gaussian", "lie", "min_max"]
ATTACK_LABELS = {"gaussian": "Gaussian", "lie": "LIE", "min_max": "Min-Max"}
# Okabe-Ito colour-blind-safe categorical hues.
ATTACK_COLORS = {"gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2"}

# Axis pinning.  Leave None to auto-scale to THIS dataset's data; set to the global
# values below to make all three per-dataset figures share identical axes:
#     RATE_MAX = 0.4524   COST_MAX = 3585   (max over all datasets, July sweep)
RATE_MAX = None
COST_MAX = None

RUN_RE = re.compile(rf"^\d\d-\d\d-\d\d_{DATASET}_(gaussian|lie|min_max)_tau([0-9.]+)$")

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
def run_metrics(run_dir: Path) -> dict:
    """ASR/FDR counts, total ConvLSTM forward passes and round count for a run dir."""
    fn = fp = tp = tn = ntot = 0
    forwards = None
    nrounds = 0

    cs = run_dir / "client_states_rounds.csv"
    if cs.exists():
        with open(cs) as f:
            for row in csv.DictReader(f):
                is_mal = int(row["is_malicious"])
                pred = int(row["predict_state"])
                ntot += 1
                if is_mal == 1:
                    tp += pred == 1
                    fn += pred == 0
                else:
                    fp += pred == 1
                    tn += pred == 0

    cc = run_dir / "convlstm_cost.csv"
    if cc.exists():
        with open(cc) as f:
            rows = list(csv.DictReader(f))
        forwards = sum(int(r["n_scored"]) for r in rows)

    rj = run_dir / "results.json"
    if rj.exists():
        try:
            import json

            nrounds = len(json.load(open(rj)))
        except (ValueError, TypeError):
            nrounds = 0

    return dict(fn=fn, fp=fp, tp=tp, tn=tn, ntot=ntot, forwards=forwards, nrounds=nrounds)


def collect() -> dict:
    """Return {attack: [row, ...]} for DATASET, sorted by tau, deduped to 50-round runs."""
    raw = []
    for date in DATES:
        dd = ROOT / "outputs" / date
        if not dd.exists():
            continue
        for name in sorted(os.listdir(dd)):
            m = RUN_RE.match(name)
            if not m:
                continue
            attack, tau = m.group(1), float(m.group(2))
            met = run_metrics(dd / name)
            if met["ntot"] == 0 or met["forwards"] is None:
                continue  # skip empty / incomplete runs
            met.update(date=date, attack=attack, tau=tau)
            raw.append(met)

    def rank(r):
        return (1 if r["nrounds"] == 50 else 0, r["nrounds"], r["date"])

    best = {}
    for r in raw:
        key = (r["attack"], r["tau"])
        if key not in best or rank(r) > rank(best[key]):
            best[key] = r

    grouped = defaultdict(list)
    for (attack, tau), r in best.items():
        n = r["ntot"]
        grouped[attack].append(
            dict(
                tau=tau,
                asr=r["fn"] / n,
                fdr=r["fp"] / n,
                forwards=r["forwards"],
                nrounds=r["nrounds"],
            )
        )
    for k in grouped:
        grouped[k].sort(key=lambda r: r["tau"])
    return grouped


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(grouped: dict) -> None:
    if COST_MAX is not None:
        cost_max = COST_MAX
    else:
        cost_max = max(r["forwards"] for rows in grouped.values() for r in rows)
    if RATE_MAX is not None:
        rate_max = RATE_MAX
    else:
        rate_max = max(max(r["asr"], r["fdr"]) for rows in grouped.values() for r in rows)
        rate_max = min(1.0, rate_max * 1.15)

    fig, ax_l = plt.subplots(figsize=(IEEE_WIDTH_IN * 0.62, 3.4))
    ax_r = ax_l.twinx()

    # Right axis (compute cost) first, coloured by attack (dotted, open markers).
    for attack in ATTACKS:
        rows = grouped.get(attack)
        if not rows:
            continue
        ax_r.plot([r["tau"] for r in rows], [r["forwards"] for r in rows],
                  color=ATTACK_COLORS[attack], linestyle=":", linewidth=1.2,
                  marker="^", markersize=2.8, markerfacecolor="white",
                  markeredgewidth=0.8, alpha=0.9, zorder=2)

    # Left axis (detection quality), foreground, coloured by attack.
    for attack in ATTACKS:
        rows = grouped.get(attack)
        if not rows:
            continue
        color = ATTACK_COLORS[attack]
        tau = [r["tau"] for r in rows]
        ax_l.plot(tau, [r["fdr"] for r in rows], color=color, linestyle="--",
                  linewidth=1.2, marker="s", markersize=3.0, alpha=0.95, zorder=3)
        ax_l.plot(tau, [r["asr"] for r in rows], color=color, linestyle="-",
                  linewidth=1.6, marker="o", markersize=3.6, zorder=5)

    ax_l.set_xlim(-0.03, 1.03)
    ax_l.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax_l.set_ylim(0.0, rate_max)
    ax_r.set_ylim(0.0, cost_max * 1.1)
    ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    # ASR sits at ~0; make that explicit since it hugs the axis.
    ax_l.annotate("ASR $\\approx$ 0", xy=(0.03, 0.006), xycoords=("axes fraction", "data"),
                  fontsize=IEEE_FONT - 2, color="#555555", va="bottom")

    ax_l.set_xlabel(r"$q_s$")
    ax_l.set_ylabel("ASR / FDR")
    ax_r.set_ylabel("Total ConvLSTM forward passes")
    ax_l.set_title(f"{DATASET_LABEL}: ASR and FDR with total ConvLSTM forwards over $q_s$")

    attack_h = [
        Line2D([0], [0], color=ATTACK_COLORS[a], linewidth=1.6, label=ATTACK_LABELS[a])
        for a in ATTACKS
    ]
    metric_h = [
        Line2D([0], [0], color="0.25", linestyle="-", marker="o", markersize=3.6,
               linewidth=1.6, label="ASR (Attack Success Rate)"),
        Line2D([0], [0], color="0.25", linestyle="--", marker="s", markersize=3.0,
               linewidth=1.2, label="FDR (False Detection Rate)"),
        Line2D([0], [0], color="0.25", linestyle=":", marker="^", markersize=2.8,
               markerfacecolor="white", markeredgewidth=0.8, linewidth=1.2,
               label="Total ConvLSTM forwards"),
    ]
    fig.legend(handles=attack_h + metric_h, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=2.5, handlelength=1.8, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"sweep_asr_fdr_forwards_{DATASET}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    grouped = collect()
    npts = sum(len(v) for v in grouped.values())
    print(f"{DATASET}: loaded {npts} (attack, tau) points across {len(grouped)} attacks.")
    plot(grouped)
    print(f"Figure saved under: {OUT_DIR.resolve()}")
