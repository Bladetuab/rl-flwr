"""plot_sweep_asr_fdr.py

Per-dataset dual-axis sweep figures over the tau_s (re-evaluation trigger quantile)
sweep logged in outputs/2026-07-06 .. 2026-07-11.

  x-axis      : tau_s = tau_s_quantile = q_s, the ConvLSTM re-evaluation trigger
                quantile, read from the run-dir name  *_tau<val>.  (server_sweep.py
                passes the swept value as tau_s_quantile=tau; the tau-s-quantile:0.8
                in run_config.json is a stale default, not the per-run value.)
  left  y     : detection error rates in [0,1]
                  - ASR (solid)  = FN / N  = (# malicious clients the detector MISSED) / (# clients)
                  - FDR (dashed) = FP / N  = (# benign clients wrongly flagged)        / (# clients)
  right y     : total ConvLSTM forward passes (dotted) = sum over rounds of n_scored
                (the number of clients the ConvLSTM scored per round)

One figure per dataset (mnist, fashion_mnist, cifar10); three attack lines each
(colour = attack). A combined 1x3 panel is also produced.

Definitions (as specified by the user):
  ASR = #(is_malicious==1 & predict_state==0) / (total client-rows)
  FDR = #(is_malicious==0 & predict_state==1) / (total client-rows)
where per-client decisions come from client_states_rounds.csv (predict_state == the
detector/RL flag: 1 => flagged malicious => dropped from aggregation), aggregated over
every logged round; the denominator N is the total number of client participations
(n_clients * n_rounds).

Duplicate (dataset, attack, tau) runs across dates are de-duplicated to the 50-round
run for cross-dataset comparability (a 100-round CIFAR-Gaussian sweep also exists on
2026-07-06 but mixing round counts is misleading, so it is excluded).

Usage (from inside rl-flwr/):
    python plot_sweep_asr_fdr.py
"""

import csv
import json
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
# Okabe-Ito colour-blind-safe categorical hues (validated: worst adjacent CVD dE 17).
ATTACK_COLORS = {"gaussian": "#D55E00", "lie": "#009E73", "min_max": "#0072B2"}

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
            d = json.load(open(rj))
            nrounds = len(d)
        except (json.JSONDecodeError, ValueError):
            nrounds = 0

    return dict(fn=fn, fp=fp, tp=tp, tn=tn, ntot=ntot, forwards=forwards, nrounds=nrounds)


def collect() -> dict:
    """Return {(dataset, attack): [row, ...]} sorted by tau, deduped to 50-round runs."""
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
            met = run_metrics(dd / name)
            if met["ntot"] == 0 or met["forwards"] is None:
                continue  # skip empty / incomplete runs
            met.update(date=date, dataset=dataset, attack=attack, tau=float(tau))
            raw.append(met)

    # de-dup: prefer 50-round run, then most rounds, then latest date
    def rank(r):
        return (1 if r["nrounds"] == 50 else 0, r["nrounds"], r["date"])

    best = {}
    for r in raw:
        key = (r["dataset"], r["attack"], r["tau"])
        if key not in best or rank(r) > rank(best[key]):
            best[key] = r

    grouped = defaultdict(list)
    for (dataset, attack, tau), r in best.items():
        n = r["ntot"]
        grouped[(dataset, attack)].append(
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
def _draw(ax_l, ax_r, grouped, dataset, cost_max, rate_max):
    # Right axis (compute cost) first, coloured by attack (dotted, open markers).
    for attack in ATTACKS:
        rows = grouped.get((dataset, attack))
        if not rows:
            continue
        ax_r.plot([r["tau"] for r in rows], [r["forwards"] for r in rows],
                  color=ATTACK_COLORS[attack], linestyle=":", linewidth=1.2,
                  marker="^", markersize=2.8, markerfacecolor="white",
                  markeredgewidth=0.8, alpha=0.9, zorder=2)

    # Left axis (detection quality), foreground, coloured by attack.
    for attack in ATTACKS:
        rows = grouped.get((dataset, attack))
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


def _legend_handles():
    attack_h = [
        Line2D([0], [0], color=ATTACK_COLORS[a], linewidth=1.6,
               label=ATTACK_LABELS[a])
        for a in ATTACKS
    ]
    metric_h = [
        Line2D([0], [0], color="0.25", linestyle="-", marker="o", markersize=3.6,
               linewidth=1.6, label="ASR"),
        Line2D([0], [0], color="0.25", linestyle="--", marker="s", markersize=3.0,
               linewidth=1.2, label="FDR"),
        Line2D([0], [0], color="0.25", linestyle=":", marker="^", markersize=2.8,
               markerfacecolor="white", markeredgewidth=0.8, linewidth=1.2,
               label="Total ConvLSTM forwards"),
    ]
    return attack_h, metric_h


def plot_per_dataset(grouped):
    cost_max = max(r["forwards"] for rows in grouped.values() for r in rows)
    rate_max = max(max(r["asr"], r["fdr"]) for rows in grouped.values() for r in rows)
    rate_max = min(1.0, rate_max * 1.15)
    attack_h, metric_h = _legend_handles()

    for dataset in DATASETS:
        fig, ax_l = plt.subplots(figsize=(IEEE_WIDTH_IN * 0.62, 3.4))
        ax_r = ax_l.twinx()
        _draw(ax_l, ax_r, grouped, dataset, cost_max, rate_max)

        ax_l.set_xlabel(r"$q$")
        ax_l.set_ylabel("ASR / FDR")
        ax_r.set_ylabel("Total ConvLSTM forward passes")
        ax_l.set_title(f"{DATASET_LABELS[dataset]}: ASR and FDR with compute over $\\tau_s$")

        leg1 = ax_l.legend(handles=attack_h, loc="upper left", title="Attack",
                           framealpha=0.9, borderpad=0.4)
        ax_l.add_artist(leg1)
        ax_l.legend(handles=metric_h, loc="upper right", framealpha=0.9, borderpad=0.4)

        fig.tight_layout()
        _save(fig, f"sweep_asr_fdr_forwards_{dataset}")


def plot_combined(grouped):
    cost_max = max(r["forwards"] for rows in grouped.values() for r in rows)
    rate_max = max(max(r["asr"], r["fdr"]) for rows in grouped.values() for r in rows)
    rate_max = min(1.0, rate_max * 1.15)
    attack_h, metric_h = _legend_handles()

    fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDTH_IN, 3.0))
    for ax_l, dataset in zip(axes, DATASETS):
        ax_r = ax_l.twinx()
        _draw(ax_l, ax_r, grouped, dataset, cost_max, rate_max)
        ax_l.set_title(DATASET_LABELS[dataset])
        ax_l.set_xlabel(r"$\tau_s$")
        if dataset == DATASETS[0]:
            ax_l.set_ylabel("ASR and FDR)")
        else:
            ax_l.set_yticklabels([])
        if dataset == DATASETS[-1]:
            ax_r.set_ylabel("Total ConvLSTM forwards")
        else:
            ax_r.set_yticklabels([])

    fig.legend(handles=attack_h + metric_h, loc="lower center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "sweep_asr_fdr_forwards_combined")


def export_csv(grouped):
    out = OUT_DIR / "sweep_asr_fdr_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "attack", "tau", "nrounds", "asr", "fdr", "total_convlstm_forwards"])
        for dataset in DATASETS:
            for attack in ATTACKS:
                for r in grouped.get((dataset, attack), []):
                    w.writerow([dataset, attack, f"{r['tau']:.1f}", r["nrounds"],
                                f"{r['asr']:.5f}", f"{r['fdr']:.5f}", r["forwards"]])
    print(f"  Saved -> {out}")


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    grouped = collect()
    npts = sum(len(v) for v in grouped.values())
    print(f"Loaded {npts} (dataset, attack, tau) points "
          f"across {len({k[0] for k in grouped})} datasets.")
    export_csv(grouped)
    plot_per_dataset(grouped)
    plot_combined(grouped)
    print(f"\nAll figures saved under: {OUT_DIR.resolve()}")
