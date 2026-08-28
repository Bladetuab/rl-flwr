"""plot_defense_curves_50r.py  --  per-attack accuracy curves, matched 50-round protocol.

Regenerates the paper's Figs. 2 and 3. The originals plotted the 30-round / 5-local-epoch
campaign against the superseded adaptive-quantile variant; every defense has since been
re-run at 50 rounds / 10 local epochs, so these curves are drawn from the same runs that
produce Table~IV and Table~V and are protocol-matched to them.

  x-axis : communication round (1..50)
  y-axis : global test accuracy, evaluate_metrics_serverapp.accuracy in results.json --
           the same quantity `_collect_metrics` reads for the final-accuracy tables, so a
           curve's right-hand endpoint IS its table entry.
  colour : defense (fixed assignment, never cycled)
  cols   : attack
  figure : one per dataset

Run provenance is the run-DIR name, taken via the `save_path` column of the sweep summary
CSVs -- the same source the tables use, so a figure can never silently disagree with a
table. run_config.json reports one combination for every run in a sweep and would mislabel
every point.

Palette: Okabe-Ito, validated for 5 categorical slots against a light surface
(all checks pass; worst adjacent CVD dE 11.4 protan / 8.5 tritan, worst normal-vision
dE 16.4). Two slots fall below 3:1 contrast, for which the required relief is present:
every series is named in the legend and its endpoint value appears in Table IV.

Figures (sweep_figures/):
    defense_curves_mnist.{png,pdf}
    defense_curves_fashion_mnist.{png,pdf}

Usage (from inside rl-flwr/):
    python plot_defense_curves_50r.py
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

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sweep_figures"
SUMMARIES = ROOT / "outputs"

DATASETS = ["mnist", "fashion_mnist"]
ATTACKS = ["gaussian", "lie", "min_max"]

# Where each defense's matched 50-round runs live. USTAD-FL is taken at its selected
# operating point q_s = 0.4; the baselines have no such knob and ran once at 0.6.
SOURCES = {
    "ustad_af":   ("sweep_summary_20260814_152111.csv", 0.4),
    "dnc":        ("sweep_summary_20260813_145249.csv", None),
    "multikrum":  ("sweep_summary_20260816_205617.csv", None),
    "fltrust":    ("sweep_summary_20260816_205617.csv", None),
    "fldetector": ("sweep_summary_20260816_205617.csv", None),
}

# Fixed order and fixed hues: colour follows the defense, never its rank in a panel.
DEFENSE_ORDER = ["ustad_af", "dnc", "multikrum", "fltrust", "fldetector"]
DEFENSE_LABELS = {
    "ustad_af": "USTAD-FL (ours)",
    "dnc": "DnC$^\\dagger$",
    "multikrum": "Multi-Krum$^\\dagger$",
    "fltrust": "FLTrust",
    "fldetector": "FLDetector",
}
DEFENSE_COLORS = {
    "ustad_af": "#0072B2",
    "dnc": "#009E73",
    "multikrum": "#E69F00",
    "fltrust": "#CC79A7",
    "fldetector": "#D55E00",
}
# Ours is drawn last and heavier: identity is still carried by the legend and hue, the
# weight only stops it being buried where curves coincide.
DEFENSE_LW = {d: 1.05 for d in DEFENSE_ORDER}
DEFENSE_LW["ustad_af"] = 1.7

DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}
ATTACK_LABELS = {"gaussian": "Gaussian", "lie": "LIE", "min_max": "Min-Max"}

ONSET_ROUND = 6
N_ROUNDS = 50

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
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


# ── Data ──────────────────────────────────────────────────────────────────────
def accuracy_curve(run_dir: Path) -> dict:
    """{round: global test accuracy} from results.json."""
    f = run_dir / "results.json"
    if not f.exists():
        return {}
    data = json.load(open(f))
    out = {}
    for entry in data:
        acc = entry.get("evaluate_metrics_serverapp", {}).get("accuracy")
        if acc is not None:
            out[int(entry["round"])] = float(acc)
    return out


def collect() -> dict:
    """{(dataset, attack, defense): {round: acc}}."""
    out = {}
    for defense, (summary, tau) in SOURCES.items():
        path = SUMMARIES / summary
        assert path.exists(), f"missing sweep summary: {path}"
        for row in csv.DictReader(open(path)):
            if row.get("error"):
                continue
            if row["dataset"] not in DATASETS or row["attack_mode"] not in ATTACKS:
                continue
            # The 20260814 sweep is USTAD-FL only; the others carry a defense column.
            if row["defense"] != defense:
                continue
            if tau is not None and abs(float(row["tau_s_quantile"]) - tau) > 1e-9:
                continue
            curve = accuracy_curve(Path(row["save_path"]))
            if curve:
                out[(row["dataset"], row["attack_mode"], defense)] = curve

    missing = [
        (d, a, m) for d in DATASETS for a in ATTACKS for m in DEFENSE_ORDER
        if (d, a, m) not in out
    ]
    assert not missing, f"no matched run for: {missing}"
    return out


def check_geometry(data: dict) -> None:
    """Every curve must be the full 50 rounds, or the panels are not comparable."""
    for key, curve in sorted(data.items()):
        assert len(curve) == N_ROUNDS, f"{key}: {len(curve)} rounds, expected {N_ROUNDS}"
        assert min(curve) == 1 and max(curve) == N_ROUNDS, f"{key}: round range {min(curve)}..{max(curve)}"
    print(f"  geometry OK on all {len(data)} curves: rounds 1..{N_ROUNDS}")


def report(data: dict) -> None:
    """Print each curve's endpoint -- these must equal the Table IV entries."""
    print("\n  Final-round accuracy (must match Table IV):")
    print(f"  {'defense':12s} " + " ".join(f"{a[:8]:>9s}" for a in ATTACKS) + "   dataset")
    for dataset in DATASETS:
        for defense in DEFENSE_ORDER:
            vals = [data[(dataset, a, defense)][N_ROUNDS] for a in ATTACKS]
            print(f"  {defense:12s} " + " ".join(f"{v:9.4f}" for v in vals) + f"   {dataset}")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(data: dict, dataset: str) -> None:
    rounds = list(range(1, N_ROUNDS + 1))
    fig, axes = plt.subplots(
        1, len(ATTACKS), figsize=(IEEE_WIDTH_IN, 2.25), sharey=True, layout="constrained"
    )
    for ax, attack in zip(axes, ATTACKS):
        ax.axvspan(0.5, ONSET_ROUND - 0.5, color="0.5", alpha=0.10, linewidth=0, zorder=0)
        ax.axvline(ONSET_ROUND, color="0.35", linewidth=0.7, zorder=1)
        ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for defense in DEFENSE_ORDER:
            curve = data[(dataset, attack, defense)]
            ax.plot(
                rounds, [curve[r] for r in rounds],
                color=DEFENSE_COLORS[defense], linewidth=DEFENSE_LW[defense],
                zorder=3 + (defense == "ustad_af"), solid_capstyle="round",
            )
        ax.set_xlim(0.5, N_ROUNDS + 0.5)
        ax.set_xticks([1, ONSET_ROUND, 10, 20, 30, 40, 50])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(ATTACK_LABELS[attack])
        ax.set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy")

    handles = [
        Line2D([0], [0], color=DEFENSE_COLORS[d], linewidth=DEFENSE_LW[d] + 0.3,
               label=DEFENSE_LABELS[d])
        for d in DEFENSE_ORDER
    ] + [Patch(facecolor="0.5", alpha=0.10, label="Warm-up rounds")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False,
               columnspacing=1.2, handlelength=1.8, handletextpad=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"defense_curves_{dataset}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    data = collect()
    print(f"Loaded {len(data)} curves ({len(DEFENSE_ORDER)} defenses x "
          f"{len(ATTACKS)} attacks x {len(DATASETS)} datasets).")
    check_geometry(data)
    report(data)
    for dataset in DATASETS:
        plot(data, dataset)
    print(f"\nFigures saved under: {OUT_DIR.resolve()}")
