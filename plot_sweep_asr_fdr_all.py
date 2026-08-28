"""Plot Attack Success Rate (ASR) and False Detection Rate (FDR) vs quantile for
every dataset and attack, in the same single-axis overlay grammar as
``plot_sweep_accuracy_all.py`` (colour = dataset, linestyle+marker = attack, one
overlay panel per metric).

Data source
    The **July 2026-07-06 .. 2026-07-11 sweep**, read via
    ``plot_sweep_asr_fdr.collect()`` -- the single source of the ASR/FDR math (it
    counts per-client detector decisions from each run's ``client_states_rounds.csv``
    and de-duplicates duplicate (dataset, attack, tau) runs to the 50-round run).
    This is the sweep produced by the current 5-channel no-valdelta detector.

    NOTE this is a *different* sweep from the accuracy figure.
    ``plot_sweep_accuracy_all.py`` plots ``final_accuracy`` from the **June-25**
    sweep (2026-06-25 .. 06-27, old detector).  So these ASR/FDR figures are NOT a
    matched trio with that accuracy figure -- they document the *current* detector.
    (The two sweeps disagree sharply on CIFAR-10/Min-Max ASR: June-25 old detector
    ~=0.35 vs this July no-valdelta sweep ~=0.005.)  To get a matched accuracy/ASR/FDR
    trio for the current detector, regenerate accuracy from the July runs too.

    ``plot_sweep_asr_fdr.py`` plots the same July metrics but in a different grammar
    (one dual-axis figure per dataset, colour = attack, ASR+FDR+compute together);
    this script is the "overlay everything on one axis" companion to the accuracy plot.

Metrics (per the definitions used throughout this project)
    ASR = FN / N = #(is_malicious==1 & predict_state==0) / (total client-rows)
        = malicious clients the detector MISSED (the attack got through) -- higher is
          worse.
    FDR = FP / N = #(is_malicious==0 & predict_state==1) / (total client-rows)
        = benign clients wrongly flagged and dropped -- higher is worse.
    N is the total number of client participations (n_clients * n_rounds), aggregated
    over every logged round.

Figures (one overlay panel each, all 9 dataset x attack lines):
    sweep_figures/sweep_asr_all_datasets_attacks.{png,pdf}
    sweep_figures/sweep_fdr_all_datasets_attacks.{png,pdf}
FDR auto-scales its y-axis to the data (~0.05-0.39).  ASR is fixed to [0, 1] to match
the accuracy figure: on this July sweep ASR is ~0 for every combo (the detector
catches essentially all attacks; max ~0.6%), so the fixed axis reads honestly as
"ASR is negligible everywhere" rather than magnifying sub-1% noise.

Usage (from inside rl-flwr/):
    python plot_sweep_asr_fdr_all.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import (incl. the two modules below)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the accuracy plot's styling + legend helpers so the figures match exactly.
from plot_sweep_accuracy_all import (
    ATTACK_STYLES,
    DATASET_COLORS,
    IEEE_WIDTH_IN,
    _attack_handles,
    _dataset_handles,
    _single_row_legend,
)

# Reuse the exact July-sweep ASR/FDR collection (per-client FN/FP counting + dedup).
from plot_sweep_asr_fdr import collect

OUT_DIR = Path("sweep_figures")
OUT_NAME_ASR = "sweep_asr_all_datasets_attacks"
OUT_NAME_FDR = "sweep_fdr_all_datasets_attacks"


def load_data() -> pd.DataFrame:
    """Flatten ``plot_sweep_asr_fdr.collect()`` into one row per (dataset, attack, tau).

    Columns mirror the accuracy plot's (``dataset``, ``attack_mode``,
    ``tau_s_quantile``) so the shared legend/style helpers work unchanged.
    """
    grouped = collect()
    rows = []
    for (dataset, attack), recs in grouped.items():
        for r in recs:
            rows.append(
                {
                    "dataset": dataset,
                    "attack_mode": attack,
                    "tau_s_quantile": float(r["tau"]),
                    "asr": r["asr"],
                    "fdr": r["fdr"],
                    "nrounds": r["nrounds"],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "attack_mode", "tau_s_quantile"])


def _plot_metric(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    out_name: str,
    ylim: "tuple[float, float] | None" = None,
) -> None:
    """Single-axis overlay of ``column`` vs quantile, one line per dataset x attack.

    Styling is identical to ``plot_sweep_accuracy_all.plot_accuracy``.  If ``ylim`` is
    given the y-axis is fixed to it (e.g. ``(0, 1)`` to match the accuracy figure);
    otherwise it auto-scales to the data with a little headroom.
    """
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_IN, 3.15))

    for (dataset, attack), grp in df.groupby(["dataset", "attack_mode"]):
        grp = grp.sort_values("tau_s_quantile")
        style = ATTACK_STYLES.get(
            attack, {"label": attack, "linestyle": "-", "marker": "o"}
        )
        ax.plot(
            grp["tau_s_quantile"],
            grp[column],
            color=DATASET_COLORS.get(dataset, "black"),
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.15,
            markersize=3.4,
            alpha=0.95,
        )

    if ylim is not None:
        bottom, top = ylim
    else:
        ymax = float(df[column].max())
        top = ymax * 1.08 if ymax > 0 else 1.0
        # A hair of negative headroom so markers sitting exactly at 0 aren't clipped
        # by the spine.
        bottom = -0.02 * top

    ax.set_xlabel("Quantile ($q_s$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(bottom, top)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _single_row_legend(ax, _dataset_handles(df) + _attack_handles(df, with_lines=True))
    fig.subplots_adjust(bottom=0.22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{out_name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_asr(df: pd.DataFrame) -> None:
    _plot_metric(
        df,
        column="asr",
        ylabel="Attack Success Rate",
        title="Attack Success Rate vs Quantile Across Datasets and Attacks",
        out_name=OUT_NAME_ASR,
        ylim=(0.0, 1.0),  # fixed to match the accuracy figure; ASR is ~0 on this sweep
    )


def plot_fdr(df: pd.DataFrame) -> None:
    _plot_metric(
        df,
        column="fdr",
        ylabel="False Detection Rate",
        title="False Detection Rate vs Quantile Across Datasets and Attacks",
        out_name=OUT_NAME_FDR,
    )


if __name__ == "__main__":
    data = load_data()
    print(
        f"Loaded {len(data)} runs across "
        f"{data['dataset'].nunique()} datasets and "
        f"{data['attack_mode'].nunique()} attacks "
        f"(rounds present: {sorted(data['nrounds'].unique())})."
    )
    print(
        f"  ASR range [{data['asr'].min():.4f}, {data['asr'].max():.4f}]  "
        f"FDR range [{data['fdr'].min():.4f}, {data['fdr'].max():.4f}]"
    )
    plot_asr(data)
    plot_fdr(data)
