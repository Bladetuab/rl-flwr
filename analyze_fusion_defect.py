"""
analyze_fusion_defect.py — evidence for IMPROVEMENT.md Section 1.

Measures, from ALREADY-LOGGED runs (no re-run required), why the shipped
`fused = max(stat_ewma, convlstm_rel)` scoring loses the separation that the
statistical screen alone achieves:

  1. the temporal arm is informative   (AUC 0.92-0.98 -- not noise)
  2. it separates well WITHIN a round  (0% benign FP in 29-42 of 45 rounds)
     but its scale drifts ACROSS them, so no fixed global threshold holds
  3. `max` fusion makes that decisive  (crossover set is 93-100% BENIGN)
  4. re-weighting restores RANKING     (AUC 1.0000 in all six cells)
     but NOT thresholdability -- the single-cut margin stays negative in 5/6,
     so per-round calibration, not a new fusion weight, is the actual fix.

Reads `stat_anomaly.csv` written by CustomFedAvg_adaptive_new._append_stat_csv.
Restricted to rounds >= 6 (the attacked rounds); pooling the warm-up rounds
inflates the methods unequally and is not comparable across them.

Usage:  python analyze_fusion_defect.py [--tau tau0.40] [--abs 0.40]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

ONSET_ROUND = 6  # adversary is dormant through the T_w = 5 warm-up rounds


def auc(y, s):
    """Rank-based Mann-Whitney AUC (avoids a scipy/sklearn dependency)."""
    y = np.asarray(y)
    r = pd.Series(np.asarray(s)).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def robust_z(df, col):
    """Per-round robust z against the round's own median and MAD."""
    g = df.groupby("round")[col]
    med = g.transform("median")
    mad = g.transform(lambda x: np.median(np.abs(x - np.median(x))))
    return (df[col] - med) / np.maximum(mad, 1e-6)


def cell_name(d, tau):
    base = os.path.basename(d)
    dataset, attack = base.split("_ustad_af_")
    return f"{dataset[9:]}/{attack.replace('_' + tau, '')}"


def find_runs(tau):
    runs = glob.glob(f"outputs/2026-08-1*/*ustad_af_*{tau}")
    # exclude the ablation variants; we want the shipped configuration
    return sorted(d for d in runs if "absonly" not in d and "meansvd" not in d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", default="tau0.40", help="operating point directory suffix")
    p.add_argument("--abs", type=float, default=0.40, help="tau_abs, the absolute cap")
    args = p.parse_args()

    runs = find_runs(args.tau)
    if not runs:
        raise SystemExit(f"no ustad_af runs found for {args.tau} under outputs/")

    hdr = (f"{'cell':<24}{'AUC stat':>9}{'AUC rel':>8}{'AUC max':>8}"
           f"{'AUC zmax':>9}{'AUC wsum':>9}{'relM':>7}{'relB':>7}"
           f"{'FP med':>8}{'FP mean':>9}{'clean rds':>11}{'cross':>7}{'benign%':>9}")
    print(hdr)
    print("-" * len(hdr))

    for d in runs:
        df = pd.read_csv(os.path.join(d, "stat_anomaly.csv"))
        a = df[df["round"] >= ONSET_ROUND].copy()
        y = a.is_malicious.values
        mal, ben = a[y == 1], a[y == 0]

        # two repair candidates: calibrate-then-max, and a plain weighted sum
        a["zmax"] = np.maximum(robust_z(a, "stat_ewma"), robust_z(a, "convlstm_rel"))
        a["wsum"] = 0.8 * a.stat_ewma + 0.2 * a.convlstm_rel

        # Benign clients that also fire at the cut catching every attacker.
        # Measured PER ROUND: pooling all rounds lets one low-scoring malicious
        # client-round set the floor for the whole campaign, which reports ~97%
        # mechanically and says nothing about within-round separation.
        per_round = []
        for _, g in a.groupby("round"):
            m, b = g[g.is_malicious == 1], g[g.is_malicious == 0]
            if len(m) and len(b):
                per_round.append(100 * (b.convlstm_rel >= m.convlstm_rel.min()).mean())
        per_round = np.array(per_round)
        clean_rounds = f"{(per_round == 0).sum()}/{len(per_round)}"

        # clients `fused` pushes over the cap that `stat_ewma` alone would not
        cross = a[(a.fused_score > args.abs) & (a.stat_ewma <= args.abs)]
        cross_benign = 100 * (cross.is_malicious == 0).mean() if len(cross) else float("nan")

        print(f"{cell_name(d, args.tau):<24}"
              f"{auc(y, a.stat_ewma):>9.4f}{auc(y, a.convlstm_rel):>8.4f}"
              f"{auc(y, a.fused_score):>8.4f}{auc(y, a.zmax):>9.4f}"
              f"{auc(y, a.wsum):>9.4f}"
              f"{mal.convlstm_rel.mean():>7.3f}{ben.convlstm_rel.mean():>7.3f}"
              f"{np.median(per_round):>8.1f}{per_round.mean():>9.1f}"
              f"{clean_rounds:>11}{len(cross):>7}{cross_benign:>9.1f}")

    print("\nrelM/relB: mean convlstm_rel for malicious / benign clients.")
    print("FP med/mean: per-round % of benign clients firing at the cut that catches")
    print("             every attacker that round. clean rds: rounds with exactly 0%.")
    print("cross:     client-rounds pushed over tau_abs by fusion alone; benign% of those.")


if __name__ == "__main__":
    main()
