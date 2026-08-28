# Provenance — Table IV (tab:main_results), main results table

**Status 2026-08-12: Fang REMOVED from the paper at user request.** Table IV
is back to 3 attacks x 2 datasets (30 rounds), plus a derived worst-case
column. The Fang runs are real and remain on disk; the numbers are kept below
so the decision is reversible, but nothing in the Fang sections is currently
cited by `fl_sec.tex`. Drafts: `fl_sec.pre_fang_20260812.tex` (before Fang was
ever added), current `fl_sec.tex` (Fang added, then removed; retains the
math reduction and the consistency framing).

Added 2026-07-02. Every cell traces to a real run in
`/home/tuan/get-local-model-bt-id/rl-flwr/outputs/`. Metric =
`evaluate_metrics_serverapp.accuracy` at the final round (30) of each run's
`results.json`. Table shows 3 decimals; raw 4-decimal values below.
All values were additionally verified visually against the endpoints of
Figs. 2–4 (accuracy_plot_*.png).

## MNIST

| Method | Gaussian | LIE | Min-Max |
|---|---|---|---|
| FedAvg (no defense) | 0.0891 — `2026-04-01/gaussian` | 0.1135 — `2026-04-01/fedavg_lie` | 0.0982 — `2026-04-01/fedavg_min_max` |
| FLDetector | 0.1034 — `2026-04-09/22-33-56_gaussian_FLdetector_mnist` | 0.1028 — `2026-04-09/22-22-04_lie_FLdetector_mnist` | 0.1028 — `2026-04-09/21-47-29_minmax_FLdetector_mnist` |
| FLTrust | 0.6504 — `2026-04-08/15-15-33_gaussian_FLtrust_mnist` | 0.6930 — `2026-04-08/15-33-56_lie_FLtrust_mnist` | 0.7644 — `2026-04-08/14-52-58_minmax_FLtrust_mnist` |
| Multi-Krum | 0.8848 — `2026-04-08/18-25-18_gaussian_multikrum_mnist` | 0.8881 — `2026-04-08/18-13-13_lie_multikrum_mnist` | 0.5769 — `2026-04-08/17-19-36_min_max_multikrum_mnist` |
| USTAD-FL | 0.8884 — `2026-04-08/21-11-36_gaussian_mnist` | 0.8808 — `2026-04-08/20-59-54_lie_mnist` | 0.8701 — `2026-04-08/19-05-26_min_max_mnist` |
| FedAvg (no attack) | 0.8587 — `2026-04-01/fedavg_no_attack` | | |

## Fashion-MNIST

| Method | Gaussian | LIE | Min-Max |
|---|---|---|---|
| FedAvg (no defense) | 0.1001 — `2026-04-02/17-35-06_gaussian` | 0.1000 — `2026-04-02/17-49-32_lie` | 0.1999 — `2026-04-02/18-02-27_min_max` |
| FLDetector | 0.1003 — `2026-04-10/11-46-15_gaussian_FLdetector_famnist` | 0.1000 — `2026-04-10/12-06-09_lie_FLdetector_famnist` | 0.7036 — `2026-04-10/12-19-12_minmax_FLdetector_famnist` |
| FLTrust | 0.5865 — `2026-04-09/19-42-14_gaussian_FLtrust_famnist` | 0.6508 — `2026-04-09/19-58-34_lie_FLtrust_famnist` | 0.6656 — `2026-04-09/20-36-37_minmax_FLtrust_famnist` |
| Multi-Krum | 0.7256 — `2026-04-09/16-29-46_gaussian_multikrum_famnist` | 0.7286 — `2026-04-09/16-42-20_lie_multikrum_famnist` | 0.7332 — `2026-04-09/16-54-24_min_max_multikrum_famnist` |
| USTAD-FL | 0.7442 — `2026-04-09/16-16-15_gaussian_famnist` | 0.7355 — `2026-04-09/15-44-04_lie_famnist` | 0.7414 — `2026-04-09/15-57-26_min_max_famnist` |
| FedAvg (no attack) | 0.7427 — `2026-04-02/17-22-55_no_attack` | | |

## Fang results (RUN AND VERIFIED, but NOT in the current paper)

Same metric and horizon as the rest of Table IV (round 30, 5 local epochs).

| Method | MNIST | Fashion-MNIST |
|---|---|---|
| FedAvg (no defense) | 0.7784 — `2026-04-10/15-45-53_fang_FedAvg_mnist` | 0.7120 — `2026-04-10/16-00-36_fang_FedAvg_famnist` |
| FLDetector | 0.8777 — `2026-04-09/22-01-08_fang_FLdetector_mnist` | 0.7379 — `2026-04-10/12-32-47_fang_FLdetector_famnist` |
| FLTrust | 0.7025 — `2026-04-08/14-33-19_fang_FLtrust_mnist` | 0.6318 — `2026-04-09/20-17-08_fang_FLtrust_famnist` |
| Multi-Krum | 0.0958 — `2026-04-08/16-53-19_fang_multikrum_mnist` | 0.5414 — `2026-04-09/17-08-36_fang_multikrum_famnist` |
| USTAD-FL | 0.7701 — `2026-04-08/20-14-25_fang_mnist` | 0.6984 — `2026-04-09/15-27-35_fang_famnist` |

**Worst-case column (in the current paper).** Derived, not a separate run: the
row-wise minimum over the SIX accuracy cells of that row (3 attacks x 2
datasets), undefended FedAvg row included. Values: FedAvg 0.089, FLDetector
0.100, FLTrust 0.587, Multi-Krum 0.577, USTAD-FL **0.736** -- USTAD-FL leads
the strongest baseline (FLTrust) by 14.9 points. Row means over the same six:
FedAvg 0.117, FLDetector 0.202, FLTrust 0.668, Multi-Krum 0.756,
USTAD-FL 0.810 (+5.4).

*(If Fang is ever restored, the eight-cell versions are: worst = FedAvg 0.089,
FLDetector 0.100, FLTrust 0.587, Multi-Krum 0.096, USTAD-FL 0.698, margin
+11.1; means = 0.274 / 0.354 / 0.668 / 0.647 / 0.791, margin +12.3.)*

**Duplicate-run note.** `final_accuracy_by_dataset.md` lists a second MNIST
USTAD-FL Fang run, `2026-04-09/15-04-42_fang_mnist` (0.7832). Table IV uses
the **2026-04-08** run (0.7701) because the other three MNIST USTAD-FL cells
(Gaussian/LIE/Min-Max) are all from the 2026-04-08 campaign; mixing dates
within one row would break the matched comparison. Using 0.7832 instead does
not change any ranking: it is still below FLDetector (0.8777) and below
undefended FedAvg (0.7784).

**Fang sweep (50-round campaign; not cited by the paper).** 22 runs,
`outputs/2026-07-24/*_mnist_fang_tau*` and
`outputs/2026-07-24|25/*_fashion_mnist_fang_tau*`. MNIST range
[0.7841, 0.8834] (spread 9.9 pts; max at tau=0.50, min at tau=1.00);
F-MNIST [0.6272, 0.7141] (spread 8.7 pts; max at tau=0.80, min at tau=0.40).
No-attack 50-round references: MNIST 0.9622
(`2026-07-10/15-16-55_mnist_no_attack_tau0.00`), F-MNIST 0.8167
(`2026-07-11/13-40-49_fashion_mnist_no_attack_tau0.00`).

**TRAP — do not use `final_accuracy_by_attack_defense_dataset.md`** for
Table IV. That file pairs 30-round baselines with the *best-tau 50-round*
USTAD-FL run (e.g. Fang/Our Method/MNIST = 0.8834), which is a different
training budget (50 rounds, 10 local epochs) and a best-of-11 selection. It
makes USTAD-FL win every column by comparing incommensurable campaigns.

**Detector-level analysis excludes Fang.** `decider_results.csv` is 66 rows
(3 attacks x 2 datasets x 11 q_s). The Fang sweep runs carry *identical*
instrumentation (`stat_anomaly.csv` with
`stat_raw,stat_ewma,convlstm_rel,fused_score,is_malicious`), so
`decider_offline.py` could extend the ablation to 88 rows; this has not been
run. Moot while Fang is out of the paper, but the prerequisite for putting it
back with detector-level evidence rather than accuracy alone.

## Why the table is limited to two benchmarks

An audit on 2026-07-02 found that the third-dataset results then present in the
draft were **not backed by real runs**. The dataset is deliberately unnamed here;
what matters is the failure mode, because it is the reason this table exists and
the guardrail against repeating it:

1. **Replotted curves.** The "Our Method" curves in one figure were a single
   no-attack FedAvg run (`2026-04-02/18-41-29_no_attack`, final 0.6170)
   replotted identically across all three attack panels. The baseline curves in
   those panels were genuine, which is what made the substitution hard to see.
2. **Fabricated sweep rows.** In `outputs/sweep_summary_20260625_182323.csv`
   (mtime Jun 27 19:15, ~16 h after the last run), 30 of 33 rows for that
   dataset contradicted the `results.json` of the run directories they
   referenced. Its LIE rows had been copied from the May 6-7 sweep, which is
   **Fashion-MNIST** — verified by an exact accuracy-and-loss match against
   genuine June F-MNIST runs; most Min-Max rows were copied from the Apr 25-26
   sweep. The MNIST and Fashion-MNIST rows (66/66) were genuine.
3. **What the real runs actually showed.** All 33 genuine runs for that dataset
   plateaued far below the two retained benchmarks, and several collapsed
   outright.

**Guardrail.** Before any third benchmark is added to this table or to any
figure, each cell must trace to a `results.json` under `outputs/` whose final
round matches the reported value, and the sweep CSV must be regenerated rather
than reused. Points 1 and 2 above are the two specific ways that check was
bypassed before.

## Known open inconsistencies (not addressed by the table)

- Hyperparameter table claims T_total = 100 rounds and κ = 0.3; actual runs:
  30 rounds (defense comparison), 50 rounds (sweeps), full participation
  (`fraction-train: 1`).
- Resolved 2026-08-12: the third-dataset curves that remained in two figures
  have been removed and the figures regenerated from the two evaluated
  benchmarks only; the manuscript no longer references a third dataset anywhere.
