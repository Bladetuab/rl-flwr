# §V (Experiments and Results) — verified provenance for the 2026-07-14 rewrite

Every number below was recomputed from the raw run directories in
`/home/tuan/get-local-model-bt-id/rl-flwr/outputs/`, not from any summary CSV.
(The June `sweep_summary_20260625_182323.csv` is the fabricated one; it is NOT used.)

## Scope decision (user, 2026-07-14) — FINAL: CIFAR-10 REMOVED

**CIFAR-10 is removed from the paper entirely.** The paper is now a two-benchmark study
(MNIST + Fashion-MNIST). All §V numbers below were recomputed on the 6 remaining
configurations; the 9-config numbers from the first pass are superseded.

Removed from the .tex: Fig. 1(c) CIFAR subfloat; the fabricated Fig. 5 CIFAR defense
panel; the CIFAR ASR/FDR subfloat; the `krizhevsky2009cifar` bib entry (now orphaned);
every CIFAR number and the "three benchmarks" framing. CIFAR-10 survives only as a
future-work item in the Conclusion.

**Figures still to regenerate WITHOUT CIFAR (user is doing this):**
1. `detection_tau06.png` — drop `"cifar10"` from `DATASETS` in `plot_detection_tau06.py`
   (line ~73), re-run. Becomes 2x2 instead of 2x3.
2. `sweep_accuracy_forwards_all_datasets_attacks.png` — drop cifar10 from the dataset
   list in `plot_sweep_accuracy_all.py`. NOTE: that script currently **crashes**
   (`KeyError: 'final_accuracy'`) because `plot_sweep_asr_fdr.collect()` was changed on
   07-13 and no longer returns that key; it needs fixing before it will run.

Until both are regenerated, the compiled PDF is internally inconsistent: the text says
two datasets, those two figures still show three.

### Why CIFAR was dropped rather than reported (kept for the record)
No USTAD-FL CIFAR run exists at the defense config (30 rounds / 5 epochs). The July
CIFAR runs are 50 rounds / **10** epochs and are therefore not comparable to the April
30r/5e baselines. At matched round 30 — and even with a 2x local-epoch advantage — the
July USTAD-FL CIFAR runs reach 0.428/0.395/0.403 (Gauss/LIE/Min-Max) against FLTrust's
0.489/0.485/0.469. **FLTrust beats USTAD-FL on CIFAR-10.** A referee who asks for
CIFAR will get an unfavourable answer; the honest options remain (a) run the 3 missing
USTAD-FL CIFAR runs (~45 min) and report the loss, or (b) leave CIFAR as future work,
which is the current choice.

## THE OPERATING POINT — the paper was wrong

`run_config.json` is **unreliable**: it echoes the static `pyproject.toml` defaults and
reports `attack-mode=gaussian` even for the LIE and Min-Max runs. It cannot be used to
recover q_s.

The true q_s was recovered empirically from `convlstm_cost.csv:ratio_scored`, which is a
direct function of q_s. Reference (MNIST/Gaussian sweep, rounds >= 11):

| q_s | 0.0 | 0.3 | 0.5 | **0.6** | 0.7 | **0.8** | 1.0 |
|---|---|---|---|---|---|---|---|
| ratio_scored | 0.764 | 0.727 | 0.550 | **0.460** | 0.370 | **0.280** | 0.105 |

**Every April defense-comparison run has `ratio_scored = 0.280` -> q_s = 0.8, exactly.**

=> Table IV and Figs. 2-4 were produced at **q_s = 0.8**, not the q_s = 0.6 that
Table II claimed. The sweep (§V-E) shows final accuracy is insensitive to q_s
(max spread 4.9 pts across all 9 configs), so §V-C's conclusions are unaffected;
q_s = 0.6 is *recommended* on precision grounds, and the detection study uses it.

## Two experimental configurations (both real, and different)

| | Defense comparison (§V-C) | q_s sweep + detection (§V-D, §V-E) |
|---|---|---|
| Runs | April 2026 | July 6-11 2026 |
| Rounds | 30 | 50 |
| Local epochs | 5 | **10** |
| q_s | 0.8 | swept 0.0-1.0 (0.6 highlighted) |
| Participation | full (fraction-train = 1) | full |
| Datasets | MNIST, F-MNIST | MNIST, F-MNIST, CIFAR-10 |

100 clients; 40 malicious, attacking from **round 6** (warm-up = 5 rounds).
Verified by `plot_detection_tau06.py`: "100 clients/round, 40 malicious from round 6".
Single seed per configuration — no error bars. This is a real limitation.

## Table II — 8 entries were WRONG (verified against code + runs)

| Parameter | Paper claimed | ACTUAL (source) |
|---|---|---|
| Client participation kappa | 0.3 | **1.0** (`fraction-train: 1`) |
| Total rounds T_total | 100 | **30 / 50** (see above) |
| Local epochs E | 5 | **5 (defense) / 10 (sweep)** |
| EWMA decay lambda | 0.9 | **0.6** (`anom_decay`, strategy.py:2199) |
| Re-evaluation quantile q_s | 0.6 | **0.8** for main results (recovered above) |
| Temporal window T | 8 | **5** (`window_len`, strategy.py:2222) |
| Spatial grid H x W | 4 x 4 | **16 x 16** (`feature_H/W`, strategy.py:2219) |
| Latent dim K = H*W | 16 | **256** (`svd_target_len`) |
| ConvLSTM hidden channels | 32 | **16** (`hidden_dim`, strategy.py:2230) |
| Filter quantile range | [0.4, 0.8] | **[0.35, 0.90]** (strategy.py:2312) |

Also corrected: `T_cs` 10 -> **8** (`score_all_first_n_rounds`), `R_max` 5 -> **10**
(`score_refresh_max`), `B_max` 50 -> **80** (`adaptive_cap_max`), `S_min` 10 -> **2**
(`min_clients`, strategy.py:2846, enforced at :2897).

The stale `B_max = 50` traces to `max_scores_per_round = 50`, which strategy.py:2260
explicitly marks as vestigial ("kept for CSV logging compatibility"); the live cap is
`adaptive_cap_max = 80`.

Confirmed correct as printed: N = 100; malicious fraction 0.4; Dirichlet alpha = 0.5
(`task.py:254`); eta = 0.01 (`server_lr`); warm-up 5; feature channels C = 6; ConvLSTM
kernel 3x3; `B_min` = 10 (`adaptive_cap_min`); `alpha_q` = 0.7 (`q_smooth`, and the
update at strategy.py:2735 matches the paper's Eq. exactly).

**Every entry of Table II is now traced to code or to the runs.** Nothing is asserted
that was not verified.

**NOTE:** T, H x W, K and hidden_dim also appear in the §III notation table / §IV method
text. Corrected there too, so §III and §V agree.

## §V-E — q_s sweep (genuine; 66 cells = 2 datasets x 3 attacks x 11 q_s)

ConvLSTM forward passes (mean over the 6 configs; near-identical across configs):

| q_s | 0.0 | 0.6 | 1.0 |
|---|---|---|---|
| forwards | 3,522 | 2,320 | 824 |
| reduction vs q_s=0 | — | **34.1%** | **76.6%** |

The paper's "approximately 34%" and "up to 77%" are **CORRECT** and survive unchanged.
(For reference, the defense comparison's q_s = 0.8 corresponds to a ~56% reduction.)

Final accuracy is flat in q_s in **all six** configs (the old "8 of 9, CIFAR-Gaussian is
a wild outlier" claim was an artifact of the fabricated CSV — it is gone):

| dataset | Gaussian | LIE | Min-Max |
|---|---|---|---|
| MNIST | [0.953, 0.960] | [0.932, 0.960] | [0.946, 0.959] |
| Fashion-MNIST | [0.800, 0.817] | [0.772, 0.817] | [0.796, 0.825] |

Max spread within any single config = **4.6 pts** (F-MNIST/LIE). Pooled per-dataset
envelopes: MNIST [0.932, 0.960]; F-MNIST [0.772, 0.825]. The paper distinguishes these
two quantities explicitly, because the pooled envelope (5.3 pts on F-MNIST) is larger
than the per-config spread and would otherwise appear to contradict it.

**Detection recall is q_s-invariant: min = 0.983, max = 1.000 over all 66 cells.**
Plotted "ASR" = FN/N is ~0.000-0.006. Throttling never lets attackers through.

**The savings are NOT free — they are paid in precision.** Benign rejection rate
FP/(FP+TN), q_s = 0.0 -> 1.0. Gaussian is the only clean, consistent trend
(both datasets); LIE/Min-Max are single-seed and bistable, so **no trend is
narrated into them** in the paper:

| config | q_s=0.0 | q_s=0.6 | q_s=1.0 |
|---|---|---|---|
| MNIST / Gaussian | 0.078 | 0.112 | **0.212** |
| F-MNIST / Gaussian | 0.080 | 0.107 | **0.225** |

Benign rejection at the recommended q_s = 0.6, all six cells: MNIST 0.112 / 0.100 /
**0.279** and F-MNIST 0.107 / 0.124 / 0.103 (Gauss / LIE / Min-Max). The paper states
"five of six at 0.100-0.124, the sixth (MNIST/Min-Max) elevated at 0.279" — it does
**not** quote the flattering [0.100, 0.124] range as if it bounded all six.

Mechanism (a real, identified code defect): an unscored client's ConvLSTM EWMA is
**decayed**, not held — `new_ewma = lam * prev`, lam = 0.6 (`strategy.py:~3470`).
Skipped clients drift geometrically toward zero, and the two-sided relative score's
low branch (`z_lower`, which exists to catch LIE) reads "not recently re-evaluated" as
"anomalously low" — i.e. as a LIE signature. Throttling therefore manufactures false
positives. **Reported as a limitation + future work (hold instead of decay).**
The defect is NOT fixed in the code as of this rewrite (strategy.py mtime 2026-07-07,
predating its discovery on 2026-07-13).

## §V-D — detection at the highlighted operating point q_s = 0.6

From `plot_detection_tau06.py` (reads per-run `client_states_rounds.csv`):

| dataset | attack | recall @ r6 | recall r7+ | missed rounds | FDR last-10 |
|---|---|---|---|---|---|
| MNIST | Gaussian | 1.000 | 1.000 | none | 0.010 |
| MNIST | LIE | 0.425 | 1.000 | none | 0.010 |
| MNIST | Min-Max | 0.350 | 1.000 | none | **0.146** |
| F-MNIST | Gaussian | 1.000 | 1.000 | none | 0.015 |
| F-MNIST | LIE | 0.525 | 1.000 | none | 0.037 |
| F-MNIST | Min-Max | 0.500 | 0.999 | [13] | 0.026 |

(CIFAR-10 rows removed with the descope. For the record, they were: Gaussian
1.000/1.000/0.020, LIE 1.000/1.000/**0.214**, Min-Max 0.475/1.000/0.016.)

- Recall = 1.000 from round 7 in 5/6 cells; F-MNIST/Min-Max is 0.999 (one client missed
  at round 13). Stated with the exception, not rounded away.
- Round 6 (attack onset) shows **lock-on latency** for LIE/Min-Max (0.350-0.525): a
  temporal detector has no prior malicious history to score against on the first
  attacked round. Gaussian already reaches 1.000 on both datasets.
- Rounds 1-5 have **no malicious clients**, so recall is undefined (0/0) and is not
  plotted. FDR ~0.46 there is the cold start, shaded in the figure.
- Tail FDR is low (1-4%) in **5 of 6** cells. The one exception — MNIST/Min-Max (0.146)
  — is named explicitly in the text, not hidden.

## Metric-naming caveat (a referee WILL flag this)

The sweep figures are labelled "ASR" and "FDR", but as plotted both are normalised by
**N = all 100 participants**:
- plotted "ASR" = FN/N. "Attack success rate" is a targeted/backdoor term of art; these
  are untargeted attacks. The meaningful quantity is recall = TP/(TP+FN).
- plotted "FDR" = FP/N is **neither** the statistical false-discovery rate FP/(FP+TP)
  **nor** the false-positive rate FP/(FP+TN).

The prose therefore (a) defines exactly what the figure axes plot, and (b) quotes the
properly normalised per-class rates (recall, benign rejection) whenever it makes a claim.
Relabelling the figures would be better still — flagged for the user.

## What is NOT claimed

- No claim that the ConvLSTM arm is *necessary*. Recall never drops even with ConvLSTM
  forwards cut 77%, and the statistical-only vs temporal-only vs fused ablation was
  never run. The text says "the fused detector attains perfect recall", never "the
  ConvLSTM is what catches the attackers".
- No claim of superiority on CIFAR-10.
- No multi-seed / error-bar claims. Single seed throughout.

---

## 2026-07-23 salvage reframe — new Table (tab:ablation_auc), §V-D "Ablation"

Added a decision-layer ablation subsection (§V-D, `\label{sec:ablation}`) and Table
`tab:ablation_auc`, answering reviewer C-001 (stat vs temporal vs fused) with **zero new
compute**. Provenance of every number:

- **Source:** `DECIDER_OFFLINE_FINDINGS.md` + `decider_results.csv`, produced by
  `rl-flwr/decider_offline.py` re-ranking the per-client scores logged in
  `stat_anomaly.csv`/`client_states_rounds.csv` across the 66 genuine MNIST+F-MNIST runs
  (`outputs/2026-07-0[6-9]|1[01]/*`). No retraining.
- **AUC table (rounds t>=6, mean over 11 q_s):** stat=1.000 all 6 cells; ConvLSTM
  0.922/0.726/0.663 (MNIST G/L/MM), 0.944/0.754/0.643 (F-MNIST G/L/MM); fused
  0.994/0.936/0.834, 0.996/0.921/0.832. Fused AUC < stat in 6/6 (mean drop 0.081).
- **Matched-budget (pooled 66 runs):** recall stat 0.9998 / fused 0.9924 / convlstm 0.8476;
  benign-rejection stat 0.1817 / fused 0.1867. Stat recall < fused in 0/66; fusion improves
  precision in 0/66.
- **Honesty caveat stated in-text:** this is a decision-layer re-ranking, NOT a from-scratch
  stat-only system. The from-scratch stat-only run (+ SignGuard/DnC) is the pending
  experiment (SALVAGE_SCOPE.md Part 1); the abstract carries an in-source TODO to that effect.
