# The DECIDER, run offline — does the ConvLSTM arm earn its place?

**Date:** 2026-07-23
**Compute used:** none (re-ranking of already-logged per-client scores; no training).
**Data:** the 66 genuine July MNIST + Fashion-MNIST runs
(`rl-flwr/outputs/2026-07-0[6-9]|1[01]/*`, 2 datasets × 3 attacks × 11 q_s),
100 clients × 50 rounds, 40 malicious from round 6.
**Script:** `rl-flwr/decider_offline.py`  ·  **Raw table:** `decider_results.csv`

## Why this could be done with no compute

`stat_anomaly.csv` logs, per client per round, both arms' scores and ground truth:
`round, client_id, stat_raw, stat_ewma, convlstm_rel, fused_score, is_malicious`,
and `client_states_rounds.csv` logs the detector's actual filter decision
(`predict_state`). The two arms are computed independently each round and fused by
`fused = max(stat_ewma, convlstm_rel)`. So the ablation the reviewer demanded (C-001)
— stat-only vs temporal-only vs fused — is recoverable by re-ranking the columns that
are already on disk, without retraining anything.

Two threshold-honest views, rounds ≥ 6 only:
- **(A) Threshold-free separation:** ROC-AUC of `is_malicious` vs each arm's score.
- **(B) Matched-budget operating point:** per round remove the *same* number of clients
  K_r the fused detector actually removed, but chosen by each arm alone → recall and
  benign-rejection. Isolates "given a fixed removal budget, which arm picks right?"

## Result — the statistical arm is a near-perfect detector; the ConvLSTM is dominated

### (A) ROC-AUC (mean over the 11 q_s), 1.0 = perfect ranking, 0.5 = none

| dataset | attack | stat-only | convlstm-only | fused |
|---|---|---|---|---|
| MNIST | Gaussian | **1.000** | 0.922 | 0.994 |
| MNIST | LIE | **1.000** | 0.726 | 0.936 |
| MNIST | Min-Max | **1.000** | 0.663 | 0.834 |
| F-MNIST | Gaussian | **1.000** | 0.944 | 0.996 |
| F-MNIST | LIE | **1.000** | 0.754 | 0.921 |
| F-MNIST | Min-Max | **1.000** | 0.643 | 0.832 |

`auc_stat = 1.000` in every config and is **invariant to q_s** (as it must be — q_s only
throttles the ConvLSTM), which independently validates the pipeline. Fusing the ConvLSTM
*lowers* AUC in all 6 configs (mean −0.081). Temporal-only never beats stat-only: **0 of 6**.

### (B) Matched-budget detection (mean over q_s): recall | benign-rejection

| dataset | attack | recall stat / cl / fused | ben-rej stat / cl / fused |
|---|---|---|---|
| MNIST | Gaussian | 1.000 / 0.906 / 1.000 | 0.070 / 0.133 / 0.070 |
| MNIST | LIE | 1.000 / 0.847 / 0.991 | 0.211 / 0.313 / 0.218 |
| MNIST | Min-Max | 1.000 / 0.783 / 0.986 | 0.270 / 0.414 / 0.279 |
| F-MNIST | Gaussian | 1.000 / 0.925 / 1.000 | 0.054 / 0.104 / 0.054 |
| F-MNIST | LIE | 1.000 / 0.839 / 0.992 | 0.196 / 0.303 / 0.201 |
| F-MNIST | Min-Max | 0.999 / 0.785 / 0.986 | 0.289 / 0.432 / 0.298 |

Pooled over all 66 runs: **stat recall 0.9998 vs fused 0.9924** (convlstm 0.8476);
**benign-rejection stat 0.1817 vs fused 0.1867** (convlstm 0.2832).
- Stat-only recall < fused recall in **0 / 66** runs (stat is *never* worse).
- Fusion improves precision (fused ben-rej < stat) in **0 / 66** runs (it only adds FPs).

## Interpretation — this decides which paper is writable

On the three evaluated attacks the ConvLSTM temporal arm **does not earn its place**. The
five statistical signals rank every malicious client above every benign one (LIE/Min-Max
emit identical vectors → clone score saturates; Gaussian inflates the norm), so the `max()`
fusion can only add false alarms on top of a perfect statistical ranking — and it does, in
every configuration, by two independent methodologies. This is exactly reviewer C-001, now
confirmed with the paper's own data.

**This takes down two of the four headline contributions, not one.** The efficiency
contribution ("~34% fewer ConvLSTM forward passes via adaptive budgeting") is a scheduler
for the ConvLSTM. If the ConvLSTM is redundant, the honest efficiency limit is "run it 0% of
the time" — you cannot sell efficiently *scheduling* a component you have just shown is
unnecessary. So Contribution (ii) [the dual detector] and Contribution (iii) [the compute
budgeting] both fall to this single result.

**The consequence for compute:** the affirmative "temporal modeling matters" thesis
**cannot be recovered by seeds or re-runs on Gaussian/LIE/Min-Max** — those attacks are
statistically saturated (AUC_stat = 1.000). It can be recovered *only* by exhibiting an
attack where the statistical signatures do **not** separate (AUC_stat < 1) and the temporal
arm does. That single experiment — not more seeds, not the LIE fix, not a re-run of the
existing suite — is the only compute worth spending to save the ConvLSTM.

## Limitations (stated, not hidden)

1. **Single seed per config.** These are the existing runs. Multi-seed would tighten the
   numbers but cannot overturn AUC_stat = 1.000 exactly (perfect ranking is saturation,
   not a noisy margin).
2. **Decision-layer ablation, not from-scratch retraining.** It re-ranks the per-arm scores
   each run already computed; it does not train a stat-only system from initialization.
   This is faithful because the statistical scores are computed identically regardless of
   the ConvLSTM, and detection is a per-round filtering decision — but a from-scratch
   stat-only run is the clean confirmation to add when compute is available (WP4).
3. **Existing attack suite only.** The finding is "the ConvLSTM is redundant *for these
   attacks*," which is precisely why the affirmative path is a *new* attack, not a re-run.
4. Absolute benign-rejection is high (0.05–0.29): a separate issue (the `anom_decay` decay
   defect + strict filter). The ablation claim is *relative* — fused ≥ stat on FP everywhere.

## So what — do not read this as a comfortable A/B menu

- **The "statistical-screen reframe" is NOT a safe floor — its viability is UNKNOWN until one
  experiment runs.** The review mandates a SignGuard baseline (M-001), and the carrying arm here
  *is* norm + sign + similarity + clone ≈ SignGuard + FoolsGold. A reasonable *prior* is that
  SignGuard will match USTAD-FL — in which case the reframed paper shows no measurable gain over a
  mandatory baseline (2 datasets, single-seed) and is a reject. But that is a prior, not a proof;
  **the SignGuard (+DnC) comparison is the experiment that decides**, and passing it is necessary
  but not sufficient. Because the offline decider already settled stat/temporal/fused, this
  comparison — not the ablation — is now the single most informative run to spend compute on.

- **The affirmative attack is effectively the only route to a Transactions paper — and its
  gating question is compute-free.** Before any harness or run, answer on paper: *can a
  statistically-invisible-but-temporally-visible attack even exist here?* The tension is real:
  an attack evades per-round statistics by looking benign **each round**, which usually means
  smooth / slow / temporally *consistent* — which is exactly what a next-frame temporal
  forecaster also cannot see. The "per-round-benign yet temporally-anomalous" region may be
  empty or tiny, especially for this small FC-head ConvLSTM. If you cannot construct that
  attack on paper and argue why the temporal arm would catch it while the statistics miss it,
  path B fails and **no compute should be spent on it.**

**Bottom line.** The evaluated-attack centerpiece (and the efficiency story with it) is
empirically dead. Whether *any* Transactions-grade paper exists here reduces to a single
compute-free thinking task: constructing — or proving the impossibility of — a
temporally-visible / statistically-invisible attack. That argument is the real decider, and
it can be done now.
