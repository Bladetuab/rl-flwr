# Salvage scope — SignGuard/DnC comparison + honest reframe

**Date:** 2026-07-23 · **Goal:** determine whether a defensible Transactions paper exists around
the *statistical* detector (the ConvLSTM being dead by argument — see
`ATTACK_EXISTENCE_ARGUMENT.md`), and prepare the reframe.
**Two parts:** (1) the one decisive experiment; (2) the reframe, split into edits that are safe
now vs. edits that branch on the experiment's outcome.

---

# Part 1 — The decisive experiment: statistical arm vs. SignGuard vs. DnC

The offline decider already settled stat/temporal/fused (`DECIDER_OFFLINE_FINDINGS.md`), so this
is the single most informative run left. It answers: **does the assumption-free statistical
detector measurably beat the two nearest prior-art defenses the review mandates (M-001)?**

## 1.1 What to build (two new strategy classes)

Both follow the existing baseline pattern — a `FedAvg` subclass overriding aggregation, selected
by uncommenting a block in `server_backup.py` (same as `CustomKrum`, `FLTrustCustomFedAvg`,
`FedAvgWithLBFGSHVP`). Work on the per-client update deltas (global − client), as the current code
already does.

**SignGuard** (Xu et al., ICDCS 2022) — `SignGuardFedAvg(FedAvg)`, ~100 lines:
1. *Norm filter:* per-client ‖Δ‖₂; median M; keep clients with ‖Δ‖/M ∈ [0.1, 3.0].
2. *Sign filter:* random-sample ~10% of coordinates; per client form sign-statistics
   (fraction of +, −, 0); cluster with MeanShift (fallback k-means, k=2); keep the largest cluster.
3. Aggregate the **intersection** of the two benign sets (mean, optional norm-clip to M).
   No root data, no #malicious. **Confirm the [0.1,3.0] bound and the clustering choice against the
   paper — a strawmanned baseline is exactly the M-011 mistake made with FLDetector.**

**DnC** (Shejwalkar & Houmansadr, NDSS 2021 — the defense from the *same* paper as the Min-Max
attack) — `DnCFedAvg(FedAvg)`, ~80 lines:
1. Random-subsample b coordinates (b≈10k); center the update matrix.
2. Top right singular vector v via power iteration; outlier score sᵢ = ⟨Δᵢ − mean, v⟩².
3. Remove the c·m highest scores (c≈1.5; **m = malicious-count estimate — privileged, disclose it
   exactly as Multi-Krum's count is disclosed**), average the rest. Average over a few subsamples.

**USTAD-FL statistical-arm-only variant** (the actual contribution under test): add a flag that, in
`_run_convlstm_analysis`, skips steps 2–3 and sets `fused = ewma_stat` (i.e. `rel_s = 0`). This is
the "stat-only" system a reviewer will demand you report as *the method*, not an ablation.

## 1.2 The run matrix (decisive core)

| Axis | Values |
|---|---|
| Methods | **USTAD-stat-only, SignGuard, DnC** (decisive); plus re-run Multi-Krum/FLTrust/FLDetector for a complete, same-seed table |
| Datasets | MNIST, Fashion-MNIST |
| Attacks | Gaussian, LIE, Min-Max (all with the corrected LIE, `norm.ppf` — WP3) |
| Seeds | **≥3 (5 preferred)** — C-003; report mean ± std + a paired significance test |
| Fixed config | N=100, 40% malicious from round 6, Dirichlet α=0.5, **one horizon** (pick 50 and align the whole paper — resolves C-004's 30/50/100 tangle) |

Compute: the decisive core (3 methods × 2 × 3 × 3 seeds) ≈ **54 runs**; the full same-seed table
adds ~54 more. At ~10–15 min/run this is ~10–25 GPU-hours — one overnight batch.

## 1.3 What to measure (also closes M-003)

1. **Final accuracy** — mean ± std over seeds; paired test (stat-only vs SignGuard, vs DnC).
2. **Detection quality** — recall, benign-rejection, ROC-AUC of malicious ID per method that
   filters (stat-only, DnC, SignGuard's kept-set). Reuse `decider_offline.py`'s AUC/matched-budget
   machinery on the new `stat_anomaly.csv`/`client_states_rounds.csv`.
3. **Cost** — wall-clock/round: stat arm's O(N²) clone term vs DnC's SVD vs SignGuard's clustering.

## 1.4 The decision rule (this is the go/no-go for the paper)

**Calibrate to your own data first.** The decider found detection *saturated* (AUC 1.000, recall
0.9998) — because these three attacks are statistically trivial (identical vectors → clone/clustering
separate them; norm inflation → norm filter catches them). SignGuard and DnC exploit the *same*
blatant signatures, so they will hit the same ceiling. **A within-noise tie is the honest null and
the most probable outcome. Do not predefine GO as "beats by a margin" — on a shared ceiling that is
just noise-mining.**

The only realistic win is a **baseline failure mode**, with a nameable mechanism:
- SignGuard's sign-**clustering** is fragile under non-IID (α=0.5 can split benign clients into
  several clusters → false positives / dropped benign). USTAD's median-referenced signals have no
  clustering step.
- DnC's single top-singular-direction can miss a **multi-signal** attack. USTAD scores five axes.

So decide on *collapse*, not margin:
- **GO** — a baseline **collapses on ≥1 (attack, dataset) cell** (e.g. SignGuard's benign-rejection
  spikes under non-IID, or DnC misses an attack) while USTAD-stat-only **holds across all cells**.
  That is a real "robust across attack types with fewer assumptions" contribution with a mechanism.
- **TIE (expected)** — all three co-saturate within seed-noise. The detector is not a detection
  advance; the paper then rests entirely on §2.2-TIE cards (fewer assumptions + cost + negative
  result). Reassess appetite before writing.
- **LOSE** — USTAD-stat-only underperforms a baseline on any cell → no salvage; revisit the pivot.

**Run it regardless of the expected tie.** M-001 mandates the baselines' *presence*: a reviewer
rejects on a *missing* SignGuard/DnC, not on a tie. You are running this batch to clear the mandate
and to check for a collapse cell — not because a detection win is likely.

---

# Part 2 — The honest reframe

The draft is already honest where it counts (line 820 refuses to claim which arm detects), so this
demotes an already-hedged component. Removing ConvLSTM-dependence also makes the **"no clean root
dataset" claim true** (the statistical arm is purely per-round cross-sectional; the server set only
ever gated the ConvLSTM) — this *resolves C-002*, the second Critical issue, for free.

## 2.1 Safe now — independent of the experiment outcome

| Where (line) | Change |
|---|---|
| Title (18) | Drop "Spatio-Temporal". e.g. *"USTAD-FL: Assumption-Free Robust-Statistics Screening for Byzantine-Robust Federated Learning."* |
| Abstract (31) | Recast around the 5-signal statistical detector as **the** method; mention the ConvLSTM only as an explored, non-additive component (the negative result); keep the honest no-root-set / no-#malicious / no-attack-type claim and the detection metrics. |
| Contributions (54+) | (i) assumption-free, **aggregator-agnostic**, no-root-set statistical detector; (iii) **a rigorous negative result** — spatio-temporal/ConvLSTM modelling adds no detection value for untargeted attacks, with the *why* (harm⟺persistence coupling, `ATTACK_EXISTENCE_ARGUMENT.md`). Drop the "spatio-temporal framework" headline and the efficiency contribution (it scheduled a redundant arm). |
| §III-B (188) | Retitle "System and Communication Model" → **"System Model"**; de-wireless (BS→server, trim 6G/edge veneer) — M-006. |
| §III-F / P1 (284) | Remove "ConvLSTM is the tool for P1"; state P1 modestly as motivation. |
| §IV-C, §IV-E (343, 446) | Move "Spatio-Temporal Feature Construction" + "ConvLSTM-EWMA Scoring" out of the method spine into one **"Explored Extension: Why Temporal Modelling Does Not Help"** subsection carrying the negative-result argument + the decider evidence (throttle→0, recall invariant). |
| §V-E (822) | Reframe the q_s/efficiency subsection as **evidence of redundancy** (ConvLSTM cuttable to 0% at no recall cost), not an efficiency selling point. |
| Conclusion (842) | Align to the statistical detector + negative result. |

## 2.2 Branches on the experiment

- **GO:** statistical detector is *the* contribution; §V-C Table IV gains SignGuard/DnC columns
  (multi-seed) showing the measurable win; the negative result is the strong secondary contribution.
- **TIE (expected):** the paper no longer rests on detection at all. It rests on three cards, in
  descending strength: (1) **fewer assumptions** — no root set (now genuinely true), no #malicious,
  no attack-type — the strongest real card; (2) **lower cost** — but *only if the measurement
  favours you*: the stat arm carries an O(N²) clone term vs SignGuard's clustering / DnC's SVD, so
  this is not a given; (3) the **negative result**. Be clear-eyed: "ties SOTA + a negative result"
  is a *hard* Transactions sell — negative results are notoriously hard to place at TIFS/TDSC — so
  the realistic ceiling of the salvage path in the TIE world is a **mid/low-tier Transactions or a
  short/characterization paper**, framed as "a cheap assumption-free screen matches SOTA
  detect-and-filter defenses, and here is precisely when temporal modelling can and cannot help."

## 2.3 Carry over unchanged (already honest, already done)

Real Table IV (`results_table_provenance.md`), the detection-quality subsection, the precision-cost
finding, corrected Table II, the CIFAR descope. None of this is touched by the reframe.

---

## Sequencing

1. **Now, no compute:** all of Part 2 §2.1 (safe reframe) + build/unit-test the two strategy classes
   + the stat-only flag. Leaves the paper honest and the experiment turnkey.
2. **One overnight batch:** run Part 1's matrix.
3. **Decide** via §1.4, then finish Part 2 on the matching branch.
