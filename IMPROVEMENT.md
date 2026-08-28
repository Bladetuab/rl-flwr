# USTAD-FL v2 — Redesign Toward an IEEE Top-Tier Submission

Revision date: 2026-08-25
Scope: **keep** the two core ideas — (i) malicious-behavior detection by **temporal–spatial
behavior monitoring**, and (ii) **parameter reduction** on the monitored update — and rebuild the
method around them so that the limitations identified in review are resolved rather than
side-stepped.

Artefacts: `rl_flwr/CustomFedAvg_adaptive_no_valdelta.py`, `CustomFedAvg_adaptive_new.py`,
`CustomFedAvg_adaptive_assumption_free.py`, `strategy.py`, `convlstm_detector.py`,
`baseline_defenses.py`, `server_app.py`, manuscript `convlstm_ano/convlstm_ano/fl_sec.tex`,
and the logged per-client scores in `outputs/2026-08-1*/…/stat_anomaly.csv`.

> **This revision reverses the previous recommendation.** The earlier version of this document
> advised deleting the temporal arm and reporting it as a negative result. The measurements in
> §1 show that the negative result was an artefact of the *fusion rule*, not of the temporal
> idea — the arm has never had a fair test. Keeping it is the better-founded call.

---

## 1. What the logged data actually says

Everything in this section is measured from existing runs (the six `ustad_af … tau0.40` cells,
rounds ≥ 6, i.e. attacked rounds only). No re-run was needed — `_append_stat_csv` already logs
`stat_ewma`, `convlstm_rel`, `fused_score` and `is_malicious` per client per round.

### 1.1 The temporal arm is informative, not noise

| Cell | AUC `stat_ewma` | AUC `convlstm_rel` | AUC `fused` (shipped) |
|---|---:|---:|---:|
| MNIST / Gaussian | 1.0000 | 0.9797 | 0.9993 |
| MNIST / LIE | 1.0000 | 0.9801 | 0.9993 |
| MNIST / Min-Max | 1.0000 | 0.9700 | 0.9972 |
| Fashion / Gaussian | 1.0000 | 0.9228 | 0.9974 |
| Fashion / LIE | 1.0000 | 0.9695 | 0.9990 |
| Fashion / Min-Max | 1.0000 | 0.9233 | 0.9929 |

Mean `convlstm_rel` is **0.73–0.87 for malicious** against **0.22–0.25 for benign** in every
cell. The temporal arm ranks attackers above honest clients reliably. It is a working detector.

### 1.2 Its scale drifts across rounds, and `max` fusion imports the damage

The arm separates well *within* a round and badly *across* rounds. Taking the cut that catches
every attacker and measuring how many benign clients also fire:

| Cell | per-round **median** | per-round mean | rounds with **0 %** |
|---|---:|---:|---:|
| MNIST / Gaussian | 0.0 % | 10.7 % | 38 / 45 |
| MNIST / LIE | 0.0 % | 4.5 % | 42 / 45 |
| MNIST / Min-Max | 0.0 % | 5.5 % | 42 / 45 |
| Fashion / Gaussian | 0.0 % | 25.3 % | 29 / 45 |
| Fashion / LIE | 0.0 % | 5.7 % | 42 / 45 |
| Fashion / Min-Max | 0.0 % | 16.3 % | 36 / 45 |

In 29–42 of 45 attacked rounds the temporal arm separates attackers from honest clients
**perfectly**, with zero benign false positives. It degrades in a minority of rounds.

The failure is therefore one of **scale drift, not of discrimination**: the absolute level of
`convlstm_rel` moves from round to round, so no single fixed threshold works across the campaign
even though a per-round threshold usually would. (Pooling all rounds and taking one global
malicious minimum reports 95–99 % benign firing, but that number is a pooling artefact — one
low-scoring malicious client-round sets the floor for the whole pool. The per-round figures above
are the honest measurement.)

`fused = max(stat_ewma, convlstm_rel)` (`CustomFedAvg_adaptive_new.py:420`) then makes that drift
decisive: it compares an EWMA in [0,1] against a `tanh`'d z-score — **different units, neither
re-referenced per round** — and takes the larger.

So on the rounds where the temporal scale has drifted up, every benign client is promoted over
threshold regardless of its statistical score. Partitioning the clients that `fused` pushes above
`τ_abs = 0.40` when `stat_ewma` alone would not:

| Cell | crossover client-rounds | **fraction benign** |
|---|---:|---:|
| MNIST / Gaussian | 67 | **100.0 %** |
| MNIST / LIE | 63 | **100.0 %** |
| MNIST / Min-Max | 103 | 93.2 % |
| Fashion / Gaussian | 77 | **100.0 %** |
| Fashion / LIE | 63 | **100.0 %** |
| Fashion / Min-Max | 118 | 95.8 % |

Every honest client that the shipped system wrongly excluded was excluded by this mechanism:
the temporal arm flags legitimately divergent non-IID benign clients, and `max` lets that
override their low statistical score. This is the true cause of the honest-client exclusions the
manuscript reports at `fl_sec.tex:635`, and of the margin inversion at `fl_sec.tex:490`.

### 1.3 Changing the fusion rule recovers the ranking — but not the threshold

Counterfactual on the same logged scores — no retraining, no re-run:

| Cell | AUC `max` (shipped) | AUC per-round z-cal + `max` | AUC weighted sum 0.8/0.2 |
|---|---:|---:|---:|
| MNIST / Gaussian | 0.9993 | 1.0000 | **1.0000** |
| MNIST / LIE | 0.9993 | 1.0000 | **1.0000** |
| MNIST / Min-Max | 0.9972 | 0.9998 | **1.0000** |
| Fashion / Gaussian | 0.9974 | 0.9999 | **1.0000** |
| Fashion / LIE | 0.9990 | 1.0000 | **1.0000** |
| Fashion / Min-Max | 0.9929 | 0.9994 | **1.0000** |

A weighted sum restores perfect **ranking** in all six cells while retaining the temporal arm,
and improves the single-cut margin about fivefold (−0.465…−0.650 → −0.132…+0.022).

**Be precise about what this does and does not fix.** AUC 1.0000 with a still-negative margin
means the scores are perfectly *rankable* but **no single fixed threshold separates them**. Your
filter cuts on a threshold, not on a ranking. So changing the fusion rule alone does *not* recover
the property the gap filter actually needs — `stat_ewma` alone still has it (+0.053…+0.183) and
the fused score, however combined, still does not in five of six cells.

What §1.3 does establish is that **the temporal arm is not noise and the shipped fusion rule is a
real, quantified defect** costing roughly a fivefold margin degradation. Recovering a positive
fused margin needs the per-round calibration of §3.3 together with the training fixes of §2 — not
a change of fusion weights.

**Conclusion.** The temporal–spatial idea was never the problem. A fusion defect and four
training defects were. The published "the temporal arm contributes nothing" finding is not sound
and must not be reported.

### 1.4 What this does *not* establish

Weighted-sum margins remain slightly negative in five of six cells, whereas `stat_ewma` alone is
positively separable (+0.053…+0.183). So on **these three attacks** the temporal arm still adds
nothing: the cross-sectional screen is already at ceiling and there is no headroom to occupy.

That fact sets the entire research agenda below. **The temporal arm must be justified on threat
models where cross-sectional detection fails by construction** (§4). Retaining it purely to
match a perfect screen is not defensible; retaining it because it sees an attack class the screen
is blind to is a contribution.

---

## 2. Four verified implementation defects in the temporal arm

These are separate from the fusion bug and all are load-bearing.

### D1. The training gate reads the server's test set — and the paper says it doesn't

`_maybe_update_detector` (`strategy.py:2845`) trains only when `round_is_good`, and
`round_is_good = self._round_is_good(acc_val, loss_val)` at `strategy.py:4018` is fed from
`evaluate_fn(current_round, arrays)` — the server's **global test-set evaluation**.

The manuscript claims, three times, that the method uses "no server-side data of any kind"
(`fl_sec.tex:278`, `394`, `642`). That is **factually wrong as implemented**, and it is worse
than a mere dependency: it is test-set information steering a training decision, which is a
methodological leak a reviewer with the code will find immediately.

*Fix — and it is elegant.* Replace the gate with a signal you already compute and which needs no
data: **train when the gap branch abstained**, i.e. when `τ_t^gap = +∞` and the round's score
distribution is unimodal. That is the method's own, server-data-free definition of "this round
looks quiet." It restores the no-server-data claim and removes the leak in one change.

### D2. The detector is trained on its own labels

`_collect_clean_windows_for_training` (`strategy.py:3520-3553`) selects windows from clients
whose `last_anomaly_scores` fall below `good_client_quantile = 0.5` — the detector is trained on
whoever **the detector itself** just called benign. Any attacker it accepts becomes training data
reinforcing that acceptance: a textbook confirmation loop, and a standing invitation to a
slow-poisoning adaptive adversary.

*Fix.* Never train on self-selected clients. Two defensible sources instead:
- the **robust consensus** — the coordinate-wise median trajectory of the round, which a minority
  cannot control by the bound already in the threat model; or
- clients selected by the **statistical screen**, which is a different, non-self-referential
  signal (acceptable, but weaker — it still couples the two arms).

Prefer the first. It also motivates the architectural change in §3.2.

### D3. Training is disabled exactly when the attack is loudest

`skip_update = (drop_rate > 0.4)` at `strategy.py:4019`. A high drop rate means the filter is
working hard, i.e. an attack is in progress — precisely the regime the detector most needs to
learn from and the one in which it is frozen.

*Fix.* Remove it. With D1's gate (abstention ⇒ quiet round) the condition is redundant: a loud
round already fails the gate.

### D4. Training is starved

`update_every = 2`, `max_steps_per_update = 50`, `batch_size = 8`,
`buffer_min_windows = 64`, `detector_lr = 1e-4`, gated on a `K_good = 2` streak. Over a 50-round
run that is on the order of 10³ SGD steps total for a ConvLSTM — undertrained by any standard.

*Fix.* Pre-train on the warm-up window to convergence, then maintain with the D1 cadence.
Report the training loss curve; it is currently unreported and unverifiable.

---

## 3. The redesign

### 3.1 Fix the "spatial" axis — it currently means nothing

This is the deepest design flaw and the one that most limits the idea you want to keep.

**What happens now.** `feature_H = feature_W = 16` (`strategy.py:2540-2541`) gives a 256-cell
grid, but the latent dimension is `K = min(HW, n_keep) = 60`, and the code zero-pads 60 → 256
(`strategy.py:3421-3422`). So:

- **77 % of every frame (196 of 256 cells) is structurally zero.** The 3×3 convolutions spend
  most of their receptive field on padding.
- The 60 live cells are **SVD components in row-major order**. Adjacent cells are adjacent
  singular values. There is no spatial relationship between neighbours, so a convolution — whose
  entire inductive bias is local correlation — is sliding over an arbitrary permutation.

A reviewer who asks "why a *convolutional* LSTM over this layout?" currently has no answer.

**Fix A (mandatory, easy): make the grid exact.** Choose `K = 64` on an **8×8** grid and all
padding disappears. This is immediately attainable: with `N = 100` reporting every round, two
warm-up rounds give `n_b = 200`, so at `ρ = 0.40`, `n_keep = 120 ≥ 64`. Raise
`svd_bootstrap_min_samples` accordingly and drop the pad branch.

**Fix B (the real contribution): give the spatial axis meaning.** Instead of laying out singular
components row-major, allocate **one contiguous tile per model layer** — a per-layer block
layout. Then the spatial axis *is model architecture*, neighbouring cells genuinely belong to the
same layer, and a convolution has a defensible inductive bias: it detects poisoning localized to
particular layers. This is a genuine idea, and it connects directly to the layer-based result of
FedRoLA (KDD 2024), which reports that DNN layers contribute unequally to poisoning detection.

Fix B also removes the arbitrariness of the current fully-connected-head-only layer selection
(`_get_selected_layer_names`, `D = 66,954`), which is unexamined in the paper.

### 3.2 Reference the temporal score to the population, not to the client alone

Currently the score is a **one-step forecast error of a client's own trajectory**
(`ConvLSTMForecaster.anomaly_score`), wrapped by a two-sided relative z-score
(`CustomFedAvg_adaptive_new.py:324-325`) so that anomalously *low* error also scores high. The
wrapper is why LIE is caught despite emitting a constant vector, and it is why the arm is
mis-calibrated: any client in **either** tail of the EWMA distribution fires, and under non-IID
data many honest clients live in the tails (§1.2).

**Redesign: forecast the population, score the individual.** Fit the ConvLSTM to predict the
*median* trajectory of the round — a target a minority provably cannot displace far, by the
`0.5/(1−ε)` bound already in the threat model. A client's temporal score is then the deviation of
its own trajectory from the predicted population dynamics.

Three things follow, and they are the reasons to prefer it:

1. **The confirmation loop of D2 disappears by construction** — the training target is the robust
   consensus, not a self-selected client set.
2. **The score becomes one-sided and interpretable** — "how far does this client's evolution
   depart from where the honest population is going" — which removes the two-tail false-positive
   mechanism that §1.2 measured.
3. **It is the right test for the threat models of §4** — a slowly drifting attacker is precisely
   a client whose *trajectory* diverges from the population's while each individual round looks
   ordinary.

**Score persistence, not magnitude — this is the part that must be got right.** Under Dirichlet
`α = 0.5` honest clients legitimately deviate from the population median, and the most
non-IID-skewed honest clients deviate *most*. Scoring raw deviation magnitude would therefore
recreate the very false-positive mechanism §1.2 measured, by a different route: the top-deviation
clients would be the most label-skewed honest ones, not the attackers.

It would also contradict §4.1, whose flagship attack is *defined* by staying inside the benign
cluster every round — per-round deviation magnitude is exactly the thing it does not exhibit.

The discriminative quantity is the **temporal signature** of the deviation, not its size:

- an honest skewed client deviates in a direction fixed by its local label distribution, and its
  deviation **shrinks** as the global model converges toward a solution that covers its classes;
- a drifting attacker's deviation **persists or grows**, and holds a consistent direction that no
  amount of global convergence explains.

So the scored quantity should be the persistence, trend, and directional autocorrelation of a
client's deviation from the predicted consensus — not `‖deviation‖`. This is also what makes the
`window_len` increase below necessary rather than cosmetic: a trend is not estimable from five
points.

Also lengthen the temporal context. `window_len = 5` with `history_max = 10`
(`strategy.py:2543`, `2547`) is very short for detecting slow drift; the attacks in §4 need a
window on the order of 10–20 rounds.

### 3.3 Fuse on a common scale, with the arms in defined roles

Replace `max` (`CustomFedAvg_adaptive_new.py:420`). Per §1.3, a weighted sum already restores
AUC 1.0000 everywhere. Specify the fusion properly:

1. **Calibrate each arm per round** to a common scale before combining — per-round rank, or a
   robust z against the lower-half MAD you already compute in `_benign_scale`. Never combine raw
   scores in different units.
2. **Assign roles rather than treating the arms as peers.** The cross-sectional screen is the
   primary detector (it is provably at ceiling on overt attacks and costs almost nothing). The
   temporal arm's job is (a) attacks the screen is blind to by construction (§4) and (b)
   resolving the marginal band near the threshold. A gated or weighted combination expresses
   that; `max` does not.
3. **Report the fusion weight as a swept hyperparameter**, not a constant. §1.3 used 0.8/0.2 as
   an illustration, not a recommendation.

### 3.4 The parameter-reduction arm needs to earn its claim

Keeping "reduce the number of parameters monitored" as a contribution requires two things the
paper does not currently have.

**(a) State the compression as a measured result.** `D = 66,954 → K = 64` is a **1046×**
reduction. That is a headline number and it is never reported as one. Report the
detection-quality-versus-`K` curve so the choice is empirical rather than incidental — at present
`K = 60` is not chosen at all, it is whatever `n_keep` happened to be.

**(b) Show SVD beats cheaper reductions.** This ablation is mandatory and currently missing.
Compare, at matched `K`, against:
- **random projection** (Johnson–Lindenstrauss) — no fitting, no warm-up, no basis to poison;
- **top-`k` magnitude sparsification**;
- **count-sketch / hashing**.

If a random projection matches SVD, the SVD machinery — and the warm-up assumption it drags in —
is unjustified, and the honest paper says so and uses the random projection. If SVD wins,
you have a result. Either outcome is publishable; not running it is not.

**(c) Refresh the basis.** The basis is fitted once at `t = 1` and frozen
(`fl_sec.tex:301`). The benign update distribution drifts substantially as training converges, so
by round 50 a round-1 basis is stale. Add a periodic robust re-fit on the current trusted set, or
an incremental SVD. Keep the median-centred, `ρ`-trimmed robustness.

**(d) A second reading of "reduce updated parameters," worth one paragraph.** The phrase can also
mean reducing what clients *transmit* — sparsified or low-rank client updates, as in
communication-efficient FL. That interacts with security in a way worth stating: fewer
transmitted coordinates means fewer places for an attacker to hide a payload, but also less
evidence for the detector. If the paper is positioned for a communications venue this becomes a
first-class contribution; for TIFS/TDSC one honest paragraph suffices.

### 3.5 Keep the gap-based exclusion rule unchanged

The abstention rule of `fl_sec.tex:376-394` is the strongest part of the current method and needs
no redesign — it is the piece that lets the filter decline to cut. Two notes only:

- The exclusion argument now rests on a *correct* score. §1.2 showed the honest-client exclusions
  came from fusion, so once §3.3 lands those numbers should improve markedly.
- `τ_abs = 0.40` remains a scale-dependent constant, conceded at `fl_sec.tex:394`. Derive it from
  the round's own robust scale so it is scale-free like the gap branch, or demonstrate transfer
  on a third dataset (§5, P1-5).

---

## 4. What the temporal arm must be tested on — the core of the new paper

§1.4 established there is no headroom on Gaussian / LIE / Min-Max. So the temporal–spatial arm is
justified only by attacks that a per-round cross-sectional screen **cannot** see. These must be
built; they are the paper's new experimental core, and they are also the answer to the missing
adaptive-attack objection.

1. **Time-distributed drift.** The attacker stays inside the benign cluster *every round* — norm
   matched to `ν`, cosine within the benign range — but moves consistently in one direction across
   rounds. Any purely per-round screen is blind by construction; only a temporal model
   accumulates the evidence. **This is the flagship experiment.**
2. **Intermittent / on-off attackers.** Attack every `k`-th round. Cross-sectional detection
   catches only the active rounds; a temporal model detects the *pattern* and can attribute the
   quiet rounds to the same client.
3. **The detector-aware adaptive attacker.** The manuscript hands the reviewer this recipe at
   `fl_sec.tex:456`: published weights, none exceeding 0.30 against `τ_abs = 0.40`, and a clone
   score that is a hinge at `τ_c = 0.95` — so de-correlating just below 0.95 zeroes it. Build it:
   per-client jitter pushing pairwise cosine under `τ_c`, norms matched to `ν`, and a
   budget-constrained joint evasion of the weighted sum. If it defeats the statistical screen and
   the temporal arm catches it, that single result justifies the entire architecture.
   **The de-correlation mechanism already exists** — `strategy.decorrelate_cos`, set from
   `DECORRELATE_COS` at `server_app.py:865` and applied identically to every defense in the
   comparison. Sweep it rather than rebuilding it.
4. **Backdoor / targeted attacks.** Weak per-round signature, but must persist across many rounds
   to implant — structurally a temporal signal. This also closes the scope gap at `fl_sec.tex:92`.
5. **Slow-ramp sybils.** Coordinated clients whose collusion strengthens gradually, defeating any
   fixed-threshold per-round test.

**Framing.** The contribution is not "temporal detection is more accurate." It is: *cross-sectional
screening is provably sufficient for overt attacks and provably blind to time-distributed ones;
temporal–spatial behavior monitoring on a compressed representation closes exactly that gap, at
bounded cost.* That is a defensible top-tier claim, and §1 is the evidence that the mechanism
works once fused correctly.

---

## 5. Carried forward from the previous review — still required

These are orthogonal to the redesign and none is superseded by it.

| Item | Why it still blocks |
|---|---|
| **Multi-seed** | Seed hardcoded 42 (`task.py:255,268`; `server_app.py:589`), no config key. Every cell is one run. "We tie with DnC" is inadmissible without a variance estimate; the binomial floor quoted at `fl_sec.tex:569` covers test-set sampling only. Plumb a seed, run ≥5, use a paired test across the six cells. |
| **Attack-free 50-round run** | Rounds 1–5 sit inside the `T_cs = 8` cold-start window, forcing the manuscript to hedge its own headline number as "an upper bound" (`fl_sec.tex:629`). A genuine `no_attack` run deletes the hedge. Cheapest remaining item. |
| **CIFAR-10, run fresh** | Two grayscale 28×28 benchmarks is below bar. Do **not** reuse `cifar10_results_report.md` — mixed 30/50/100-round horizons, earlier quantile detector, best-of-tau selected. Encouraging: Gaussian 0.6203 there sits above the no-attack 0.6170. |
| **A regime where exclusions cost accuracy** | At `N = 100` full participation, accuracy is structurally blind to honest-client loss (`fl_sec.tex:633`), so "we win on exclusions" reads as a self-chosen metric. Re-run at `N = 20`, or `κ < 1`, or Dirichlet `α = 0.1`, where a dropped client is not recoverable. If DnC's fixed count costs accuracy there and yours does not, the exclusion argument becomes a primary result. |
| **`ε` sweep** | Everything is at exactly 40 %. Sweep `ε ∈ {0, 0.1, …, 0.45}` and plot the measured breakdown against the `0.5/(1−ε)` bound — theory and experiment on one axis. |
| **Hyperparameter sensitivity** | Seven asserted constants: `β` weights, `τ_c = 0.95`, `τ_abs = 0.40`, `γ = 1.0`, `ρ = 0.40`, `λ_r = 0.4`, `S_min = 2`. "Set a priori" (`fl_sec.tex:364`) is the right posture but is not the same as insensitive. |
| **Missing baselines** | Add **coordinate-wise median** and **trimmed mean** (`yin2018byzantine`, already cited, ~20 lines each) — the paper says AGG is plain FedAvg so all robustness is the filter's, which invites exactly this comparison. |
| **`SignGuardDefense` is implemented and unreported** | `baseline_defenses.py:129`, fully wired. It needs no adversary count either, so it is immune to your central argument. If you release the code a reviewer finds it. Include it and argue on the right axis. |
| **Promote the median-displacement analysis** | `fl_sec.tex:200` contains genuine theory buried in prose: the `0.5/(1−ε)` bound, the `[0.183, 0.817]` capture band, and LIE's calibration sitting at the band edge **as an algebraic identity**. Make it a numbered Proposition. |

**On the latent-space gap in that theory:** the paper concedes it is a parameter-space property
not shown to transfer to the latent median. Be realistic — the coordinate-wise median is *not*
equivariant under a general linear map, and `B` is a dense SVD basis, so the transfer almost
certainly does not hold as a theorem. Do not attempt that proof. **Measure it instead** (one extra
column in `_append_stat_csv` logging the malicious vector's empirical latent quantile), scope the
proposition to parameter space, and present the measurement as corroboration.

### Novelty check to run before anything else

**PRODIGY** — Ergisi, Maßny & Bitar, *"PRODIGY: Proximity- and Dissimilarity-Based
Byzantine-Robust Federated Learning,"* IEEE FLTA 2025 (arXiv 2509.09534) — describes a dual
scoring system that *"promotes natural similarity among honest clients while detecting suspicious
uniformity as a potential indicator of an attack."* That is, in substance, your clone score
`r^(L)`, with the same non-IID motivation. It is not cited. Read it and position against it
explicitly; discovering this in review is much worse. Two more for currency:
**FedRoLA** (Yan et al., KDD 2024 — layer-based, no historical/validation data, and unusually it
*reports false positives*: 6.4 % FPR / 92.8 % TPR — the best baseline candidate, and the direct
inspiration for §3.1 Fix B) and **FedDet** (Yang, Gu & He, IEEE IoT J. 2024 — unified
reformulation of adaptive model-poisoning with perturbation upper bounds; relevant to §4.3 and to
the theory). For the related-work table only: **Sophon** (IEEE TDSC 2025) and **FLTG** (2025),
both of which require server-side data and so strengthen your Section II-C argument at no cost.

### Also carried forward

- **Secure aggregation.** USTAD-FL needs individual updates in the clear, so it is incompatible
  with secure aggregation. Unmentioned in the paper. State it as a limitation; two sentences on
  whether the signals could be computed on masked or sketched updates is enough. Same for
  client-side DP noise, which perturbs exactly `r^(N)` and `r^(L)`.
- **Repository hygiene before release.** `backup.py` (3,542 lines), `strategy.py` (4,046), nine
  `fl_sec.pre_*.tex` snapshots, ~46 output dirs. Also: `run_config.json` records
  `attack-mode=gaussian, tau-s-quantile=0.8` for *every* run — the sweep runs all combinations
  inside one `flwr` run — so anything trusting it mislabels 100 % of its points. The run-directory
  name is the authoritative source; document that or fix the writer.

---

## 6. Execution order

Stages 1–3 are code changes validated on **existing logs or a handful of runs**; they are cheap
and they gate everything else.

| # | Work | Runs | Gate |
|---|---|---:|---|
| 0 | Read PRODIGY (§5). No compute; may change framing. | 0 | is the clone score still novel? |
| 1 | **Fusion fix** (§3.3): calibrate, replace `max`. ✅ *implemented* | 6 | **positive single-cut margin**, not just AUC 1.0 — see §1.3 |
| 2 | **Grid fix** (§3.1 Fix A): `K=64`, 8×8, delete padding. | 6 | no accuracy/AUC regression; zero pad cells |
| 3 | **Training fixes** (D1–D4): abstention gate, consensus target, drop `skip_update`, pre-train. ✅ *implemented* | 6 | detector loss converges; no server-eval dependency remains |
| 4 | Seed plumbing + attack-free run + multi-seed main table. | ~100 | exclusions → ~0 post-cold-start |
| 5 | **Layer-block spatial layout** (§3.1 Fix B). | ~12 | does layer structure beat row-major? |
| 6 | **Population-referenced temporal score** (§3.2). | ~12 | is the benign tail gone? |
| 7 | **Build the §4 attacks** — drift, intermittent, adaptive, backdoor. | ~60 | does the cross-sectional screen fail? does the temporal arm catch it? |
| 8 | Projection ablation (§3.4b): SVD vs. random projection vs. top-`k`. | ~40 | is SVD justified at all? |
| 9 | CIFAR-10; exclusion-sensitive regime; `ε` sweep; sensitivity; baselines. | ~250 | — |
| 10 | Theory (§5) + full rewrite. | 0 | — |

Stage 7 is the one that decides whether this is a top-tier paper. Stages 1–3 are what make
stage 7 meaningful — running it on the current fused score would measure the bug, not the idea.

### Stages 1 and 3 are implemented

`rl_flwr/CustomFedAvg_adaptive_calibrated.py` — subclasses
`CustomFedAvg_adaptive_no_valdelta`, changing only the fusion rule and the detector's training
regime. Registry entries in `server_app.py`:

| entry | configuration |
|---|---|
| `ustad_cal` | the fix — calibrated boost fusion, `w = 0.2` |
| `ustad_cal_raw` | boost fusion without per-round calibration |
| `ustad_cal_wsum` | weighted sum instead of boost (§1.3's form) |
| `ustad_cal_maxfuse` | the shipped `max` rule, for reproducing the defect |
| `ustad_cal_statonly` | `temporal_weight = 0` — statistical screen alone, the §1.4 reference |

Fusion is `stat + w·rel·(1 − stat)` rather than a weighted sum, because the boost form can never
lower the primary score and therefore cannot *rescue* a client the statistical screen would drop.
Measured offline on the six logged runs (single-cut margin, higher is better):

| fusion form | cells with margin > 0 | worst-cell margin |
|---|---:|---:|
| `max` (shipped) | 0 / 6 | −0.650 |
| weighted sum, raw | 1 / 6 | −0.132 |
| **boost, raw** | **4 / 6** | **−0.066** |
| **boost, calibrated** (default) | **4 / 6** | −0.084 |

Training fixes: `_round_is_good` now ignores the server accuracy/loss it is handed and returns
the unsupervised quiet-round flag (D1); `_collect_clean_windows_for_training` selects by distance
to the coordinate-wise median latent frame rather than by the detector's own scores (D2);
`skip_update` is ignored (D3); a `pretrain_steps` burst plus a steadier cadence replaces the
starved schedule, with every episode written to `detector_training.csv` (D4).

Validated by `test_calibrated_fusion.py` (7 groups, all passing) — six synthetic-input groups for
the decision rules plus one regression that replays the six real runs through the actual
implementation and reproduces the table above. **Not yet validated on a live run**: the training
fixes change what the detector learns and cannot be checked offline at all.

**Add one gate criterion to stages 1 and 3: `detector_training.csv` must be non-empty.** The D1
quiet gate can plausibly never open — the adversary is active from round 6, so almost no round is
unimodal, and windows only start existing at round 5 once `window_len = 5` frames of history have
accumulated. If the buffer threshold is not reachable within the warm-up rounds the detector never
trains and `ustad_cal` silently runs a *randomly initialised* ConvLSTM while the run completes
normally. `buffer_min_windows` is lowered from the inherited 64 to 32 for exactly this reason, and
every blocked training attempt is now logged. If the CSV still comes back empty, the fix is to
make the warm-up rounds supply more windows — not to loosen `quiet_gap_mult`, which would let the
detector train on attacked rounds and reintroduce D2 by another route.

---

## 7. Where this leaves the paper's story

**Before:** a five-signal cross-sectional screen that ties with DnC on accuracy and wins on a
self-introduced exclusion metric, with a deleted temporal component reported as a negative result
that §1 shows was never a fair test.

**After:** a two-arm behavioral detector on a compressed representation —

- a **cross-sectional screen** that is provably at ceiling on overt attacks and costs almost
  nothing (AUC 1.000, measured);
- a **temporal–spatial behavior model** over a layer-structured, 1046×-compressed latent, trained
  against the robust population consensus, which closes the class of *time-distributed* attacks
  the screen is blind to by construction;
- fused on a **common scale** with defined roles rather than an incommensurable `max`;
- and a **gap-based exclusion rule that can abstain**, so the number of exclusions is an output
  rather than a malicious-fraction prior.

That is a coherent architecture in which every component has a job no other component can do, and
each claim is measurable. It keeps both ideas you asked to keep, and it converts the three
weakest points of the current draft — the deleted arm, the missing adaptive attack, and the
self-chosen metric — into the three things the new experiments are designed to demonstrate.

---

## Appendix — reproducing §1

The measurements in §1 come from the six `ustad_af`/`tau0.40` run directories under
`outputs/2026-08-14/` and `outputs/2026-08-15/`, restricted to `round >= 6`. Columns used:
`stat_ewma`, `convlstm_rel`, `fused_score`, `is_malicious` from `stat_anomaly.csv`. AUC is the
rank-based Mann–Whitney statistic; margin is `min(malicious) − max(benign)`; the crossover set is
`fused_score > 0.40 & stat_ewma <= 0.40`.

Every number in §1.1–§1.3 is reproduced by **`analyze_fusion_defect.py`** at the repo root:

```
python analyze_fusion_defect.py            # defaults: --tau tau0.40 --abs 0.40
```

It reads only the already-logged `stat_anomaly.csv` files, so it requires no re-run and no
GPU. It is the evidence for the central design decision of this document (§3.3) and should be
kept under version control alongside it.
