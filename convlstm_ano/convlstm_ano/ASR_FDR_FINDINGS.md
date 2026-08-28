# What the new ASR/FDR-vs-$q_s$ results actually say

Analysis of `sweep_asr_fdr_forwards_{mnist,fashion_mnist,cifar10}.png` and the runs
behind them (`rl-flwr/outputs/2026-07-06..11`; 100 clients x 50 rounds; 40 malicious
attacking from round 6, hence 1800/5000 = 36% of labels).

**Headline: do not write these up as "ASR ~ 0 everywhere, so the compute savings are
free." Read correctly they (a) contradict a claim already in the paper, and (b)
confirm the reviewer's Critical Issue #1 — while pointing at a specific, fixable
defect. That defect is the opportunity here.**

---

## 1. The reviewer already predicted this figure

Review C-001 (`reviews/20260706_180631_.../claude.md`):

> "The $q_s$ sweep (77% fewer ConvLSTM passes at unchanged accuracy) is itself
> evidence the temporal arm does little work."

The new figure makes that argument **stronger**: it shows detection *recall* is also
unchanged (ASR ~ 0 at every $q_s$, including $q_s = 1.0$, where ConvLSTM inference is
cut by 77%). A referee reads the flat ASR line as: *the headline detector can be
switched off without consequence.*

## 2. Recall is perfect and $q_s$-invariant; the statistical arm carries it

True per-class rates (not the plotted FN/N):

| MNIST-Gaussian | recall (TP/malicious) | TPs where stat arm was the max | TPs where ConvLSTM was the max |
|---|---|---|---|
| $q_s = 0.0$ | 1.000 | 32% | 68% |
| $q_s = 1.0$ | 1.000 | **73%** | 27% |

Recall never falls below **0.983** in any of the 99 (dataset x attack x $q_s$) cells;
for Gaussian it is exactly 1.000 everywhere (FN = 0). Fusion is
`max(stat, convlstm)`, so with the temporal arm starved the statistical arm still
catches every attacker.

## 3. The false positives track the ConvLSTM arm, and throttling multiplies them

Attributing each false positive to whichever arm produced the max
(`stat_anomaly.csv`: `fused = max(stat_ewma, convlstm_rel)`):

| MNIST-Gaussian | FP total | FP where stat was max | FP where ConvLSTM was max |
|---|---|---|---|
| $q_s = 0.0$ | 250 | 165 | 85 |
| $q_s = 1.0$ | **546** | **165** (unchanged) | **381** (4.5x) |

The statistical arm's false positives sit at a **constant 165** across the sweep;
the entire growth in FPs is on ConvLSTM-max client-rounds. Benign rejection rate
(FP / benign) rises 0.078 -> 0.212 (MNIST), 0.080 -> 0.225 (F-MNIST),
0.084 -> 0.288 (CIFAR-10) — consistent across all three datasets.

> *Caveat, deliberately:* this is a max-arm **attribution within the fused runs**. It
> does **not** establish that a stat-only system would have exactly 165 FPs — the
> filter threshold is a quantile of the *fused* score distribution, so removing an arm
> moves the threshold and reshuffles who gets cut. That counterfactual is exactly what
> the ablation in Step 2 measures, and why Step 2 is still necessary.

## 4. Root cause: skipping a client manufactures a LIE signature for it

Two mechanisms collide:

1. `strategy.py:3470` — a client that is **not** re-scored has its ConvLSTM EWMA
   **decayed**, not held: `new_ewma = lam * prev` (`lam = anom_decay = 0.6`,
   `strategy.py:3368`). Skipped clients drift geometrically toward zero.
2. `CustomFedAvg_adaptive_new.py:290-330` — `_compute_convlstm_relative_scores` is a
   **two-sided** MAD z-score. Its `z_lower` branch exists to catch LIE, whose clients
   score *anomalously low*: `z_lower = 0.6745 * max(0, median - ewma) / mad`.

So **"not recently re-evaluated" and "is a LIE attacker" produce the same signal.**

| MNIST-Gaussian | ConvLSTM-max benign FPs | of which BELOW median (the `z_lower`/LIE branch) | mean ConvLSTM EWMA: benign / malicious |
|---|---|---|---|
| $q_s = 0.0$ | 85 | 42 (49% — no direction, just noise) | 0.145 / 1.047 (**7.2x separation**) |
| $q_s = 1.0$ | 513 | **425 (83%)** | 0.093 / 0.191 (**2.1x — collapsed**) |

Throttling does two bad things at once: it decays the *malicious* signal away
(1.047 -> 0.191, so the temporal arm stops discriminating), and it decays *benign*
clients into the LIE region (so the temporal arm starts false-alarming).

**The ConvLSTM is not inherently redundant — the re-evaluation policy starves it into
redundancy.** Fresh ($q_s = 0$) it is genuinely discriminative: 7.2x separation, and
it is the max arm on 68-89% of true positives.

## 5. This contradicts a claim currently in the paper

Conclusion (`fl_sec.tex:768`) and Sec. V-D:

> "reducing the total number of forward passes by approximately 34% **without
> compromising detection accuracy**"

True for recall. **Unmeasured, and false, for precision:** at the default
$q_s = 0.6$ benign rejection is already 0.112 on MNIST-Gaussian vs 0.078 at
$q_s = 0.2$. The savings are paid for in discarded honest clients. The paper never
saw this because it only reported *accuracy*, which is insensitive here — the model
still converges when 40% of *benign* updates are dropped, because every attacker is
excluded anyway.

## 6. Bonus: the new CIFAR-10 data is genuine, and it rehabilitates §V-D

`results_table_provenance.md` records that the **old** CIFAR-10 Fig. 6 rows were
fabricated (30/33 rows copied from Fashion-MNIST / April sweeps). The **new** ASR/FDR
CIFAR figure is read straight from per-run `client_states_rounds.csv` in genuine July
run dirs — it is real data. Two consequences:

- The paper's §V-D claim that CIFAR-10 accuracy under LIE/Min-Max sits "within
  [0.71, 0.78]" is **false**. Real runs: LIE 0.445-0.477, Min-Max 0.430-0.478
  (50 rounds); Gaussian 0.455-0.495 (50 rounds), 0.598-0.620 (100 rounds).
- The paper's excuse that "CIFAR-10 under the Gaussian attack... fluctuates widely
  ... from near-random to roughly 0.72" is an **artifact of the fabricated rows**.
  The genuine CIFAR-Gaussian sweep is *stable* (0.455-0.495 across all 11 $q_s$).
  That whole hand-wavy variance paragraph can be deleted and replaced with real,
  well-behaved data — which is strictly better for the paper.
- Note `outputs/` also holds **11 x 100-round** CIFAR-Gaussian runs (acc 0.60-0.62)
  that the plot scripts deliberately discard ("prefer 50 rounds"). Only Gaussian has
  them, which is presumably why — but Table III claims $T_{\text{total}} = 100$.

---

## Recommended order of work

### Step 1 — fix the defect, before writing anything

Decouple "not scored this round" from "looks like LIE". Options:

- **(a)** Hold instead of decay for unscored clients: `new_ewma = prev`
  (`strategy.py:3470`). Simplest and most defensible.
- **(b)** Freshness-gate the low branch: apply `z_lower` only to clients actually
  re-scored this round.
- **(c)** Compute the median/MAD over freshly-scored clients only.

Then re-run the $q_s$ sweep. If benign rejection flattens in $q_s$, the "34% savings
for free" claim becomes *measured* rather than asserted. Cheapest high-value
experiment available.

### Step 2 — run the ablation the reviewer demands (C-001)

`stat-only` vs `temporal-only` vs `fused`, reporting **recall and benign rejection**
— not accuracy. Accuracy will *not* separate the variants (the reviewer predicts this
correctly); the detection metrics will. Use >= 3 seeds (review C-003).

**Treat this as genuinely two-sided.** It is the test that *decides* whether the
temporal arm earns its place, not one expected to confirm it. The evidence currently
cuts both ways: fresh ConvLSTM separates 7.2x (good), but even fresh it is the max arm
on 85 false positives *on top of* the statistical arm's 165 (bad). A stat-only variant
matching or beating the fused system on precision is a live outcome. Better to find
that out now than in the next review round.

### Step 3 — then write, with these corrections

1. **Rename the metrics.** "Attack Success Rate" is a targeted/backdoor term of art;
   these are untargeted attacks and the plotted quantity is FN/N. Use
   **malicious evasion rate** = FN/(TP+FN) (or just report recall) and
   **benign rejection rate** = FP/(FP+TN). As plotted, "FDR" = FP/N is neither the
   statistical false-discovery rate (FP/(FP+TP)) nor the FPR; a referee will flag it.
2. **Fix the denominators.** Normalising by N = all participations understates both
   rates: true benign rejection is FDR_plotted / 0.64, ~1.6x larger than plotted.
3. **Explain 36% vs 40%.** Malicious labels are 1800/5000 = 36%, not the stated 40%,
   because the 40 attackers are labelled benign during the 5 warm-up rounds
   (40 x 45 = 1800). State it, or report per-class rates, which are invariant to it.
4. **Declare the horizon.** These sweeps are **50 rounds**. Table III says 100;
   Figs. 2-5 stop at 30. That is a *third* horizon (review C-004). Fix globally.
   (`results_table_provenance.md` already logs this, plus $\kappa = 0.3$ vs the actual
   `fraction-train: 1`, $E = 5$ vs actual 10, $\lambda = 0.9$ vs actual `anom_decay`
   0.6, $R_{\max} = 5$ vs code default 10, $B_{\max} = 50$ vs code default 80.
   **Table III does not describe the runs.**)
5. **Do not narrate a trend into the LIE/Min-Max curves.** Single-seed and bistable,
   jumping between benign rejection ~0.08 and ~0.55 (MNIST-Min-Max across $q_s$:
   0.087, 0.505, 0.071, 0.543, ...). Only the Gaussian trend is clean and consistent
   across all three datasets. Narrating a trend into that noise is precisely C-003.
6. **Reconcile the default.** The data does *not* make $q_s = 0.6$ the
   precision-optimal point. Any text must justify the default against the measured
   curves, or the new figure contradicts the paper's own operating point.

---

## Bottom line

The new results are the most informative thing in the paper — but right now they are a
**diagnostic, not a selling point**. Writing cannot fix a reject grounded in missing
experimentation. What these numbers do is tell you *exactly* which experiment to run,
and point at a concrete defect whose fix would plausibly restore the efficiency claim.
Whether it also restores the ConvLSTM's necessity is an open question that Step 2 —
not any amount of rewriting — will answer.
