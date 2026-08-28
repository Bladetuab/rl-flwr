# Reviewer-verified findings (checked in source, not taken from subagents on faith)

## V1 — CONFIRMED: the two-sided normalization is credited to the statistical screen but exists only in the ConvLSTM arm

Raised by the Novelty Verifier; independently verified against `fl_sec.tex`.

**What the method actually does.** The only two-sided score in the paper is

    rho_{k,t} = tanh( 0.6745 * |e_{k,t} - median(e_t)| / (2 (MAD_t^(e) + eps)) )

and `e_{k,t}` is the **ConvLSTM EWMA** (eq:ewma). All five statistical signals are one-sided:

| Signal | Formula | Sidedness |
|---|---|---|
| Norm r^(N) | `min(1, (1/5) log2 max(1, ||delta||/N_trim))` | one-sided (clamped below by max(1,.)) |
| Cosine r^(C) | `(1/2)(1 - cos)` | monotone in deviation, not a two-tail test |
| Distance r^(D) | `tanh(0.6745 * max(0, d - d_med) / (2 MAD))` | **explicitly upper-tail only** |
| Sign r^(S) | weighted disagreement ratio | non-negative |
| Clone r^(L) | `min(1, max(0, (m - tau_c)/(1 - tau_c)))` | hinge, one-sided |

The smoking gun is r^(D) vs rho: **identical structure**, differing only in `max(0, d - d_med)` versus
`|e - e_med|`. The authors demonstrably knew how to write a two-sided score and used the one-sided
form in the statistical arm.

**Where it is misattributed.** Four sites place the two-sided normalization immediately after the
five-signal screen, joined by a semicolon:
- Abstract: "...each referenced to the coordinate-wise median...; a two-sided robust normalization
  resolves the score-polarity inversion..."
- Contribution bullet 2: same construction
- Sec. III closing paragraph: "...two-sided MAD-based normalization, embedded in a dual-threshold
  adaptive filter..."
- Conclusion (most explicit): "...and a two-sided robust normalization neutralizes the score-polarity
  inversion...; **the resulting per-round score** is filtered by a dual adaptive-quantile/absolute
  threshold" -- ties it directly into the per-round statistical pipeline.

Only one site scopes it correctly (pipeline stage 4, the temporal arm).

**Why it matters.** Sec. V-F concludes "the principled choice is to remove [the temporal arm] and
rely on the statistical screen alone." Under that recommended configuration one of the two named
novel mechanisms in the abstract does not exist in the system.

**Knock-on.** The Sec. IV-H detector-aware-adversary argument rests on three pillars; two of them
(the two-sided rho, and max-fusion requiring an update to look benign under *every* statistic *and*
be temporally consistent) disappear if the temporal arm is deleted. The paper cannot both recommend
deletion and keep that security argument.

**Severity calibration -- important.** This is a **claim-integrity** defect, not a functional
failure. The reported experiments were run on the *fused* system, where rho does exist, so the
numbers stand. And the statistical arm empirically handles LIE anyway (AUC 1.000) -- via the *clone*
score, since LIE attackers submit identical vectors, not via two-sided normalization. So the system
works; the description of why it works is wrong. Fix by rescoping the claim, not by re-running
experiments.

## V2 — CONFIRMED: live TODO in the source

Lines 30-33 carry `% TODO(salvage 2026-07-23)` conceding that headline numbers are from the *fused*
system, that SignGuard/DnC are "mandated baselines" not yet run, and referencing `SALVAGE_SCOPE.md`.
It is a LaTeX comment so it does **not** appear in the compiled PDF, but many IEEE journals require
source upload at submission. Must be deleted before submission.

## V3 — CONFIRMED: dead bibliography entries
`elmhamdi2018bulyan` and `pillutla2022rfa` appear in `thebibliography` but are never cited in text.

## V4 — CONFIRMED (reviewer's own check): single-seed caveat is misplaced
Disclosed only in Sec. V-A. The abstract and introduction state precise margins (14.9-point
worst-case lead, >=0.999 detection, a 0.007 margin over Multi-Krum on MNIST/LIE) with no variance
indication. Honest, but placed where a skimming reader reaches it only after believing the numbers.

## V6 — CONFIRMED BY DIRECT IMAGE INSPECTION: two figures plot a third dataset the paper never evaluates

Raised by the Visual Evaluator; I opened both PNGs myself.

- **Fig. 5 (`detection_tau06.png`)**: THREE columns, the third a dataset that appears nowhere in the
  experimental setup, fully populated — detection dips to ~0.47 at round 6 then holds 1.000; FDR shows
  LIE elevated at 0.18-0.25 throughout.
- **Fig. 6 (`sweep_accuracy_forwards_all_datasets_attacks.png`)**: legend lists the two evaluated
  benchmarks plus that same third dataset, whose curves sit at ~0.44-0.50.

Meanwhile the third dataset is named exactly ONCE in `fl_sec.tex` — in the conclusion's future-work
list.

The paper therefore claims that evaluation is future work while displaying completed results for it in
two figures. It also breaks the countable claims: "66 configurations" and "all six configurations"
appear in the abstract, contributions, Sec. V-E and the conclusion, but the figures show 3x3 = 9
configurations and 3x3x11 = 99 runs.

**Aggravating context the reviewer knows from the project's own provenance file**
(`results_table_provenance.md`): a 2026-07-02 audit found that this third dataset's results in an
earlier draft were NOT backed by real runs — 30/33 rows of the June sweep CSV contradicted the
`results.json` of the run directories they referenced, with LIE rows copied from a Fashion-MNIST
sweep. It was deliberately removed from the narrative for that reason. Leaving those curves in two
figures re-imports precisely the data whose provenance the authors themselves could not establish.
This elevates it from a cosmetic slip to a research-integrity concern.

## V7 — CONFIRMED BY DIRECT IMAGE INSPECTION: Fig. 1 depicts the opposite of the paper's thesis

The only architecture figure shows: Feature Extraction (SVD / feature construction / temporal
buffer) -> **CONVLSTM-EWMA ANOMALY SCORING** -> a second box also titled **CONVLSTM-EWMA ANOMALY
SCORING** (actually the filter) -> ROBUST AGGREGATION.

**The five-signal statistical detector does not appear anywhere.** No median reference, no five
signals, no max-pool fusion node. The paper's central claim is that the statistical screen is
load-bearing (ROC-AUC 1.000 in all six settings) and the ConvLSTM arm is redundant and should be
deleted. The figure draws a pure-ConvLSTM system.

Additional defects visible in the image:
- Two boxes share the identical title "CONVLSTM-EWMA ANOMALY SCORING"
- The filter uses `e_{k,t}` (ConvLSTM EWMA), not the fused `Phi_{k,t}` of Eq. (35)/(39); the
  `tau_abs = 0.40` backstop — half of the "dual-threshold" contribution — is absent
- `Quantile_{ns}` should be `Quantile_{q_s}`
- "Broad cast global model" (sic)
- `Predict: F_hat in f_theta(S)` uses set membership where assignment is meant
- 144 effective DPI raster, while an unused vector `pipeline.pdf` sits in the same directory

Almost certainly a legacy figure predating the reframe around the negative result — the directory
still contains `fl_sec.prereframe_20260723.tex`.

## V5 — CONFIRMED: abstract/body numeric consistency is clean
Every quantitative claim in the abstract (0.736, 0.888, 0.999, 66 configurations) traces to a
matching value in the body. No abstract-body drift. Credit where due.
