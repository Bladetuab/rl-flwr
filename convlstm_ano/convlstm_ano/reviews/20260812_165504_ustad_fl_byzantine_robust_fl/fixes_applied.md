# Critical-issue fixes applied — 2026-08-12

Backup of the pre-fix manuscript: `fl_sec.pre_review_fixes_20260812.tex`
Compile after fixes: 0 errors, 0 undefined references, 2 pre-existing overfull boxes, 22 pages.

## FIXED (9 of 9 criticals addressed; 4 addressed only as scoped disclosures — see below)

| ID | Issue | What was done |
|----|-------|---------------|
| C-002 | Unevaluated third dataset shown in two figures | **Fully fixed.** Regenerated Figs. 5 and 6 with the third dataset excluded. `plot_detection_tau06.py`: `DATASETS` filtered and the subplot grid changed from a hardcoded 3 columns to `len(DATASETS)`. `plot_sweep_accuracy_all.py`: dataset filter added, plus a repair — `collect()` no longer returns `final_accuracy`, so the loader now joins against `final_accuracy_by_dataset.csv` on (dataset, attack, tau). Both regenerated figures were opened and verified to contain only the two evaluated benchmarks, with ranges matching the paper's stated [0.932, 0.960] and [0.772, 0.825]. |
| C-005 | "Attackers cannot move the median" is false at eps=0.4 | **Fully fixed.** All three "cannot move" assertions softened to "bounded". Sec. III-D now states the actual bound: displacement to the `0.5/(1-eps)` benign quantile, i.e. 0.967 sigma at eps=0.4, degrading as eps -> 1/2. Also states the median-capture band `[s/(N-M), (N-M-s)/(N-M)] = [0.183, 0.817]` for shared-vector attacks, and notes that eq. (17) places LIE at exactly 0.183 — the lower edge — because LIE's supporter condition and the median-capture condition are the same inequality. Scoped honestly: proved in parameter space, not established for the latent median. |
| C-001 | ~44% of honest clients discarded with no attacker | **Fully fixed (reporting).** Sec. V-D now reports the defended attack-free control: 43.6% (MNIST) / 44.1% (F-MNIST) mean per round, rising to 45.5% / 44.9% over the final ten rounds, at q_s=0 with the detector enabled for all 50 rounds. The "cold start" attribution is corrected — the rate is shown to follow from the filtering rule (retention <= q_max = 0.90; default q_t = 0.55 discards 45%), not from a transient. Both halves reported: the accuracy cost is negligible (0.9622 / 0.8167, matching the attacked ceiling) while the participation cost is real. Also notes that benign scores stay below tau_abs, so the absolute branch alone would have sufficed here. |
| M-001 | Two-sided normalization credited to the statistical screen | **Fully fixed.** All four misattribution sites (abstract, contribution bullet 2, Sec. III closing, conclusion) rescoped to the temporal arm, each now stating that the five statistical signals are one-sided and that LIE is exposed in the statistical arm by the *clone* score. |
| C-006 | Fig. 1 contradicts the paper's thesis | **Partially fixed.** `pipeline.png` -> `pipeline.pdf` (vector; fixes the 144-DPI defect), and the caption now states plainly that the figure shows the SVD projection, temporal arm, and aggregation stage, and that the five-signal statistical screen runs in parallel and is not drawn. This converts a figure that contradicted the thesis into one that is incompletely labeled. **A redraw showing both arms, the max-pool fusion node, and the tau_abs backstop remains owner work** — the diagram's other defects (duplicate box title, filter shown on `e_{k,t}` instead of the fused `Phi_{k,t}`, missing tau_abs, "Quantile_{ns}", "Broad cast") are inside the source drawing. |
| C-008 | Single seed; headline margins unsupported | **Fixed as far as editing allows.** Sec. V-C now states the binomial SE floor (0.0045 / 0.0062), identifies MNIST/Min-Max (+0.106) as the only decisive win, F-MNIST/Gaussian (+0.018) as marginal, the three remaining nominal wins (+0.003, +0.007, +0.008) as ties, and MNIST/LIE as a loss. The 14.9-point worst-case margin is withdrawn everywhere it appeared (abstract, Sec. I, Sec. II-C, contributions, Sec. V-C) and replaced by the qualitative claim the data supports: USTAD-FL is the only method with no catastrophic configuration. **Multiple seeds still required.** |
| C-003 | Degenerate benchmark; AUC 1.000 is an attack property | **Disclosed, not resolved.** Sec. V-E now states that LIE and Min-Max emit one shared vector, that the clone score is defined to saturate on exactly that condition, that the separation is therefore structural rather than incidental, and that an adversary adding per-client jitter below tau_c would zero the signal. The AUC claim is explicitly reframed as conditional. **The de-cloned attack run is still required.** |
| C-004 | No adaptive attack; "Adapt." claimed | **Disclosed, not resolved.** Table I's Adapt. mark for USTAD-FL downgraded from `\checkmark` to `$\circ^{\ddagger}$` with a footnote stating no detector-aware attack was evaluated. Sec. IV-G reframed as explicitly hypotheses rather than results. The false "must appear benign under *every* spatial statistic" claim corrected — eq. (25) is a weighted sum, not a max, and no single signal's weight (max 0.30) reaches tau_abs = 0.40. **The adaptive attack run is still required.** |
| C-007, C-009 | SignGuard/DnC uncompared; dataset scope | **Not fixable by editing.** Now stated as bounding limitations in the conclusion. |

## Minors also fixed
- Live `% TODO(salvage 2026-07-23)` block removed from the preamble.
- Dead bibliography entries `elmhamdi2018bulyan` and `pillutla2022rfa` removed.
- eq. (17): `Phi` (CDF) corrected to `Phi^{-1}` (quantile function); prose updated to match.
- Table V caption now names the metric ("ROC-AUC ... over the attacked rounds t >= 6, averaged across the eleven values of q_s").
- Fig. 5 in-figure title: "optimal quantile thresholding" -> "at the recommended operating point", matching the text.
- Warm-up FDR restated as declining from ~0.46 to ~0.35 rather than sitting at 0.46.
- "Collapse to ~0.10 across all attack types" corrected to five of six, naming the F-MNIST/Min-Max exception at 0.200.
- "Monotonically increasing accuracy from the first round onward" replaced with the lock-on-latency-consistent statement.
- Contribution bullet 4 corrected from "every configuration" to "five of six", naming the MNIST/LIE loss.

## STILL OPEN — requires new experiment runs, not edits

1. **De-cloned attack variant** (C-003). Per-attacker jitter keeping max pairwise cosine below tau_c = 0.95. Determines whether the detection result and the negative result survive at all. Highest leverage.
2. **Adaptive attack against USTAD-FL** (C-004). Minimize the weighted fused score subject to a displacement constraint.
3. **SignGuard and DnC baselines** (C-007). The paper's own stated priority.
4. **Multiple seeds with paired tests** (C-008).
5. **A harder benchmark and/or higher-capacity model** (C-009) — chosen so benign clients disagree more sharply under heterogeneity, with verified provenance for any dataset re-introduced.
6. **M/N sweep** — the headline "no knowledge of M" claim is never tested at any other adversarial fraction.
7. **Fig. 1 redraw** (C-006) showing both arms and the correct filter.

Until items 1-2 are run, the paper's detection claims remain conditional on an attack implementation
property, and this is now stated in the text rather than left implicit.
