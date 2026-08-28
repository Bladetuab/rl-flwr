# ASR/FDR over (round × tau_s) — Composite Figure Design

## Context

The project already has two views of the same detector sweep, and neither can show what
this figure needs to show:

- `plot_sweep_asr_fdr_<dataset>.py` collapses each run to **one scalar** ASR/FDR and puts
  the swept `tau_s` on the x-axis. The round dimension is gone.
- `plot_rounds_asr_fdr.py` keeps the round dimension and puts round on the x-axis, but
  then draws all 11 `tau_s` runs as overlapping thin lines — `tau_s` is spaghetti, not an
  axis.

This figure makes **both** `tau_s` and `round` real axes at once, and pins the run-level
scalar next to the per-round field so the reader can see how one becomes the other.

### The thesis

The run-level headline is "ASR ≈ 0". That number is a 50-round average of a metric whose
entire mass sits in **one round**:

| metric | rounds 1–5 | round 6 | rounds 7–50 | overall max |
|---|---|---|---|---|
| ASR | exactly 0 | mean 0.114, max **0.300** | max 0.010 | 0.300 |
| FDR | mean **0.410** | mean 0.250 | mean 0.112 | 0.480 |

Averaging the round-6 spike over 50 rounds dilutes it to a run-level ASR of ~0.002. The
left panel of this figure reports that ~0; the right panels show *why* it is misleading —
the spike is a single vertical stripe at the attack onset. Left and right are the same
data at two resolutions.

**They are the same data in an exact, not approximate, sense.** Every round has exactly
100 clients (asserted for every run by `check_geometry()` in `plot_rounds_asr_fdr.py`), so

```
run-level ASR = ΣFN / Σn = (1/50) · Σ_r (FN_r / 100) = mean_r(ASR_r)
```

The left panel is the **row-mean of the heatmap to its right**, not an independently
computed statistic. The figure is therefore self-consistent by construction, and that
relationship is stated on the figure itself.

## Goals

- One figure per dataset, with `tau_s` and `round` both as axes and ASR/FDR as colour.
- Left panel: the run-level ASR/FDR-vs-`tau_s` curve (i.e. the existing sweep figure,
  transposed so its `tau_s` axis becomes the shared vertical axis).
- Make the round-6 ASR spike and the pre-attack FDR cold-start impossible to miss.
- Do not let a near-zero ASR be rendered in a way that looks like signal, or a structural
  zero be rendered in a way that looks like success. (See Integrity guards.)

## Non-goals

- The adaptive `filter_quantile` (per-round, logged in `convlstm_cost.csv`). It is a
  *different* quantile from the swept `tau_s` and is out of scope here — see Appendix.
- Re-running any experiment. This reads existing logs only.
- Touching `plot_sweep_asr_fdr*.py` or `plot_rounds_asr_fdr.py`. This is a new, standalone
  script, consistent with how those two already avoid depending on each other.

## Data & provenance

Source: every `outputs/<date>` directory **on or after 2026-07-06**. Today that resolves to
07-06 … 07-11 (nothing later exists), which is the current 5-channel no-valdelta detector —
the same run set the existing sweep trio reads.

"Since 2026-07-06" is a **start date, not a fixed list**. The sibling scripts hardcode the
six dates, which was complete when they were written but silently drops any run that lands
later. This script discovers the dirs from a `SINCE` constant instead, so a rerun after new
runs appear picks them up.

**Provenance comes from the run-directory name, never from `run_config.json`.**
`run_config.json` is written once per outer Flower session and reports
`attack-mode: gaussian, tau-s-quantile: 0.8` for *every* run regardless of what the run
actually was — because `server_sweep.py` sweeps all `(dataset, attack, tau)` combos
*inside a single* `flwr run`. The directory name (`HH-MM-SS_<dataset>_<attack>_tau<val>`)
is the only reliable label. Existing scripts already parse the dir name; this one does the
same. **Reading `run_config.json` would silently mislabel every point.**

Per-run metrics come from `client_states_rounds.csv` (`round, client_id, is_malicious,
predict_state`), with the project-wide normalisation, evaluated per round:

- `ASR_r = FN_r / N_r` — malicious clients **missed** in round `r`
- `FDR_r = FP_r / N_r` — benign clients **wrongly flagged** in round `r`
- `N_r` = all client rows in round `r` (100), not just the malicious ones

Duplicate `(dataset, attack, tau)` runs across dates are de-duplicated to the 50-round run
(prefer 50 rounds, then most rounds, then latest date) — the same ranking as
`plot_sweep_asr_fdr.collect()` and `plot_rounds_asr_fdr.collect_rounds()`, so all three
scripts read the identical run set.

Grid completeness is verified: 3 datasets × 3 attacks × 11 taus × 50 rounds = **4950
cells, zero gaps**.

## Layout

One file per dataset (matching the `sweep_asr_fdr_forwards_<dataset>` convention).
**Shared y-axis = swept `tau_s` (11 rows, 0.0 … 1.0) across every panel.**

```
  x = ASR/FDR        ASR by round      FDR by round
1.0┌─────────┐  gauss ┌──────────┐ ┌──────────┐
   │ ●    ■  │        │▓░░░░░░░░░│ │▓▓░░░░░░░░│
   │ ●   ■   │        └──────────┘ └──────────┘
   │ ●   ■   │  lie   ┌──────────┐ ┌──────────┐
q_s│ ●  ■    │        │▓░░░░░░░░░│ │▓▓▓░░░░░░░│
   │ ●  ■    │        └──────────┘ └──────────┘
   │ ●   ■   │  mnmx  ┌──────────┐ ┌──────────┐
   │ ●    ■  │        │▓░░░░░░░░░│ │▓▓░░░░░░░░│
0.0└─────────┘        └──────────┘ └──────────┘
    0  .2  .4          1 6   20 50  1 6   20 50
    rate value            round        round
```

- **Left**: one tall panel spanning all 3 rows. x = rate value, y = `tau_s`. All three
  attacks overlaid — colour = attack (Okabe-Ito, reused from the existing scripts),
  solid ● = ASR, dashed □ = FDR. This is the existing sweep figure, transposed.
- **Right**: 3 rows (attacks) × 2 cols (ASR, FDR) heatmaps. x = round (1…50),
  y = `tau_s`, colour = the per-round rate.

## Colour

**One shared sequential scale over [0, 0.48], with a single colorbar**, spanning both
metrics and all panels.

This is a load-bearing choice, not a default. Post-attack ASR maxes out at **0.010**. Under
an independent per-panel scale, that number stretches to full saturation and detector
*noise* renders as though it were *signal*. A shared scale keeps the ASR panels honestly
near-empty, keeps the round-6 stripe as the only thing that lights up, and makes ASR and
FDR magnitudes directly comparable against each other (both are fractions of the same 100
clients, so the comparison is meaningful).

The ramp is **single-hue, light→dark** (`Purples`), so more ink = more error and a
near-white panel reads as "nothing happened" without consulting the colorbar. Single-hue
is required: a multi-hue/rainbow ramp for magnitude is a known anti-pattern. Purple is
chosen because it collides with none of the three categorical attack hues (orange, green,
blue) used in the left panel, and it leaves neutral grey free to mean "masked".

Validated: the categorical trio `#D55E00 / #009E73 / #0072B2` passes all six checks
(worst adjacent CVD ΔE 37.2 protan, 17.0 tritan — both above the ≥12 target).

## Integrity guards

These are the reason this figure needs care rather than a two-line `imshow`.

1. **Mask rounds 1–5 in the ASR panels.** ASR is *structurally* 0 there: rounds 1–5
   contain zero malicious clients, so there is nothing to miss. Rendering those cells with
   the "0.0" colour makes the detector look perfect during the exact window in which it was
   never tested. They are drawn in a neutral grey (distinct from the colormap) and labelled
   "no attackers present". `plot_rounds_asr_fdr.py` already makes this point in prose; the
   heatmap must not quietly undo it.
   FDR cells in rounds 1–5 are **not** masked — false positives are real and meaningful
   with no attacker present (indeed that is where FDR peaks, ~0.41).
2. **Draw the onset rule at round 6** on every heatmap.
3. **Assert the geometry** (100 clients/round, 40 malicious from round 6) on every run
   before plotting, reusing the `check_geometry()` contract. The masking and the ASR
   ceiling are drawn identically on every panel, so they are only correct if the geometry
   holds for every run — checking one run would leave the claim unverified for the rest.
4. **Fail loudly on a missing cell.** The (tau × round) grid must be complete; a silently
   absent run would otherwise render as a 0-valued (i.e. "good") cell.
5. **Assert the left panel really is the row-mean.** The figure *claims* left = row-mean of
   right. So assert it: for every (attack, tau), `ΣFN/Σn` (computed from the raw counts)
   must equal `mean_r(ASR_r)` (the heatmap row) to float tolerance. This turns the figure's
   central claim into a checked invariant instead of a caption.

## Output

- `sweep_figures/sweep_tau_rounds_<dataset>.{png,pdf}` for `mnist`, `fashion_mnist`,
  `cifar10`.
- `sweep_figures/sweep_tau_rounds_marginal.csv` — the left panel's data:
  `(dataset, attack, tau, asr_run, fdr_run, asr_rowmean, fdr_rowmean)`, the last two columns
  recording guard 5 so the identity is auditable outside the script. The per-cell grid is
  **not** re-exported: `plot_rounds_asr_fdr.py` already writes exactly those cells to
  `rounds_asr_fdr_per_round.csv`, and a second copy would be a divergence risk.
- IEEE styling identical to the existing scripts: serif/Times, 8 pt, `IEEE_WIDTH_IN = 7.16`,
  600 dpi on save. **Exception:** gridlines are solid hairlines, not dashed — dashed grid is
  a documented anti-pattern (it reads as "threshold" when it is just a grid).

Script: `plot_tau_rounds_asr_fdr.py`, standalone (its own dedup + metric code), so it
cannot perturb the shared `collect()` helpers the existing figures depend on.

## Appendix — the two quantiles

The codebase has two distinct quantities both called a "quantile". They are not
interchangeable, and only the first is used by this figure:

| | `tau_s_quantile` | `filter_quantile` |
|---|---|---|
| where | run-dir name (`_tau<val>`), swept 0.0–1.0 | `convlstm_cost.csv`, per round |
| kind | **static** per run (config knob) | **adaptive**, recomputed each round |
| range | 0.0 … 1.0 | clipped to [0.35, 0.90], default 0.55 |
| meaning | EWMA threshold above which a client is **re-scored** by the ConvLSTM (`strategy.py:3400`) — a re-scoring *frequency* knob | aggregation **filtering strictness** (`_get_adaptive_filter_quantile`, `strategy.py:2636`); lower q = stricter |

Because `tau_s_quantile` is constant within a run, "quantile vs round" cannot be a line for
it — which is precisely why it becomes an **axis** here and the rate becomes the colour.
A twin-axis "quantile vs round" line plot is only possible for `filter_quantile`, and is a
different figure (out of scope, see Non-goals).
