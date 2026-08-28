# Can a statistically-invisible, temporally-visible attack exist here?

**Date:** 2026-07-23 · compute used: none (pure analysis, grounded in source).
**Question (the go/no-go for a ConvLSTM-centric Transactions paper):** does there exist
an attack that (a) passes all five per-round statistics, (b) does real harm, and
(c) is caught by *this* ConvLSTM temporal detector — and is *not* equally caught by a
cheap temporal statistic? If yes → path B is alive. If no → the ConvLSTM cannot be the
contribution and no compute should be spent chasing it.

Verdict up front: **NO for this design.** Three independent structural reasons, below.

---

## 1. What "statistically invisible" pins down (`_compute_per_round_stat_scores`, `_new.py:190-284`)

All five signals compare client k's update to the **coordinate-wise median** of the round:
| Signal | Constraint to pass |
|---|---|
| norm (log2 ratio vs trimmed median) | ‖v‖ ≈ benign median norm |
| cosine ((1−cos to median)/2) | v aligned with the median direction |
| distance (MAD-z of ‖v−median‖) | v within the benign radial spread |
| sign (magnitude-weighted disagreement) | v sign-agrees with the median on heavy coords |
| clone (max pairwise cosine, τ_c=0.95) | v not near-identical to another client |

"Invisible" ⟺ **v lies inside the benign cloud**: right length, right rough direction,
sign-consistent, near-median, non-duplicated. The cloud has width (benign clients span it,
especially non-IID), so there is *room* to move — this is what path B would exploit.

## 2. What the ConvLSTM actually sees — and why invisibility forces a typical frame

The frame (`_build_frame_for_client`, `no_valdelta.py:58-97`) is **the update
cross-sectionally standardized against this round's population**:
`x1=(v−μ_v)/σ_v`, `x2=sign(v)`, `x3=(|v|−μ_m)/σ_m`, `x4=std. norm`, `x5=std. cosine-to-mean`.
Channels 4–5 are *literally the norm and cosine the statistical arm already tests.*

Consequence (stated precisely — this is a *necessary*, not sufficient, condition):
an update inside the benign cloud is, by definition, near the population mean in exactly these
coordinates, so **each frame is, marginally, a typical benign frame in every channel.** This
does *not* by itself imply temporal invisibility — the forecaster keys on frame-to-frame
*transitions*, not on the marginal region, so a within-cloud attacker with non-benign
*dynamics* can still be temporally visible (that is exactly the §4 wedge). What §2 establishes
is narrower but load-bearing: the frame carries **no information beyond the (standardized)
update the statistics already test**, so the temporal arm's *only* possible edge is dynamics —
and §3 shows dynamics is coupled to harm.

## 3. The harm ⟺ persistence coupling (the core impossibility)

The forecaster flags a client only when its frame sequence is **non-persistent** (frame t
poorly predicted from t−4..t−1; it is a shared next-frame MSE model, `anomaly_score`).
So to be temporally visible the attacker's standardized frame must **change over rounds**.
But:
- **Harm requires a persistent idiosyncratic bias.** Model damage = the attackers' sustained
  pull on the aggregate in an adversarial direction. A persistent within-cloud bias is a
  *constant* standardized deviation → perfectly predictable → temporally invisible.
- **Temporal visibility requires a time-varying deviation.** A zero-mean time-varying
  component largely averages out across rounds in the aggregate → little cumulative harm.

The harmful part of the update lives in the temporally-**predictable** subspace; the
temporally-**surprising** part is the harmless part. You cannot have both at once.

## 4. Steelman: the one wedge — duty-cycle / rotation attacks — and why it still fails

The only construction that escapes §3: a **colluding rotation**. Each round a rotating subset
of the 40 attackers sits at the *edge* of the cloud in the adversarial direction (within-cloud,
so per-round-invisible) while the rest act benign. The aggregate always receives a bias
(persistent harm), yet each *individual* attacker's frame oscillates edge↔center
(non-persistent → temporally visible). This genuinely separates per-round statistics from a
temporal detector. But it does not rescue **the ConvLSTM**:

1. **The signal is necessarily first-order — this follows from §3, not from the specific
   attack.** For the rotation to do harm, its sustained component must be a mean-offset in the
   adversarial direction (anything zero-mean averages out across rounds). So the only
   temporally-varying, harm-bearing structure an invisible attacker can carry is an on/off
   modulation of that mean-offset — exactly what an **EWMA/variance of the client's own
   `dist_score` or frame** detects. Any *higher-order* temporal structure an attacker could add
   (cross-coordinate, non-stationary) lives in the harm-neutral subspace (§3), so a detector
   tuned to it would be flagging harmless noise. Hence a **scalar temporal feature — a sixth
   cheap statistic — dominates the ConvLSTM by construction**, before this design's other flaws
   even enter.
2. **The spatial machinery is inert.** The latent is the SVD coefficients (K = min(256,
   basis-rank) ≈ 100 in these runs) **zero-padded to 256 and reshaped 16×16 by singular-component
   index** (`strategy.py:3099-3102`, `svd_target_len=256`). SVD components carry **no spatial
   locality**, so a 3×3 convolution mixes unrelated singular directions (and, since much of each
   frame is zero-padding, largely zeros). The "Conv" contributes nothing a plain per-coordinate
   temporal model wouldn't — and this holds for *any* K < 256, independent of the exact count.
3. **The forecaster cannot even use per-client structure.** It is a *single shared* cell
   (hidden=16, one layer) with no client identity input, on a **frozen round-1 SVD basis**. It
   can learn only generic persistence; it cannot learn "client k's signature," which is what
   would be needed to make within-cloud rotation stand out above benign SGD noise.

So even where a temporal signal exists, it argues for **"add a temporal statistic to the cheap
arm,"** not for a ConvLSTM. And the harm is bounded to the cloud radius (stealthy/weak by
construction), so the payoff for all this machinery is small.

## 5. The only genuine ConvLSTM justification is a different paper

A ConvLSTM earns its spatial+temporal machinery only if (i) the frame has **real spatial
structure** — e.g. actual per-layer / per-filter update maps, where a coordinated perturbation
is a spatial pattern — and (ii) the threat is **targeted/backdoor**, where the adversary must
inject a spatially-structured trigger that persists and evolves. That is a different frame
representation *and* a different threat model than USTAD-FL (which is untargeted, SVD-latent).
It is a **new project needing its own novelty check** — backdoor detection via update structure
is already crowded (DeepSight, FLAME, FLARE) — not a rescue of this one.

## 6. Cross-check against the offline decider (`DECIDER_OFFLINE_FINDINGS.md`)

Theory predicts the temporal channel is weak, inconsistent, and never additive. The data agree:
fresh (q_s=0) ConvLSTM AUC is 0.987 (MNIST-Gaussian) but 0.517 (MNIST-LIE, ≈ random) and swings
0.53↔0.97 on Min-Max across datasets; it never beats the statistical arm (AUC 1.000) and lowers
fused AUC in 6/6 configs. A detector arm that is near-random on one of three attacks and additive
on none is exactly what §2–§4 predict.

---

## Bottom line

There is **no attack that is statistically-invisible, harmful, and specifically-ConvLSTM-visible**
on this design. The wedge that exists (stealthy duty-cycle attacks) is caught by a cheap scalar
temporal feature, over a latent whose spatial arrangement and zero-padding make the
convolutional/forecasting machinery inert. **Path B (save the ConvLSTM) is closed for USTAD-FL as
architected — do not spend compute on it.**

Two verdicts, two *different* epistemic statuses — do not blur them:
- **ConvLSTM path B: dead by argument.** No experiment can revive it; stop.
- **The statistical-detector paper: viability UNKNOWN, gated on one experiment not yet run.**
  Its fate turns on the mandatory **SignGuard (+DnC) baseline** (review item M-001). Because the
  carrying arm is ≈ SignGuard's own norm/sign/similarity signals, a reasonable *prior* is that
  SignGuard matches it — but that is a prior, not a proof, and the comparison decides. Passing it
  is *necessary but not sufficient* (2 datasets / single-seed is still weak). **This re-prioritizes
  compute:** the offline decider already settled the stat/temporal/fused ablation, so the single
  most informative run to do next is **SignGuard + DnC vs. the statistical arm**, not the ablation.
- **A new project** (separate, needs its own novelty check): targeted/backdoor threat + real
  per-layer spatial frames, where a ConvLSTM has genuine structure to model — months of new work
  in a crowded area, not a revision of USTAD-FL.
