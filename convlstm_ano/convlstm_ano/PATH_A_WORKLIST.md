# Path A — Finish the Honest USTAD-FL Paper (MNIST + F-MNIST)

**Date:** 2026-07-06
**Canonical manuscript:** `/home/tuan/paper1/convlstm_ano/convlstm_ano/fl_sec.tex` (862 lines, USTAD-FL, TIFS-framed)
**Code + data:** `/home/tuan/get-local-model-bt-id/rl-flwr/` (runnable; outputs current as of 2026-07-06)
**Supersedes:** the Tier-0 of `reviews/20260706_125858_convlstm_anomaly_detection/IMPROVEMENT_ROADMAP.md` — that roadmap was written against the **stale CER-FedAvg PDF**; much of its Tier-0 is already done in this tex (see reconciliation).

## Locked decisions (user, 2026-07-06)
| Decision | Choice |
|---|---|
| Goal | **Path A** — finish the honest paper (writing + targeted experiments); accept incremental novelty |
| CIFAR-10 | **Descope** → keep MNIST + Fashion-MNIST only |
| Source of truth | paper1 USTAD-FL rewrite (this file's folder) |

---

## Reconciliation: review findings vs. the CURRENT tex

### Already fixed in the USTAD-FL rewrite (no action)
| Review ID | Was | Now (verified in current tex) |
|---|---|---|
| D1 / method identity | 3 names | USTAD-FL only (×19); title/abstract/algorithm/conclusion aligned |
| C-002 | unbacked ablation Tables IV/V | **removed** — no fraction/α ablation tables exist |
| C-003 | false "rounds-to-80%" convergence table | **removed** — no such table |
| M-001 | "over 37%", 2000→1250 | "approximately 34%", ~3,500→2,320→820 (matches data), consistent in abstract/intro/§V-D/conclusion; q_s=0 no longer called "exhaustive" |
| M-009 | K/C/ρ/T symbol collisions | remapped: K=latent, C=channels, κ=participation, T=window, T_total=rounds, ν_t=IQR, ρ_{k,t}=relative score, ξ_drop=drop target |
| m-002 | Eq. (1) errors | fixed (sum to N, w_{k,t+1}, no stray F_k) |
| m-003 | forward ref to EWMA | fixed (EWMA Eq. 20 now precedes ρ Eq. 24) |
| D2/D5 | garbled sentence, `[?]` citation | fixed; nist_modified_zscore added |

### Still open — Path A work (grouped into packages below)
| Review ID | Status in current tex | Package |
|---|---|---|
| **C-001** CIFAR unsupported | **OPEN** — Fig. 1(c), Fig. `defense_cifar` (3 panels), Fig. 6 (9 configs incl. CIFAR), abstract "three benchmarks", intro/setup/§V-D/conclusion all still include CIFAR | **WP1** |
| **M-002** hyperparam table wrong | **OPEN** — Table II: `T_total=100`, `κ=0.3`; real runs 30 (defense)/50 (sweep) rounds, full participation. Also Alg. line 576 REQUIRE says "total rounds $T$" (should be $T_{\text{total}}$) | **WP2** |
| **C-004** LIE mis-implemented | **OPEN, both sides** — tex Eq. (27) uses Φ (needs Φ⁻¹); `attack.py:189` `norm.cdf` (needs `norm.ppf`); `attack.py:170` `std ** 0.5` (needs `std`) | **WP3** |
| **M-004** no ConvLSTM-vs-statistical ablation | **OPEN** — the headline contribution is unvalidated; no detection precision/recall/TPR/FPR anywhere | **WP4** |
| **M-005** incremental novelty, closest works uncited | **OPEN** — bib (24 refs) has none of SignGuard, FoolsGold, Fedcvae, FedGuard, Ding, FLANDERS | **WP5** |
| M-006 "communication model" | **PARTIAL** — §III-B still titled "System and **Communication** Model" but models no channel; intro still "6G", "wireless edge", "low-resource network", "BS" throughout | **WP6** |
| M-008 "linear complexity" | **PARTIAL** — §IV-G now states the O(\|S_t\|²) clone term but still concludes "scales linearly"; ConvLSTM cost uses C² where hidden_dim² (32²) dominates | **WP6** |
| M-010 unreported hyperparams / no seeds | **PARTIAL** — Table II expanded, but β_N…β_L, w_1…w_4, τ_c, τ_abs, ξ_drop, λ_r, σ_g, Δ_acc still missing; §IV-F still falsely says β "reported in Section V"; still single-run (no seeds/error bars) | **WP4/WP6** |
| M-007 warm-up circularity | **PARTIAL** — §IV-H "Security Analysis" now owns it in prose (vs FLDetector); still not empirically tested (attacks-from-round-0) | WP6 (accept as stated, or add expt) |
| M-011 spoofable channel-6 | **OPEN** — Eq. ch6 still client-reported Δℓ; not addressed in §IV-H | WP6 |
| M-012 two-sided FP modes | **OPEN** — not analyzed | WP6 |
| m-004 λ_r not tabulated; m-006 thin refs / arXiv survey; m-008 abstract | OPEN (minor) | WP5/WP6 |

---

## Work packages

### WP1 — Descope CIFAR-10 (writing + 1 figure regen)   [no new data needed except Fig. 6 re-plot]
- Fig. 1 (`fig:accuracy_plots`, l.670): drop subfig (c) → 2-panel MNIST/F-MNIST.
- Remove Fig. `defense_cifar` (l.697–704) entirely.
- **Fig. 6** (`fig:computation_reduction`, l.706): re-generate `sweep_accuracy_forwards_all_datasets_attacks.png` with **MNIST + F-MNIST only** (6 configs) via `rl-flwr` `plot_sweep_accuracy_all.py`. Rewrite §V-D (l.755–759): drop the CIFAR [0.71,0.78] claim and the CIFAR-Gaussian outlier paragraph.
- Text: abstract "three benchmarks"→"two" (l.31); intro contribution (l.57); §V-A setup dataset list (l.628); §V-B vanilla-FedAvg (l.715); conclusion (l.770).
- Remove now-orphaned `krizhevsky2009cifar` bib entry.

### WP2 — Fix hyperparameter table + rounds notation (writing)
- Table II (l.638–660): set `T_total` and `κ` to the **real** run config (30/50 rounds; full participation, κ=1.0). Reconcile every `T_total`/`κ` reference.
- Alg. REQUIRE (l.576) and RETURN (l.606): `T` → `T_{\text{total}}`.

### WP3 — Fix + re-run the LIE attack (CODE + EXPERIMENT + writing)   ⚠ compute-dependent
- Code: `attack.py:189` `norm.cdf`→`norm.ppf`; `:170` `std ** 0.5`→`std`; (m-005) compute stats over benign set `B_t`.
- Tex: Eq. (27) Φ→Φ⁻¹.
- **Re-run** LIE for USTAD-FL + Multi-Krum + FLTrust + FLDetector on MNIST + F-MNIST (defense comparison) **and** the q_s sweep. Update Table IV LIE columns (currently MNIST 0.881 / F-MNIST 0.736), Figs. 3–4 LIE panels, Fig. 6 LIE curves.
- ⚠ **Risk:** the corrected LIE is *stronger* (z_max 0.79→0.90). USTAD-FL's LIE accuracy may **drop**. This is the honest result and must be reported as-is.

### WP4 — ConvLSTM-vs-statistical ablation + detection metrics (CODE + EXPERIMENT + writing)   ⚠ compute-dependent
- Code: toggles for statistical-only / ConvLSTM-only / fused; log per-round TP/FP/FN vs known malicious set.
- Run on MNIST + F-MNIST × 3 attacks; report **precision/recall/TPR/FPR** + final accuracy (both reviewers demanded this — it's what earns the "ConvLSTM matters" claim).
- Add one new table (and optionally add the missing detector hyperparameters β/w/τ here to also close M-010).

### WP5 — Citations + honest positioning (writing)
- Add + differentiate: SignGuard (ICDCS'22), FoolsGold (RAID'20), Fedcvae (IPDPS'21), FedGuard (Cluster'23), Ding (Sci.Rep. 2024), FLANDERS (TIFS'25). Soften the "no prior defense does X" gap claims. Replace arXiv-only `shi2022challenges` survey.

### WP6 — TIFS framing + rigor cleanups (writing)
- M-006: retitle §III-B → "System Model"; de-wireless (BS→server; trim 6G/"low-resource"/"wireless edge" veneer) for a clean ML-security paper.
- M-008: make the complexity claim self-consistent (state O(\|S_t\|²) total or justify the practical reduction; use hidden_dim² in the ConvLSTM cost).
- M-010: fix the false "β reported in Section V"; add missing hyperparameters; add ≥3 seeds + error bars if re-running.
- M-011 / M-012: add a sentence each in §IV-H (spoofable channel-6; two-sided FP modes) — or drop channel-6.

---

## Sequencing & dependencies
1. **Experiments first where numbers feed writing:** WP3 (LIE re-run) and WP4 (ablation) change Table IV / Figs 3–4 / Fig 6, so run them **before** finalizing that prose.
2. **Parallel, no-new-data:** WP1 text, WP2, WP5, WP6 can proceed immediately (WP1's Fig. 6 re-plot waits for WP3's LIE re-run so it's regenerated once, CIFAR-free + LIE-corrected together).
3. **Final coherence pass** (`ieee-latex-align` + compile) after all WPs.

## The one blocking question for the user
**Experiments (WP3 LIE re-run, WP4 ablation) need the rl-flwr env + compute.** Decide:
- (a) I drive the runs here (env is live) — you confirm compute is OK to use; or
- (b) you run them and hand me the outputs; or
- (c) defer experiments → then LIE and the ablation claims must be *scoped/marked* honestly (weaker paper).

Until this is answered, only the writing-only WPs (1-text, 2, 5, 6) can start.
