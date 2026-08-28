# Review Report

## Paper Information
- **Title**: USTAD-FL: Unsupervised Spatio-Temporal Anomaly Detection for Byzantine-Robust Federated Learning
- **Authors**: Anonymous (double-blind; "First Author," "Second Author")
- **Submission Type**: Journal
- **Target Venue**: IEEE Transactions (specific title redacted as "IEEE TRANSACTIONS ON XXXX"; the security + 6G-edge framing fits IEEE TIFS / TDSC / IoT-J)
- **Review Date**: 2026-07-06
- **Reviewer**: Claude Journal Reviewer v4.1 (6 specialists; Codex and Gemini Deep Think unavailable in this environment)

## Summary

The manuscript proposes **USTAD-FL**, a server-side defense against Byzantine model-poisoning in federated learning (FL). Rather than modifying the aggregator, it filters malicious clients before aggregation. Each received update is layer-filtered, projected into a low-dimensional latent space by an SVD basis fit on clean warm-up rounds, and reshaped into a six-channel spatial "feature frame." Two detectors then score each client: (i) a cheap per-round statistical detector fusing five robust-statistics signals (norm inflation, cosine deviation, MAD-normalized distance, sign disagreement, and a clone/pairwise-similarity score), and (ii) a ConvLSTM next-frame forecaster whose prediction error is EWMA-smoothed, converted into a two-sided robust "relative score" to counter a LIE-induced polarity inversion, and computed only on a selectively re-evaluated, budget-capped subset. The two arms are fused by max-pooling and filtered by a dual adaptive-quantile/absolute threshold before FedAvg aggregation. The paper's headline claims are that USTAD-FL needs no knowledge of the malicious count, no clean server-side dataset, and no attack-type assumption, while remaining robust under non-IID data.

The method is evaluated on MNIST, Fashion-MNIST, and CIFAR-10 at N=100 clients, 40% malicious, Dirichlet α=0.5, against three attacks (Gaussian, LIE, Min-Max) and three defenses (Multi-Krum with privileged knowledge of the attacker count, FLTrust, FLDetector). The reported results show USTAD-FL best in five of six tabulated MNIST/Fashion-MNIST cells, and a q_s sweep showing ~34% (up to ~77%) reduction in ConvLSTM forward passes at "no measurable accuracy loss."

The problem is important and timely, the writing is clear and well-organized, and several design elements (the aggregator-agnostic filtering, the budget-capped re-evaluation, the disclosure that Multi-Krum is privileged) are commendable. However, a coordinated reading by six specialist reviewers found that the paper's **central architectural contribution is neither ablated nor demonstrable under the chosen attack suite**, that its **headline "no clean server-side dataset" claim is contradicted by its own mechanism**, and that its **comparative claims rest on single-run, sub-noise margins with the nearest prior-art baselines absent**. These are foundational rather than cosmetic problems, and the fixes require new experiments rather than editorial revision. The recommendation is **Reject**, with a concrete and constructive path to a substantially stronger resubmission.

## Scores

| Criterion | Score (1-10) | Weight | Weighted | Justification |
|-----------|:---:|:---:|:---:|---|
| Originality & Novelty | 4 | 0.15 | 0.60 | Incremental recombination; each component has a direct precedent (clone↔FoolsGold, sign/magnitude/similarity↔SignGuard, SVD-project-then-detect↔DnC/Li'19, learned/temporal detector↔Li'20/FedPTA). The one genuinely new element—the ConvLSTM arm—is of unproven value. |
| Technical Soundness | 4 | 0.20 | 0.80 | Method is coherent, but a headline claim is contradicted by the mechanism, the per-round median-benign premise is breached on a fraction of rounds, one input channel is attacker-controlled, eq. (9) misuses the CDF for the LIE z-score, and (P1) is posed but never solved/bounded. |
| Experimental Rigor | 2 | 0.15 | 0.30 | Single-run results; no ablation of any component; no detection metrics; missing nearest-neighbor baselines; CIFAR-10 omitted from the results table; sub-noise margins. The experiments do not support the conclusions. |
| Clarity & Presentation | 6 | 0.05 | 0.30 | Prose and organization are strong, but the pipeline figure misrepresents the method, "final round" is defined two incompatible ways, and legibility/accessibility of result plots is borderline. |
| Significance & Impact | 4 | 0.10 | 0.40 | Important problem, but the contribution is incremental, the central claim is unproven, and the evaluation is narrow (one α, one malicious fraction, no adaptive attack). |
| Reproducibility | 2 | 0.10 | 0.20 | No code/data statement; ~15 method-central hyperparameters (including fusion/strictness weights the text says are "reported in Section V" but are not) and the CNN architecture are unspecified. |
| System Model & Solution Feasibility | 4 | 0.25 | 1.00 | QUESTIONABLE: the labeled server set required for accuracy-gating is the *only* trainer of the temporal arm yet is disclaimed; threat-model arithmetic is inconsistent; the large-scale claim is untested and self-contradictory; channel 6 is adversary-controlled. |
| **TOTAL** | - | **1.00** | **3.60** | - |

## Novelty Assessment

| Claim | Verdict | Confidence | Key Evidence |
|-------|---------|:---:|---|
| (Main) Server-side spatio-temporal framework that filters before aggregation without #malicious, clean root set, or attack-type assumptions; non-IID robust | INCREMENTAL | High | Each property is individually anticipated: FLDetector (already cited: no #M, no root, filters) + FoolsGold (Fung et al., RAID 2020: no attacker bound, no auxiliary data, varied distributions). Only the "spatio-temporal" packaging is new. |
| Dual-arm detector (5-signal statistical + ConvLSTM forecaster); two-sided normalization resolves LIE polarity inversion | INCREMENTAL | High | SignGuard (ICDCS 2022) already fuses magnitude+sign+similarity filters with explicit LIE analysis; clone score ≈ FoolsGold; distance/cosine ≈ Krum/median. ConvLSTM next-frame-error anomaly detection is a standard video-anomaly transplant (Lu et al. 2019; Li et al. 2019). Two-sided median±MAD z-score is textbook robust statistics. |
| EWMA adaptive scoring + dual-threshold filter bounding cost (~34% fewer passes) | INCREMENTAL | High | EWMA control charts are textbook (paper cites Hunter 1986); budgeted selective re-evaluation is standard systems engineering. New engineering combination, no methodological novelty; 34% is an operating-point artifact. |
| Overall system as an advance over Multi-Krum/FLTrust/FLDetector | INCREMENTAL | Med | New integration of known parts with an empirical edge, but the three closest precedents (SignGuard, DnC, FedPTA) are never used as baselines, so the "advance" is untested against the nearest prior art. |

**Overall novelty: 4/10.** A competent novel-in-combination system carrying one narrow diagnostic insight (the LIE polarity inversion). The specific ConvLSTM-forecaster-on-update-frames instantiation is new, but its necessity is undemonstrated, and it patches a failure mode that only its own design creates. Note that the SVD projection premise itself is cautioned against by concurrent spectral work (SpectralKrum, 2025) which reports that Min-Max updates are "spectrally indistinguishable."

## Feasibility Assessment

### System Model Realism
- **Physical Layer / Resource Model**: Mostly Realistic — a server-side (BS), aggregator-agnostic defense with near-zero client cost is deployable in principle; the tiny ConvLSTM (4×4, 32 channels) is cheap. However, the declared system model (Sec. III-B) never mentions any server-side evaluation set, which the method silently requires.
- **Network Architecture / Threat Model**: Questionable — the threat model is arithmetically inconsistent: 40 malicious clients cannot all participate when only ~30 clients are sampled per round (κ=0.3), so "all malicious clients submit every round" (Sec. II-B) is false under the stated sampling. Under uniform sampling the per-round malicious count is Hypergeometric(100,40,30) (mean 12, SD ≈2.26), exceeding 50% of participants on roughly 6–13% of rounds, which violates the M<N/2-per-participant condition the coordinate-wise median reference depends on.
- **Parameters**: Questionable — N=100/~30 participants is a small testbed, not "large-scale"; T_w=5 warm-up < T=8 window means no temporal window can even form during warm-up, and at κ=0.3 a client accrues its first 8-frame window only after ~27 rounds—inconsistent with the "round-to-round temporal inconsistency" premise.

### Solution Feasibility
- **Computational Feasibility**: Mostly Feasible at the tested scale (N=100); the budget cap B_t is a genuinely well-designed mechanism.
- **Information Availability**: Questionable→Unrealistic *as claimed* — measuring global-model accuracy each round (Alg. 1 line ~602 "if global accuracy improves by ≥Δ_acc"; strictness signal φ^srv; β-tuning "on a held-out validation split") requires a labeled server-side set. This set is the **only** trainer of the ConvLSTM arm, yet the abstract/Contribution 1/Table I advertise "no clean reference dataset."
- **Scalability**: Limited/Unsupported — line 612 claims "scales linearly with |S_t|" one sentence after stating the clone score is O(|S_t|²); the layer-parameter count D is never reported; no wall-clock, no latency-vs-round-time, no N-sweep.
- **Deployment Practicality**: Questionable — ~20 hyperparameters (≈10 unspecified), and the tuning procedure itself needs the disclaimed labeled set.
- **Energy & Resource Realism**: Not addressed; efficiency is reported only as a ConvLSTM forward-pass proxy, excluding SVD, the O(|S_t|²) clone computation, and retraining backprop.

### Assumption Inventory

| # | Assumption | Type | Classification | Impact if Relaxed | Sensitivity Provided? |
|---|-----------|------|----------------|-------------------|----------------------|
| 1 | Clean warm-up: all clients honest for T_w=5 rounds (SVD basis + initial dynamics) | Explicit | Justified (acknowledged; standard in LIE/Min-Max lit.) | Early poisoning biases the frozen SVD basis and seeds the replay buffer | No |
| 2 | Malicious are a per-round minority (M<N/2 among *participants*); median reference is benign | Explicit globally, Implicit per-round | Questionable | Median/z_max reference invalid on the ~6–13% of rounds where sampled malicious ≥50% | No |
| 3 | Server monitors global-model *accuracy* each round ⇒ labeled server set (Alg. 1 line 602; φ^srv; β-tuning) | Implicit (never declared) | Unrealistic as claimed | Removes the only trainer of the ConvLSTM arm + one strictness signal; collapses the "no root set" advantage | No |
| 4 | Clients honestly self-report Δℓ (channel 6, eq. 15) | Implicit (contradicts arbitrary-adversary model) | Questionable/Unrealistic | Attacker feeds benign-looking loss deltas; 1 of 6 channels is adversary-controlled | No |
| 5 | Fixed malicious set 𝓜 after warm-up | Explicit | Justified (standard; conservative) | Dynamic membership would stress the temporal detector | No |
| 6 | Omniscient adversary (knows w^t and all updates) | Explicit | Justified (strong/conservative; standard) | Weaker adversary only helps | N/A |
| 7 | κ=0.3 participation vs. "all 40 malicious submit every round" | Explicit but contradictory | Unrealistic (internally inconsistent) | Literal reading impossible (40>30); charitable reading breaches per-round majority | No |
| 8 | A temporal frame exists per round (buffer indexed by t) | Implicit | Questionable (contradicts κ=0.3) | ConvLSTM arm silent until ~round 27; "8 consecutive frames" span ~27 non-stationary rounds | No |
| 9 | Frozen SVD basis (from ≤5 early rounds) valid for all 100 rounds | Explicit | Questionable | Non-stationary update geometry; fixed top-K subspace may miss late-training directions | No |

The near-empty "Sensitivity Provided?" column is itself a finding: only q_s is swept; every other assumption is asserted without robustness evidence.

### Standards Compliance
- **Aligned with**: omniscient/fixed-set adversary, untargeted focus, non-IID Dirichlet(0.5), Flower-based FedAvg evaluation — all standard and appropriate; 40% malicious is a valid (if boundary) stress test.
- **Deviations from**: accepted practice in this exact design family declares its server-side data and works to minimize reliance on it (FLTrust root set; and notably ATTA-FL-Lite, IoT-J 2026, uses a nearly identical MAD + cosine-to-median + validation-loss-z-score mechanism but computes the loss on a *server-held* clean set), whereas USTAD-FL disclaims the set and outsources the loss signal to the (possibly malicious) client. Production-FL work (Shejwalkar et al., S&P 2022) also places realistic malicious fractions well below 40%.

### Overall Feasibility Verdict: **QUESTIONABLE**

## Strengths
1. **Important, well-motivated problem (Sections I–II)**: Byzantine robustness under simultaneous non-IID heterogeneity and adaptive (LIE/Min-Max) poisoning is a genuine and active gap; the two-family taxonomy of prior defenses and their limitations is articulated clearly.
2. **Clear, rigorous exposition (Sections III–IV)**: the problem formulation, notation table, and step-by-step method are well-organized and read as professional IEEE-Transactions prose; the security-analysis subsection (IV-H) engages honestly with detector-aware adversaries at the conceptual level.
3. **Scientific honesty on the privileged baseline (Section V-C, Table IV footnote)**: explicitly disclosing that Multi-Krum is granted the exact malicious count—an advantage the proposed method does not receive—is good practice and strengthens the qualitative reading of the comparison.
4. **Well-designed efficiency mechanism (Section IV-E, eqs. 20–21)**: bounding ConvLSTM inference by a budget cap B_t driven by the EWMA IQR, rather than by |S_t|, is a sensible and genuinely useful systems contribution, independent of the detector's other claims.
5. **Aggregator-agnostic, stackable design (Section IV-F, eq. 44)**: framing the defense as a pre-filter compatible with any AGG(·) is a clean and practically attractive property.
6. **Internal figure–table consistency where checkable (Table IV vs. Figs. 3–4)**: the MNIST/Fashion-MNIST plotted endpoints match the tabulated numbers and the best-in-bold markings are all correct—evidence of careful result curation on the reported subset.

## Figure & Table Assessment

### Figures

| Figure | Location | Quality | Legibility | Accuracy | Issues |
|--------|----------|:---:|:---:|:---:|--------|
| Fig. 1 (Pipeline) | Sec. V (fig:pipeline) | Poor | Fair | No | Omits the statistical arm, max-pool fusion, two-sided ρ, and dual-threshold; duplicate "ConvLSTM-EWMA" block title; ~143 DPI raster; notation drift (`Quantile_{ns}`) |
| Fig. 2 (No-defense, a–c) | Sec. V-B | Good | Fair | Partial | Matches Table IV rows, but text's "≈0.10 across all" is contradicted by FMNIST Min-Max ≈0.20; color-only |
| Fig. 3 (MNIST defenses) | Sec. V-C | Good | Fair | Yes | All four methods match Table IV; "Our Method" ≠ "USTAD-FL"; grayscale/CVD-unsafe; x-axis stops at round 30 |
| Fig. 4 (FMNIST defenses) | Sec. V-C | Good | Fair | Yes | Matches Table IV (incl. FLDetector Min-Max 0.704); same accessibility/round-count issues |
| Fig. 5 (CIFAR defenses) | Sec. V-C | Fair | Fair | Unverifiable | Not in any table; "Our Method" curves are attack-invariant and visually identical to the clean run; single favorable draw |
| Fig. 6 (q_s sweep) | Sec. V-D | Good | Fair | Yes (matches text) | 34%/77% not annotated; 9 configs not individually separable; different font/style from Figs 2–5 |

### Tables

| Table | Location | Quality | Formatting | Caption | Data Consistency | Issues |
|-------|----------|:---:|:---:|:---:|:---:|--------|
| Table I (related comparison) | Sec. II-C | Good | Good | Good | Qualitative | USTAD row all-✓ is self-serving; contested by FLDetector/FoolsGold/FLGuard (see Missing References) |
| Table II (notation) | Sec. III-A | Good | Good | Fair | Consistent | Omits load-bearing symbols ρ, Φ, φ_t, τ_abs, B_t |
| Table III (hyperparameters) | Sec. V-A | Good | Good | Fair | Inconsistent | T_total=100 conflicts with 30-round figures; omits β, w_i, τ_c, τ_abs, λ_r, Δ_acc |
| Table IV (main results) | Sec. V-C | Good | Good | Good | Yes (bolding verified) | **Omits CIFAR-10**; no std/error bars |

### Cross-Paper Visual Comparison
No evidence of copied figures. The concern is the opposite: the CIFAR-10 "Our Method" curves are *too* clean—identical across three different attacks and indistinguishable from the no-attack reference—while on easier datasets the curves differ per attack. Combined with the admitted CIFAR-Gaussian instability, the CIFAR figures read as non-representative rather than plagiarized.

### Cross-Reference Integrity
- Orphan figures/tables: none in the manuscript; note that `sweep_tau_quantile.png` exists on disk but is unreferenced (a second sensitivity sweep, not a component ablation).
- Broken references: none. All `\ref{fig:...}`/`\ref{tab:...}`/Algorithm references resolve; every float is cited (subfloat labels are defined but never individually referenced—harmless).

## Weaknesses

### Critical Issues (Must be resolved — may require re-experimentation)

1. **No ablation isolates any component; the ConvLSTM temporal arm's necessity is undemonstrated and likely redundant**
   - *Source*: Experimental Evaluator + Methodology Assessor + Novelty Verifier (independently corroborated)
   - *Location*: Section V (entire); method Sections IV-D/IV-E; fusion eq. (37)
   - *Description*: The paper never isolates the contribution of (a) the ConvLSTM arm vs. statistical-only, (b) each of the five statistical signals, (c) the SVD projection, (d) the two-sided normalization, (e) dual- vs. quantile-only thresholding, or (f) the adaptive scoring policy. Only single-hyperparameter sensitivity sweeps exist. Worse, the fusion is max-pooling (Φ=max(r̄,ρ)), so the cheap statistical arm can independently flag any client—and it plausibly does for all three tested attacks: LIE (eq. 7) and Min-Max (eq. 11) both make every malicious client submit an *identical* vector, so the clone score (eq. 22) saturates to 1 for all attackers; Gaussian inflates the norm (eq. 16). The one clear win over privileged Multi-Krum (MNIST Min-Max, 0.870 vs 0.577) is exactly the clone-structured attack—attributable to the statistical clone score, not the ConvLSTM. The q_s sweep (77% fewer ConvLSTM passes at unchanged accuracy) is itself evidence the temporal arm does little work.
   - *Impact*: The paper's central architectural claim—that spatio-temporal (ConvLSTM) modeling is what delivers assumption-free robustness—is entirely unsupported, and the available evidence suggests it is redundant for the evaluated attacks.
   - *Suggestion*: Add a statistical-arm-only ablation across all datasets/attacks, plus per-component ablations. If (as predicted) statistical-only matches the full system, reframe the paper around the statistical detector and either drop or re-justify the ConvLSTM arm—ideally by exhibiting an attack (e.g., a non-clone, slowly-drifting, or detector-aware attack) on which the temporal arm is demonstrably necessary.
   - *Supporting Evidence*: Fung et al. (FoolsGold, RAID 2020); Xu et al. (SignGuard, ICDCS 2022); the paper's own Fig. 6 and eqs. 7, 11, 16, 22, 37.

2. **The headline "no clean server-side dataset" claim is contradicted by the method's own accuracy-gated mechanisms**
   - *Source*: Feasibility & Realism Assessor + Methodology Assessor (corroborated)
   - *Location*: Abstract; Contribution 1 (Sec. I); Table I ("No root" ✓) — versus Algorithm 1 line ~602, strictness signal φ^srv (Sec. IV-F), and β-tuning (Sec. IV-D "held-out validation split")
   - *Description*: The semi-online ConvLSTM update fires only "if the global accuracy improves by ≥Δ_acc," the strictness signal φ^srv is derived from the "consecutive-round accuracy-improvement streak of the global model," and the fusion weights are tuned on "a held-out validation split." All three require a labeled server-side evaluation set. Critically, this set is the **only** trainer of the ConvLSTM arm (line 578 merely *initializes* f_θ; T_w=5 < T=8, so no warm-up pre-training is possible), so without it the headline temporal detector cannot function as a discriminator. Yet the abstract and Contribution 1 explicitly fault FLTrust for needing a root dataset.
   - *Impact*: A defining claim of the paper and the all-checkmark Table I row are not supportable as stated; the advertised advantage over FLTrust is materially eroded.
   - *Suggestion*: Declare the monitoring set in the system model and Table III (with its size and Δ_acc value); reframe Table I/Contribution 1 as "no root *training* set / no reference-gradient bootstrapping, but a small held-out *evaluation* set for gating," and report a sensitivity analysis to that set's size. Alternatively, replace the accuracy gate with an unlabeled proxy and show the temporal arm still trains.
   - *Supporting Evidence*: Cao et al. (FLTrust, NDSS 2021); ATTA-FL-Lite (IEEE IoT-J 2026), which uses the same MAD/cosine/validation-loss mechanism but declares a server-held clean set.

3. **Comparative claims rest on single-run results with sub-noise margins; no seeds, error bars, or significance tests**
   - *Source*: Experimental Evaluator + Visual & Presentation Evaluator (corroborated)
   - *Location*: Table IV; Section V-C ("best… in five of the six," "matches or exceeds")
   - *Description*: All results are single runs. The margins on which the headline conclusions rest are far inside plausible single-run noise: MNIST-Gaussian 0.888 vs 0.885 (0.003); MNIST-LIE 0.881 vs 0.888 (0.007); Fashion-MNIST gaps of 0.008–0.018. USTAD-FL even *exceeds* the clean-training FedAvg reference (FMNIST-Gaussian 0.744 > 0.743; MNIST by ~2.9 points)—a signature of noise-level differences—while the authors simultaneously invoke "run-to-run variance" to excuse CIFAR (which the sweep shows swinging 0.10↔0.73).
   - *Impact*: No "matches or exceeds" or "best in 5/6" conclusion is statistically supportable; the central empirical claim is unverified.
   - *Suggestion*: Report mean ± std over ≥5 seeds with confidence intervals and significance tests for every cell; plot mean±band curves.
   - *Supporting Evidence*: The paper's own Sec. V-D variance admission; Table IV margins.

4. **Training-horizon / "final round" inconsistency and CIFAR-10 selective, unverifiable reporting (data-integrity)**
   - *Source*: Visual & Presentation Evaluator + Experimental Evaluator (corroborated)
   - *Location*: Table III (T_total=100) vs. Figs. 2–5 (truncated at round 30) vs. Fig. 6 "Final Accuracy" (~0.90–0.93 MNIST) vs. Table IV (0.881–0.888 MNIST); CIFAR-10 omitted from Table IV
   - *Description*: The paper carries two incompatible definitions of "final": Figs. 2–5 stop at round 30 while Table III states 100 rounds, and Fig. 6's "final" MNIST accuracy (~0.90–0.93) sits above Table IV's (0.881–0.888). CIFAR-10—a headline dataset—appears only in figures, never tabulated, despite the text conceding CIFAR-Gaussian "fluctuates widely… from near-random to ~0.72." Because no detection metrics are reported and the CIFAR "Our Method" curves are attack-invariant and identical to the clean run, the favorable CIFAR figures cannot be independently verified.
   - *Impact*: A reader cannot determine what "final accuracy" means, the two reported values disagree by 3–4 points, and the CIFAR presentation reads as selective. This undermines confidence in the reported numbers generally.
   - *Suggestion*: Fix a single horizon across Table III/Table IV/Figs. 2–6; add CIFAR-10 to Table IV as mean±std over ≥3 seeds at the default operating point, including the collapse cases; provide per-seed, per-attack CIFAR curves and release the underlying arrays.
   - *Supporting Evidence*: Table III; Sec. V-D line 757; Figs. 5–6.

### Major Issues (Must be addressed for publication)

1. **Nearest-neighbor baselines are missing—including a defense from the very paper used for the Min-Max attack**
   - *Source*: Experimental Evaluator + Literature Searcher
   - *Location*: Section V-A; Table I
   - *Description*: Only three defenses are compared (Multi-Krum, FLTrust, FLDetector). DnC—the Divide-and-Conquer defense from Shejwalkar & Houmansadr (NDSS 2021), the same paper the manuscript cites for the Min-Max attack—is neither cited as a defense nor run. SignGuard (which anticipates the norm/sign/similarity signals with explicit LIE analysis) is absent. Median and Trimmed-Mean are cited in Table I but never run, though the paper makes claims relative to them. Bulyan, RFA/geometric-median, and any 2023+ defense are also absent.
   - *Impact*: "Matches or exceeds state-of-the-art" is untested against the methods most similar to USTAD-FL; the comparison set is dated and thin for an IEEE Transactions submission.
   - *Suggestion*: Add DnC and SignGuard as baselines at minimum; also run Median/Trimmed-Mean and at least one recent (2023+) detect-and-filter defense (e.g., FLGuard).
   - *Supporting Evidence*: Shejwalkar & Houmansadr (NDSS 2021); Xu et al. (ICDCS 2022).

2. **Related-work coverage is incomplete and the Table I uniqueness claim is misleading**
   - *Source*: Literature Searcher + Novelty Verifier
   - *Location*: Section II-C; Table I; Section I (contribution framing)
   - *Description*: With only 22 actual references, two whole defense families adjacent to USTAD-FL are missing: detection/clustering defenses (FoolsGold, FLAME, DeepSight, FLARE, CONTRA, SignGuard, Auror, FLGuard) and learned-detector methods (Li'19/'20, FedPTA)—the family USTAD-FL itself belongs to. The all-checkmark Table I row is contested from within the paper's own citations: FLDetector already satisfies No-M, No-root, and Filter, while FoolsGold explicitly claims no attacker bound and no auxiliary data.
   - *Impact*: The claimed literature gap is overstated; the paper's positioning is not defensible as written.
   - *Suggestion*: Add the missing families; either expand Table I with detect-and-filter rows (FLDetector-class, FLGuard) or narrow the uniqueness claim to the specific (Adapt. ∧ Non-IID) cell and defend it.
   - *Supporting Evidence*: See Missing References.

3. **No detection metrics for a method whose core contribution is a detector**
   - *Source*: Experimental Evaluator
   - *Location*: Section V (entire)
   - *Description*: The method is a malicious-client detector, yet it is evaluated only by downstream accuracy and a forward-pass count. There are no TPR/FPR, precision/recall, or ROC/AUC of malicious-client identification—precisely the numbers needed to attribute accuracy gains to detection quality (vs. the aggregator's intrinsic robustness) and to independently verify the CIFAR curves.
   - *Impact*: The actual detection contribution cannot be assessed or verified.
   - *Suggestion*: Report per-round TPR/FPR/precision/recall and ROC/AUC per attack and dataset.
   - *Supporting Evidence*: N/A (methodological).

4. **Threat-model arithmetic is inconsistent and breaches the per-round median-benign premise**
   - *Source*: Feasibility & Realism Assessor + Methodology Assessor (corroborated)
   - *Location*: Section II-B (line 87) vs. Table III (κ=0.3); Section III-D / IV median premise; eqs. 7–8
   - *Description*: 40 malicious clients cannot all participate in ~30 sampled slots, so "all malicious clients submit every round" is false under the stated sampling. Under uniform κ, the per-round malicious count is hypergeometric with mean 12 but exceeds 50% of participants on ~6–13% of rounds, violating the M<N/2-per-participant condition on which the coordinate-wise median reference (and the LIE z_max derivation) depends.
   - *Impact*: The statistical arm's stated foundation fails on a non-trivial fraction of rounds; "robust at 40%" overstates the per-round guarantee.
   - *Suggestion*: Correct the wording ("all *sampled* malicious clients attack"), report the realized per-round malicious-participant distribution, and add a fraction with margin below the boundary (e.g., 20–30%).
   - *Supporting Evidence*: Hypergeometric(100,40,30).

5. **Channel 6 feeds an attacker-controlled signal into the detector**
   - *Source*: Feasibility & Realism Assessor + Methodology Assessor
   - *Location*: eq. (15); Section IV-H security claim
   - *Description*: Channel 6 is the client's *self-reported* change in local validation loss. Under the paper's own omniscient/arbitrary-update adversary, a malicious client can report any value—e.g., a population-typical Δℓ—neutralizing the channel and contradicting the claim that an update "must appear benign under every spatial statistic" (channel 6 is not a server-computed statistic of the update).
   - *Impact*: One of six ConvLSTM input channels is adversary-controlled and can also shape the retraining buffer for a stealthy passer.
   - *Suggestion*: Compute the loss signal server-side on the monitoring set, or drop channel 6 and show the remaining channels suffice; analyze a channel-6-spoofing adaptive attacker.
   - *Supporting Evidence*: Sec. III-D threat model; Sec. IV-H.

6. **A detector-aware/adaptive adversary is claimed to be resisted but never evaluated; σ_g unreported**
   - *Source*: Experimental Evaluator + Methodology Assessor
   - *Location*: Section IV-H; Section V-A
   - *Description*: Section IV-H argues resistance to detector-aware adaptation, but no adaptive attack against the fused score is ever run—the security narrative is untested. The Gaussian σ_g is unreported, and the paper concedes large-σ Gaussian is trivially norm-flagged, so the only non-clone attack may be trivial by construction.
   - *Impact*: The security-analysis claims are unsubstantiated; the attack suite may not stress the detector at all.
   - *Suggestion*: Implement an adaptive attack that optimizes against Φ (e.g., a clone-avoiding, norm-matched, slow-drift attack); report and sweep σ_g.
   - *Supporting Evidence*: Sec. IV-H; Sec. V-A.

7. **The large-scale/scalability claim is unsupported and internally self-contradictory**
   - *Source*: Feasibility & Realism Assessor
   - *Location*: Section IV-G (line 612)
   - *Description*: "Total per-round complexity scales linearly with |S_t|" appears one sentence after stating the clone score is O(|S_t|²); the selected-layer parameter count D is never reported; and "large-scale deployment" is tested only at N=100 with no wall-clock, no latency-vs-round-time, and no N-sweep. The efficiency figure counts only ConvLSTM forward passes, excluding SVD, the O(|S_t|²) clone, and retraining.
   - *Impact*: A headline feasibility claim is unproven and inconsistent with the paper's own complexity statement.
   - *Suggestion*: State the true complexity or subsample the clone computation; add an N-sweep with wall-clock and a per-round latency-vs-round-time comparison; report end-to-end cost.
   - *Supporting Evidence*: Sec. IV-G.

8. **Reproducibility: promised hyperparameters are absent and no code is provided**
   - *Source*: Experimental Evaluator + Methodology Assessor + Feasibility Assessor
   - *Location*: eq. (24) text ("values reported in Section V"); Table III; Section V-A
   - *Description*: The text promises the fusion weights are "reported in Section V," but they are not. Unspecified, method-central quantities include β_N…β_L, w_1…w_4, τ_c, τ_abs, λ_r, Δ_acc, ξ_drop, σ_g, n_min, H_max, the SVD layer set ℒ, the ConvLSTM learning rate, and the exact CNN architecture (only "consistent with FLDetector/FLTrust"). No code/data-availability statement.
   - *Impact*: The work cannot be reproduced as written.
   - *Suggestion*: Release code; add a complete hyperparameter appendix and an architecture specification; report all promised weights.
   - *Supporting Evidence*: eq. (24) text; Table III.

9. **The pipeline figure (Fig. 1) misrepresents the proposed method**
   - *Source*: Visual & Presentation Evaluator
   - *Location*: Fig. 1 (fig:pipeline)
   - *Description*: The overview diagram omits the entire five-signal statistical arm, the max-pool fusion Φ=max(r̄,ρ), the two-sided relative score ρ, and the absolute-threshold backstop; it shows only the ConvLSTM→EWMA→quantile→aggregate chain, and duplicates the "ConvLSTM-EWMA Anomaly Scoring" block title. Resolution is ~143 DPI (below IEEE's 300-DPI line-art floor).
   - *Impact*: The paper's own overview figure contradicts its central dual-arm contribution.
   - *Suggestion*: Redraw with both arms → fusion → dual-threshold; relabel the mislabeled block; fix `Quantile_{ns}`→`q_s`; export at ≥300 DPI or as vector art.
   - *Supporting Evidence*: Method Sections IV-D/E/F.

10. **Equation (9) misuses the standard-normal CDF for the LIE z-score**
   - *Source*: Methodology Assessor
   - *Location*: eq. (9), z_max = Φ((N−M−s)/(N−M))
   - *Description*: z_max is used as a σ-multiplier (a z-score) in μ_t − z_max·σ_t, so it must be produced by the *inverse* CDF (quantile function) Φ⁻¹, not the CDF Φ, as in the original LIE attack. As written the argument is a probability in [0,1] and Φ returns another probability, saturating in [0.5, 0.84] and becoming qualitatively wrong at small M (for N=100, M=40: paper's Φ(0.817)=0.793 vs. correct Φ⁻¹(0.817)=0.903; for M=5: 0.70 vs. 0.04).
   - *Impact*: A published equation is incorrect. The experimental impact is likely small (the implementation probably calls `norm.ppf`), but the manuscript's math is wrong as printed.
   - *Suggestion*: Replace Φ with Φ⁻¹ (the quantile function) and confirm the code matches.
   - *Supporting Evidence*: Baruch et al. (NeurIPS 2019).

11. **FLDetector appears to be run outside its design regime (fairness verification)**
   - *Source*: Experimental Evaluator
   - *Location*: Table IV; Section V-C
   - *Description*: FLDetector collapses to exactly 0.103 (random) in almost every cell yet reaches 0.704 on FMNIST/Min-Max. It requires a benign update history (L-BFGS Hessian estimation) that a 5-round warm-up followed by immediate attack denies it, and its detection-window hyperparameter is unreported, so one cannot tell whether it is filtering nothing, over-filtering, or genuinely out of regime.
   - *Impact*: The strongest history-based baseline may be strawmanned; the 0.103-vs-0.704 inconsistency is unexplained.
   - *Suggestion*: Report FLDetector's window/hyperparameters; either grant it the history it requires or explicitly frame the results as the early-attack regime where history-based methods are known to fail; explain the inconsistency.
   - *Supporting Evidence*: Zhang et al. (KDD 2022).

### Minor Issues (Recommended improvements)

1. **Temporal window vs. sparse participation**
   - *Location*: Sec. IV-C (buffer indexed by round t); Table III (T_w=5, T=8, κ=0.3)
   - *Description*: No 8-frame window can form during the 5-round warm-up, and at κ=0.3 a client's first full window spans ~27 non-stationary rounds, undercutting the "round-to-round temporal inconsistency" premise.
   - *Suggestion*: Reconcile the per-round buffer notation with participation; clarify when the ConvLSTM arm becomes active.

2. **Evaluation breadth**
   - *Location*: Sec. V-A
   - *Description*: Only α=0.5 and 40% malicious are tested, despite generality claims.
   - *Suggestion*: Sweep the Dirichlet α and the malicious fraction.

3. **Default q_s=0.6 under-motivated**
   - *Location*: Sec. V-D
   - *Description*: q_s=1.0 gives 77% savings at ~equal accuracy; the "adaptive-adversary margin" rationale for choosing 0.6 is asserted, not tested.
   - *Suggestion*: Justify empirically (e.g., under an adaptive attacker) or adopt the more efficient point.

4. **"≈0.10 across all attacks" overstated**
   - *Location*: Sec. V-B (line 715)
   - *Description*: Contradicted by the authors' own Fig. 2b/Table IV (FMNIST Min-Max ≈0.20).
   - *Suggestion*: Soften the text to match the data.

5. **Accessibility of result plots**
   - *Location*: Figs. 2–6
   - *Description*: Color-only encoding; red (Multi-Krum) vs. green (FLDetector) is a classic colorblind-confusable pair, and blue/red darken alike in grayscale.
   - *Suggestion*: Add distinct markers/dash patterns; unify Fig. 6's style with Figs. 2–5; annotate the 34%/77% reductions.

6. **Legend/label consistency**
   - *Location*: All result plots; Table II
   - *Description*: Legends say "Our Method" while the paper says "USTAD-FL"; Table II omits central symbols (ρ, Φ, φ_t, τ_abs, B_t).
   - *Suggestion*: Relabel to "USTAD-FL"; add the missing symbols to the notation table.

7. **Semi-online retraining on imperfectly-clean windows**
   - *Location*: Sec. IV-E; Alg. 1
   - *Description*: "Verified accuracy gain" rounds can still contain undetected malicious clients (false negatives), so "never trained on suspected-poison rounds" is optimistic; retraining compute is uncounted.
   - *Suggestion*: Quantify false negatives entering the replay buffer; include retraining cost in the efficiency budget.

8. **Problem (P1) is posed but never solved or bounded**
   - *Location*: Sec. III-F
   - *Description*: (P1) is formalized as constrained optimization, but the method is a heuristic filter with no convergence or false-negative guarantee.
   - *Suggestion*: Either add analysis relating the heuristic to (P1) or present (P1) more modestly as motivation.

### Issue Index

| ID | Severity | Section | Issue Title |
|----|----------|---------|-------------|
| C-001 | Critical | Sec. V, IV-D/E | No ablation; ConvLSTM arm's necessity undemonstrated and likely redundant |
| C-002 | Critical | Abstract, Sec. I, Table I vs IV-F | "No clean server-side dataset" contradicted by accuracy-gated mechanisms |
| C-003 | Critical | Table IV, Sec. V-C | Single-run results with sub-noise margins; no seeds/error bars |
| C-004 | Critical | Table III/IV, Figs. 2–6 | Training-horizon/"final" inconsistency + CIFAR-10 selective/unverifiable reporting |
| M-001 | Major | Sec. V-A, Table I | Missing nearest-neighbor baselines (DnC, SignGuard, Median/Trimmed-Mean) |
| M-002 | Major | Sec. II-C, Table I | Related-work incomplete; Table I uniqueness claim misleading |
| M-003 | Major | Sec. V | No detection metrics (TPR/FPR/precision/recall) |
| M-004 | Major | Sec. II-B vs Table III | Threat-model arithmetic inconsistent; per-round median premise breached |
| M-005 | Major | eq. (15), Sec. IV-H | Channel 6 feeds an attacker-controlled signal into the detector |
| M-006 | Major | Sec. IV-H, V-A | Adaptive/detector-aware adversary claimed but never evaluated; σ_g unreported |
| M-007 | Major | Sec. IV-G | Scalability claim unsupported and self-contradictory (linear vs O(|S_t|²)) |
| M-008 | Major | eq. (24), Table III | Reproducibility: promised weights absent; ~15 hyperparameters + architecture unspecified; no code |
| M-009 | Major | Fig. 1 | Pipeline figure misrepresents the method (omits statistical arm, fusion, dual-threshold) |
| M-010 | Major | eq. (9) | z_max uses the CDF Φ where the inverse CDF Φ⁻¹ is required |
| M-011 | Major | Table IV, Sec. V-C | FLDetector appears run outside its design regime (fairness) |
| m-001 | Minor | Sec. IV-C | Temporal window vs. sparse participation timing |
| m-002 | Minor | Sec. V-A | Only one α and one malicious fraction tested |
| m-003 | Minor | Sec. V-D | Default q_s=0.6 under-motivated |
| m-004 | Minor | Sec. V-B | "≈0.10 across all attacks" overstated |
| m-005 | Minor | Figs. 2–6 | Color-only, colorblind/grayscale-unsafe plots |
| m-006 | Minor | Plots, Table II | "Our Method" vs "USTAD-FL"; notation table omissions |
| m-007 | Minor | Sec. IV-E | Retraining on imperfectly-clean windows; cost uncounted |
| m-008 | Minor | Sec. III-F | (P1) posed but never solved/bounded |

## Missing References

### Must-Cite
1. **Fung et al., "Mitigating Sybils in Federated Learning Poisoning" (FoolsGold), RAID 2020 (arXiv 2018)** — the clone score (max pairwise cosine to flag coordinated updates) is FoolsGold's core mechanism; it also claims no attacker-count bound and no auxiliary data, directly contesting the Table I novelty framing. Cite and differentiate.
2. **Shejwalkar & Houmansadr, "Manipulating the Byzantine…" (DnC defense), NDSS 2021** — the projection/spectral outlier-removal defense from the *same* paper already cited for the Min-Max attack; must be cited as a defense and run as a baseline.
3. **Xu et al., "SignGuard…", ICDCS 2022** — filters via magnitude/sign/similarity statistics with explicit LIE analysis; anticipates three of the five statistical signals and the LIE focus. Cite, differentiate, and compare.
4. **Li et al., "Learning to Detect Malicious Clients for Robust FL," arXiv 2002.00211 (2020)** and **"Abnormal Client Behavior Detection in FL," arXiv 1910.09933 (2019)** — the "server learns to detect on low-dimensional update surrogates" framing predates both the SVD-project-then-detect design and the learned-detector arm.

### Should-Cite
1. **Mhamdi et al., "The Hidden Vulnerability of Distributed Learning" (Bulyan), ICML 2018** — canonical robust aggregator and the standard citation for why Krum-type rules leave an adaptive-poisoning margin (motivates the paper's own thesis).
2. **Pillutla et al., "Robust Aggregation for FL" (RFA/geometric median), IEEE TSP 2022** and **Chen, Su, Xu, geometric-median-of-means, SIGMETRICS 2017** — the geometric-median aggregator family is entirely absent from the taxonomy.
3. **Qian et al., "FLARE," IEEE TDSC 2022** — penultimate-layer latent-representation trust scoring; closest peer to the latent-projection idea.
4. **Awan et al., "CONTRA," ESORICS 2021** — cosine-reputation defense against coordinated poisoning (clone-adjacent).
5. **Lee et al., "FLGuard," ESORICS 2023** — assumption-free, non-IID-robust detect-and-filter ensemble; the clearest counterexample to the Table I uniqueness claim.
6. **Mu et al., "FedPTA," IEEE TIFS 2024** — multi-round updates as a tensor with a temporal-continuity prior; the closest learned-temporal analogue to the ConvLSTM arm.

### Consider-Citing
1. **Nguyen et al., "FLAME," USENIX Security 2022; Rieger et al., "DeepSight," NDSS 2022; Shen et al., "Auror," ACSAC 2016** — clustering-detection family (taxonomy completeness).
2. **Cao et al., "FLCert," IEEE TIFS 2022** — certified/ensemble robustness.
3. **Surveys**: Lyu et al., IEEE TNNLS 2020; Ferrag et al., IEEE COMST 2023; Sagar et al., 2023 — to anchor the related-work framing.

## Questions for Authors
1. Provide a statistical-arm-only ablation across all datasets and attacks. If it matches the full system, what specific attack demonstrates that the ConvLSTM arm is necessary?
2. Exactly what server-side data does USTAD-FL require to (a) trigger semi-online ConvLSTM retraining, (b) compute φ^srv, and (c) tune the fusion weights? How large is this set, and how does it differ in kind from FLTrust's root dataset?
3. Under the omniscient/arbitrary-update adversary, why can a malicious client not simply report a benign-looking validation-loss delta to neutralize channel 6?
4. What is the exact evaluation protocol behind Table IV—how many seeds, and what does "final" accuracy mean given that figures stop at round 30 while Table III specifies 100 rounds?
5. Why is CIFAR-10 omitted from Table IV, and why are the CIFAR "Our Method" curves identical across the three different attacks and indistinguishable from the no-attack run?
6. What malicious fraction actually participated per round given κ=0.3, and on what fraction of rounds did malicious clients form ≥50% of participants?
7. Was FLDetector given the benign update history its L-BFGS estimator requires? What explains 0.103 in most cells but 0.704 on FMNIST/Min-Max?
8. Please confirm whether eq. (9) should use Φ⁻¹ (the quantile function) and whether the implementation does.

## Verdict

**Recommendation**: REJECT

**Confidence Level**: 4 (High confidence — expert in this specific area)
- 1: Low confidence (outside expertise)
- 2: Some confidence (familiar with related work)
- 3: Moderate confidence (good understanding of the field)
- 4: High confidence (expert in this specific area)
- 5: Very high confidence (leading expert)

**Justification for Recommendation**:
The weighted score is 3.60/10, below the 5.0 acceptance floor, and at least four independent Critical issues are present—either condition alone dictates rejection under the review rubric. More importantly, the problems are foundational rather than editorial. The paper's central thesis is that spatio-temporal (ConvLSTM) modeling is what enables assumption-free Byzantine robustness, yet no ablation isolates that arm and the max-pool fusion means the cheap statistical detector can carry every decision. The burden is on the authors to show the temporal arm adds anything the statistical arm does not—and the design cuts against them: LIE and Min-Max both emit identical malicious vectors, which saturates the clone score, while Gaussian inflates the norm, so all three evaluated attacks carry a within-round statistical signature before any temporal modeling is applied. The likely outcome of the missing ablation—that a statistical-only variant matches the full system—would invalidate the paper's core contribution, which is precisely why this is a reject rather than a revision: the required work is new experimentation whose result could dissolve the thesis, not text that can be fixed in place.

Two further issues compound this. First, the headline "no clean server-side dataset" claim is contradicted by the method itself: the labeled server set needed to measure per-round accuracy is the only trainer of the ConvLSTM arm, so the temporal detector cannot function without the very resource the paper says it avoids, and its advantage over FLTrust is thereby overstated. Second, every comparative claim rests on single runs with margins (0.003–0.018) far inside the run-to-run variance the authors themselves invoke, with CIFAR-10 omitted from the results table and its figures unverifiable in the absence of any detection metrics. Layered on top are an incremental novelty profile (each component has a direct precedent, and the nearest baselines—DnC from the Min-Max source paper, and SignGuard—are neither cited nor compared), a threat-model arithmetic inconsistency that breaks the median-benign premise on 6–13% of rounds, an attacker-controlled input channel, and an untested "large-scale" claim.

To be clear about what is good: the problem is important, the writing is strong, the budget-capped re-evaluation is a genuinely useful mechanism, and disclosing Multi-Krum's privileged setting is exemplary. A resubmission that demonstrates (or honestly retires) the ConvLSTM arm's value, adds multi-seed results with detection metrics and the missing baselines, corrects the server-data and threat-model claims, and reports CIFAR-10 in full could become a solid contribution. That path is described below. But as submitted, the manuscript's conclusions are not supported by its evidence.

## Conditions for Acceptance

*The recommendation is Reject. The items below define what a substantially stronger resubmission would need; the "Required" tier is non-negotiable for the work to be publishable.*

### Required Changes (Non-negotiable)
1. Add a **statistical-arm-only ablation** (and per-component ablations for the five signals, SVD, two-sided normalization, dual-threshold, and adaptive policy) across all datasets/attacks. If statistical-only matches the full system, reframe the paper accordingly and/or exhibit an attack on which the ConvLSTM arm is provably necessary (e.g., a clone-avoiding, norm-matched, slow-drift or detector-aware attack).
2. **Multi-seed results** (≥5) with mean±std, confidence intervals, and significance tests for every cell, including **CIFAR-10 in Table IV**, plus the CIFAR collapse cases.
3. Report **detection metrics** (per-round TPR/FPR/precision/recall, ROC/AUC) per attack and dataset.
4. Resolve the **"no clean server-side dataset" contradiction**: declare the monitoring set (size, Δ_acc), reframe Table I/Contribution 1, and add a sensitivity analysis—or replace the accuracy gate with a verified unlabeled proxy.
5. Add the **missing nearest-neighbor baselines** (DnC, SignGuard; plus Median/Trimmed-Mean already cited) and the missing citations.
6. Fix the **threat-model arithmetic** (wording + realized per-round malicious-participant distribution) and add a fraction with margin below M<N/2 (e.g., 20–30%).
7. Fix **reproducibility**: report all promised weights and the ~15 unspecified hyperparameters + CNN architecture, and release code.
8. Resolve the **training-horizon/"final round" inconsistency** to a single, clearly-defined horizon across all tables and figures.

### Strongly Recommended
1. Evaluate at least one **adaptive/detector-aware attack** against the fused score Φ, and report σ_g.
2. Redraw **Fig. 1** to depict both arms → fusion → dual-threshold at ≥300 DPI; correct the duplicated block title and notation.
3. Substantiate or correct the **scalability claim** with an N-sweep, wall-clock, and per-round latency-vs-round-time, counting SVD and clone costs.
4. Compute **channel 6 server-side** or drop it and show the other channels suffice; analyze a channel-6-spoofing attacker.
5. Correct **eq. (9)** (Φ → Φ⁻¹) and confirm the code matches.
6. Clarify **FLDetector's configuration** and either grant it its required history or explicitly frame the early-attack regime.

### Optional Improvements
1. Sweep the Dirichlet α; justify or lower the default q_s.
2. Add markers/dash patterns for colorblind/grayscale safety; unify figure styles; annotate the 34%/77% reductions.
3. Relabel "Our Method" → "USTAD-FL"; complete the notation table.
4. Quantify false negatives entering the replay buffer and add retraining cost to the efficiency budget.
5. Either analyze the heuristic's relationship to (P1) or present (P1) more modestly.

## Detailed Comments for Authors

You have built a clean, aggregator-agnostic filtering framework and written it up clearly; the engineering instincts (budget-capped re-evaluation, dual-threshold with an absolute backstop, disclosing Multi-Krum's privileged setting) are good. The difficulty is that the paper's most prominent claims are not the ones the evidence supports, and the gap is structural.

The single most valuable thing you can do is run the statistical-arm-only ablation. Right now, the reader can construct a strong argument that the five-signal statistical detector alone would reproduce almost all of your results: the clone score saturates for LIE and Min-Max (both emit identical malicious vectors), and the norm score catches Gaussian. Your own q_s sweep—77% fewer ConvLSTM passes at unchanged accuracy—points the same way. If the ablation shows the ConvLSTM contributes little, that is publishable *if you tell that story*: a careful demonstration that a cheap statistical screen suffices for clone-structured and norm-inflating attacks, with the ConvLSTM reserved for a specific harder attack class you then exhibit. What is not publishable is asserting the temporal arm as the central contribution without isolating it. Relatedly, the two-sided normalization is presented as a key insight, but it repairs a polarity inversion that only arises because the ConvLSTM learns the stationary LIE vector—your own clone/sign signals already flag LIE by other means, so the "insight" patches a self-created problem.

The server-data issue is worth taking seriously rather than defending. Measuring per-round global accuracy needs a labeled set, and that set is the only thing that ever trains your ConvLSTM (T_w=5 < T=8 leaves no warm-up window). This does not doom the method—an *evaluation* set is arguably lighter than FLTrust's *root training* set—but you must say so, declare it, size it, and test sensitivity to it, rather than claim its absence. Comparable recent work in your exact design family (e.g., ATTA-FL-Lite, which uses the same MAD/cosine-to-median/validation-loss recipe) declares its server set and computes the loss signal server-side; adopting that honesty would also fix your spoofable channel 6.

On rigor: single-run tables with 0.003–0.007 margins cannot support "matches or exceeds" or "best in 5/6," especially when you invoke run-to-run variance elsewhere and your method occasionally beats clean training. Multi-seed results with error bars are non-negotiable, and CIFAR-10 must appear in the table—omitting your hardest dataset while showing only a favorable figure reads as selective reporting, and the attack-invariant, clean-identical CIFAR curves will draw exactly this scrutiny from any careful reviewer. Detection metrics (TPR/FPR) would let you attribute gains to detection quality and would independently corroborate the curves.

Finally, positioning: with 22 references and two whole defense families missing (coordinated-update detectors like FoolsGold/CONTRA; learned/temporal detectors like Li'19/'20 and FedPTA), the "no existing defense removes all assumptions" framing is not defensible—FLDetector already occupies much of your Table I row, and FoolsGold contests the rest. Cite and differentiate these, run DnC and SignGuard (the defenses closest to yours, one of them from the very paper you cite for Min-Max), and the contribution will stand on firmer and more honest ground. We hope these comments help you develop what is a promising system into a paper whose claims match its evidence.
