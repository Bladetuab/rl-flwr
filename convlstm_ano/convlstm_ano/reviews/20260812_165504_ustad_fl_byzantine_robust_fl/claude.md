# Review Report

## Paper Information
- **Title**: USTAD-FL: Unsupervised Statistical Anomaly Detection for Byzantine-Robust Federated Learning
- **Authors**: Anonymous (placeholder authors in source)
- **Submission Type**: Journal
- **Target Venue**: IEEE Transactions (security/forensics tier)
- **Review Date**: 2026-08-12
- **Reviewer**: Claude Journal Reviewer v4.1 (6 Claude specialists; Codex and Gemini Deep Think unavailable in this environment)

> **Conflict-of-interest disclosure.** The reviewer edited this manuscript earlier in the same working
> session (added and then removed a fourth attack, reduced the equation count, restyled captions, and
> drafted the comparative framing in Sec. I and Sec. II-C). A reviewer who wrote portions of the text is
> not independent. Mitigations: the six specialists reviewed from source without that history, and every
> load-bearing finding below was re-verified by the reviewer directly against equations, images, or run
> logs. Findings are labelled **[VERIFIED]** where the reviewer reproduced them personally.

> **Search-coverage limitation.** Two specialists independently established that the configured local
> literature index is a semantic-communications corpus containing **no** federated-learning-security
> papers; none of the manuscript's 28 references resolve in it. All external evidence therefore comes
> from Semantic Scholar. Novelty conclusions should be read as well-supported but not exhaustive.

## Summary

The manuscript proposes USTAD-FL, a server-side Byzantine defense for federated learning built on a
per-round statistical screen. Every client is scored on five robust signals — norm inflation, cosine
deviation, MAD-normalized displacement, signed disagreement, and pairwise clone similarity — each
referenced to the coordinate-wise median of the round. Scores are EWMA-smoothed, fused with a second
ConvLSTM temporal arm by max-pooling, and filtered by a dual threshold combining an adaptive quantile
with a fixed absolute cut, after which only the surviving subset is aggregated. The stated advantages
are that the method requires no knowledge of the malicious-client count, no labeled root dataset, and
no attack-type prior.

The paper's second and more distinctive contribution is a deliberate negative result: a ConvLSTM
next-frame forecaster, built as a temporal detector arm, is shown to add no detection value over the
statistical screen, supported by a decision-layer ablation over 66 sweep configurations and by a
structural "harm–persistence coupling" argument. Evaluation covers MNIST and Fashion-MNIST at a 40%
adversarial fraction under Dirichlet α=0.5 non-IID partitioning, against FedAvg, FLDetector, FLTrust,
and Multi-Krum, under Gaussian, LIE, and Min-Max attacks.

This review finds the manuscript unusually candid — it discloses its single-seed protocol, an EWMA
decay defect, the limits of its own ablation, and its signal overlap with SignGuard, and it publishes
a negative result against its own architecture. That candor is genuine and is credited throughout.
But candor is disclosure, not mitigation, and the review identifies nine critical problems, six of
which the reviewer verified personally. Two are integrity-level: two figures display results for a third
dataset the paper does not evaluate and lists only as future work, and the defense discards roughly
44% of honest clients per round on an attack-free federation — a cost invisible in the paper because its "no attack"
reference row is undefended FedAvg. The evaluation's perfect detection scores are traceable to an
implementation property of the attacks rather than to the detector.

## Scores

| Criterion | Score (1-10) | Weight | Weighted | Justification |
|-----------|:---:|:---:|:---:|---|
| Originality & Novelty | 3 | 0.15 | 0.45 | Five-signal set matched at 4/5 by two independent works; the "median-referenced, per-round, training-free, no-clean-data" mechanism is stated near-verbatim by concurrent work. Negative result is the most defensible contribution but its structural argument is anticipated by PoisonedFL (CVPR 2025) from the attacker side. |
| Technical Soundness | 3 | 0.20 | 0.60 | The central "attackers cannot move the median" premise is false at the evaluated operating point (verified algebraically); no bound is stated anywhere. Sixteen asserted constants, zero sensitivity analyses. τ_abs=0.40 cannot fire on its own motivating case. |
| Experimental Rigor | 3 | 0.15 | 0.45 | Single seed; four of six headline cells fall within noise; the detection benchmark is degenerate; no adaptive attack despite an omniscient threat model; no defended-no-attack control. |
| Clarity & Presentation | 4 | 0.05 | 0.20 | Writing is clear, well signposted, and honest; abstract-body numeric consistency is clean. Offset by an architecture figure that contradicts the paper's thesis, two figures leaking an unreported dataset, and four figures with sub-8pt text. |
| Significance & Impact | 4 | 0.10 | 0.40 | The negative-result framing is valuable and under-attempted in this literature. Its force is reduced because the result is close to arithmetically forced on a saturated benchmark and is confined to three fixed-onset attacks. |
| Reproducibility | 3 | 0.10 | 0.30 | No code, no seeds, no CNN architecture, no σ_g, no batch size, no baseline hyperparameters, no hardware. Offset by an unusually complete detector hyperparameter table and exemplary campaign disclosure. |
| System Model & Solution Feasibility | 4 | 0.25 | 1.00 | No single FL regime satisfies κ=1, N=100, M/N=0.40, per-client server state, and raw-update inspection simultaneously. Secure aggregation is precluded and never mentioned. |
| **TOTAL** | - | **1.00** | **3.40** | - |

## Novelty Assessment

| Claim | Verdict | Confidence | Key Evidence |
|-------|---------|:---:|---|
| Five-signal median-referenced per-round screen | INCREMENTAL | High | KL-CROSS (CSCWD 2026) and OptiGradTrust (Sci. Rep. 2025) each carry 4/5 signals, the latter at the same 40% non-IID regime; SignGuard carries 3/5 and is conceded at line 124 |
| Two-sided MAD normalization resolves LIE polarity inversion | **MISATTRIBUTED** | High | **[VERIFIED]** Present only in the ConvLSTM arm; all five statistical signals are one-sided. See C-002 |
| No M / no root dataset / no attack-type knowledge | DUPLICATE | High | Conceded for FLGuard at line 150; also satisfied by FoolsGold, SignGuard, FLDetector, DCFL |
| Training-free + purely per-round (mechanism parsimony) | INCREMENTAL | Med | Genuinely the lightest point in the design space, but a subtraction from SignGuard rather than a new capability |
| ConvLSTM temporal arm adds no detection value | INCREMENTAL | Med | No prior work explicitly rejects temporal modelling for untargeted FL attacks; but TMRGNN (OJ-COMS 2026) shows the opposite for delayed/slow-drift attacks, a class excluded here by construction |
| Harm–persistence coupling (structural argument) | INCREMENTAL | Med | PoisonedFL (CVPR 2025) states the identical mechanism from the attacker side one year earlier, uncited |
| Dual threshold + 4-signal strictness index | INCREMENTAL | High | FLARE (2025) has a self-calibrating threshold on convergence + attack intensity; ARC (ICLR 2025) does it with proofs |

## Feasibility Assessment

### System Model Realism
- **Federation / participation model**: **Unrealistic** — κ=1 with N=100 and per-client persistent server state is cross-silo; M/N=0.40 belongs to the cross-device threat literature. No deployment satisfies both. The paper never uses the words "cross-device" or "cross-silo" anywhere in 952 lines.
- **Threat model**: **Questionable** — omniscience is declared (line 232) but no evaluated attack is detector-aware (line 249 concedes all three are aggregator-agnostic). MITM is declared at line 91, never used, and would break the M<N/2 accounting the median reference depends on.
- **Parameters**: **Questionable** — 40% is the only adversarial fraction ever evaluated, in a paper whose headline claim is independence from that quantity.

### Solution Feasibility
- **Computational Feasibility**: **Mostly Feasible** at N=100 (≈0.67 GFLOP/round projection; 100×100 Gram matrix). Genuinely cheap where measured.
- **Information Availability**: **Questionable** — the "no root dataset" claim is hedged for the strictness signal but has a second, unconceded consumer: the ConvLSTM training gate uses the same global-improvement signal (lines 438, 585).
- **Scalability**: **Poor** — per-client EWMA state, frame buffers, and re-evaluation scheduling require stable client identities, which production cross-device FL explicitly forbids. R_max=10 is meaningless when the expected re-participation interval is 10³–10⁴ rounds.
- **Deployment Practicality**: **Impractical** — no discussion of dropout, stragglers, asynchronous arrival, identity churn, or update compression.
- **Energy & Resource Realism**: **Questionable** — no wall-clock, hardware, or memory figure anywhere; the only measured cost is of the component the paper recommends deleting.

### Assumption Inventory

| # | Assumption | Type | Classification | Impact if Relaxed | Sensitivity Provided? |
|---|-----------|------|----------------|-------------------|----------------------|
| 1 | Full participation κ=1 | Explicit | Unrealistic | All five signals fail together; paper concedes the median reference breaks | No |
| 2 | M=40 (40%), fixed, M<N/2 | Explicit | Questionable | No correctness argument above N/2; drop-floor dominates below ~10% | No — M never varied |
| 3 | Clean warm-up T_w=5 | Explicit | Unrealistic for an omniscient adversary | SVD basis, both EWMA baselines, and the replay buffer all poisonable | No |
| 4 | Adversary omniscient but not detector-aware | Implicit (232 vs 249) | Unrealistic / inconsistent | Clone-score cliff at τ_c=0.95 evadable by per-client jitter | No |
| 5 | Single, permanent attack onset | Implicit | Questionable | Round-6 recall of 0.35–0.53 would be paid at every onset under burst attacks | No |
| 6 | Stable, addressable client identities | Implicit (eq. 26) | Unrealistic for cross-device | Contradicts the "no historical trajectory" claim at line 60 | No |
| 7 | Server inspects individual raw updates | Implicit (whole method) | Unrealistic vs production | Precludes secure aggregation outright | No — zero mentions |
| 8 | Server monitoring set for global-improvement | Explicit for φ_srv; implicit for the ConvLSTM gate | Questionable | w_srv=0 removes one of two consumers, not the requirement | No |
| 9 | Retention ≤ q_max = 0.90 | Implicit (eq. 34–37) | **Unrealistic** | ≥10% of honest clients discarded every round unconditionally; measured ~44% | No — no defended-no-attack control |
| 10 | S_min = 2 minimum participation | Explicit | Unrealistic | Permits a two-client global model with no benignity guarantee | No |
| 11 | SVD basis fitted once at t=1, frozen | Explicit | Questionable | 0.70 of score mass rides a round-1 projection; untested past 50 rounds | No |
| 12 | N > n_min = 64 for clean basis fit | Explicit | Questionable | For N ≤ 12 the basis never fits in the clean window, or fits on attacked rounds | No |
| 13 | LIE/Min-Max attackers emit bit-identical vectors | Implicit | **Questionable — load-bearing** | Implementation artifact, not attack property; clone score is exactly 0 below τ_c | No |
| 14 | Layer subset L = FC head suffices | Explicit | Questionable | An adversary knowing L attacks conv layers, invisible to all five signals | No |
| 15 | Single training run per configuration | Explicit | Questionable — honestly disclosed | No error bars; several effects called "erratic" by the authors | No — acknowledged |

The pattern in the final column is itself a finding: 14 of 15 assumptions carry no sensitivity
analysis. The one dial that is swept (q_s) is, by the paper's own account at line 476, a
*computational* control governing a redundant arm, not a detection threshold.

### Standards Compliance
- **Aligned with**: Flower as a production-adjacent framework; Dirichlet-α non-IID partitioning as standard practice; appropriate baseline selection; a fair and generous comparison protocol that grants Multi-Krum exact knowledge of M.
- **Deviations from**: Kairouz et al. (2021) cross-device/cross-silo taxonomy — never engaged; Bonawitz et al. (2019) statelessness requirement — violated by per-client buffers; Bonawitz et al. (CCS 2017) secure aggregation — categorically precluded, since the coordinate-wise median is not computable from a masked sum, and never mentioned.

### Overall Feasibility Verdict: **QUESTIONABLE**

## Strengths

1. **Exceptional intellectual honesty (throughout)**: The paper publishes a negative result against its own headline architecture, discloses the single-seed protocol unprompted (line 623), diagnoses its own EWMA decay defect and names it a defect (line 830), refuses to fit trends to erratic single-run data (line 828), concedes signal overlap with SignGuard by name (line 124) and assumption parity with FLGuard (line 150), and flags that FP/N is neither FDR nor FPR (line 782). Very few submissions do any of this, let alone all of it.

2. **A disciplined numerical pipeline (Table IV, Figs. 3–4)**: All twelve defense-cell endpoints in Figs. 3–4 were checked against Table IV and match exactly. Bolding is internally correct including the non-self-serving case — MNIST/LIE bolds Multi-Krum's 0.888 over USTAD-FL's 0.881. Every abstract number traces to a matching body value.

3. **A genuinely cheap screen (Sec. IV-F)**: Training-free, per-round, ~0.67 GFLOP/round of projection at the evaluated scale — a real engineering virtue relative to FLGuard's contrastive ensembles or FLDetector's L-BFGS Hessian estimation.

4. **Correct and cleanly evidenced core observation (Sec. V-E, Table V)**: That max-fusion cannot improve an already-perfect ranker is right, and is demonstrated in 6 of 6 configurations.

5. **The best security reasoning in the paper (Sec. IV-B)**: The explicit refusal of a self-reported validation-loss channel, on the grounds that a client-transmitted quantity is attacker-controllable, is exactly the right instinct.

6. **Negative-result framing is valuable and under-attempted** in this literature, and the specific construction (ConvLSTM next-frame forecasting over per-client feature frames) is genuinely unclaimed territory.

## Figure & Table Assessment

### Figures

| Figure | Location | Quality | Legibility | Accuracy | Issues |
|--------|----------|:---:|:---:|:---:|--------|
| Fig. 1 pipeline | Sec. IV | Poor | Poor | **No** | Statistical arm absent; duplicate box title; wrong filter variable; 144 DPI raster with unused vector twin; printed 8 pages after citation |
| Fig. 2 FedAvg under attack | Sec. V-B | Fair | **Fail** (~3.2pt) | Partial | Collapse at round 4 vs stated onset 6; F-MNIST Min-Max ends at 0.200, not ≈0.10 |
| Fig. 3 defense MNIST | Sec. V-C | Fair | **Fail** (~2.2pt) | Exact | 4.4× downscale; legend occludes curves in (c) |
| Fig. 4 defense F-MNIST | Sec. V-C | Fair | **Fail** (~2.2pt) | Exact | As Fig. 3 |
| Fig. 5 detection at q_s=0.6 | Sec. V-D | Good | Good | **Out of scope** | **Plots an unevaluated third dataset**; title calls q_s=0.6 "optimal" |
| Fig. 6 accuracy vs q_s | Sec. V-F | Good | Good | **Out of scope** | **Plots an unevaluated third dataset**; caption names no datasets |
| Fig. 7 ASR/FDR vs q_s | Sec. V-F | Good | Marginal | Consistent | "ASR" label vs "evasion rate" in text — three names for one quantity |

### Tables

| Table | Location | Quality | Formatting | Caption | Data Consistency | Issues |
|-------|----------|:---:|:---:|:---:|:---:|--------|
| Table I comparison | Sec. II-C | Good | Overfull 16.8pt | Adequate | N/A | Overflows column; "Adapt. ✓" unsupported |
| Table II notation | Sec. III-A | Fair | Clean | Non-standard | **T conflicts with Alg. 1** | Omits Φ, r, β, B_t, φ_t, τ_abs |
| Table III hyperparams | Sec. V-A | Good | Overfull 9.7pt | Adequate | Yes | Δ_acc never assigned a value |
| Table IV main results | Sec. V-C | **Excellent** | Clean | **Best in paper** | **12/12 verified** | Header stacking is awkward |
| Table V ablation | Sec. V-E | Fair | Clean | **Metric never named** | Yes | No "ROC-AUC" anywhere in table or caption |

### Cross-Paper Visual Comparison
Not performed — the local index contains no comparable papers. No conclusion depends on it.

### Cross-Reference Integrity
- Orphan figures/tables: **None** — every figure and table is cited at least once.
- Broken references: **None** — the log reports no undefined or multiply-defined labels.
- Dead labels: 12 (11 subfloat labels plus `sec:complexity`). Panels print (a)/(b)/(c) but are never cited individually, even where it would help.
- **Float placement is the real defect**: Fig. 1 is cited on p7 and printed on p15; Figs. 5–7 print two pages *before* their discussion.

## Weaknesses

### Critical Issues (Must be resolved — may require re-experimentation)

1. **The defense discards ~44% of honest clients on an attack-free federation, and the paper cannot see it**
   - *Source*: Experimental Evaluator (from logs) + Methodology Assessor (from arithmetic) + Feasibility Assessor (from the retention bound) — three independent derivations, then **[VERIFIED]** by the reviewer
   - *Location*: Sec. IV-F eqs. (34)–(37); Table III (q ∈ [0.35, 0.90], default 0.55); Sec. V-D line 784; Table IV "no attack" row
   - *Description*: The filter retains Φ ≤ min(τ^(q), τ_abs) with q_t ≤ q_max = 0.90, so ≥10% of participants are discarded every round unconditionally, and ~45% at the default. The reviewer ran the authors' own attack-free defended runs: mean flagged fraction **0.4358 (MNIST)** and **0.4414 (F-MNIST)** across 50 rounds, still 0.4550/0.4490 over the final ten rounds. Splitting by designation shows the detector is firing on noise, not behaviour — designated-"malicious" clients (with the attack switched off, so behaving honestly) are flagged at 39.5%/49.9% versus 46.7%/40.9% for designated-benign, a gap that **flips sign between datasets**. For contrast, an attacked run gives recall 1.0000 and benign FP 0.0519.
   - *Impact*: Publication-blocking, and invisible to a reader. Sec. V-D attributes the ≈0.46 warm-up false-detection rate to "a cold start in which the detector has no adversarial contrast against which to calibrate" — implying a transient. It is not transient; it persists to round 50. Table IV's "FedAvg (no attack)" row is *plain FedAvg*, so the defense's benign cost is never measured. With N=100 and full participation the model still converges on ~56 clients, so this evaluation is structurally incapable of surfacing the problem.
   - *Suggestion*: Add a defended-no-attack control to both campaigns — **the data already exists on disk**, so this is a reporting change, not a compute cost. Correct the "cold start" attribution. Consider raising q_max to 1.0 or making τ_abs primary with the quantile as backstop; note that on this benchmark the absolute branch alone would have dropped exactly the 40 attackers and zero honest clients.
   - *Supporting Evidence*: `outputs/2026-07-10/15-16-55_mnist_no_attack_tau0.00/client_states_rounds.csv`; `outputs/2026-07-11/13-40-49_fashion_mnist_no_attack_tau0.00/`

2. **Two figures display results for a third dataset the paper does not evaluate**
   - *Source*: Visual Evaluator, **[VERIFIED]** by the reviewer opening both images
   - *Location*: Fig. 5 (`detection_tau06.png`); Fig. 6 (`sweep_accuracy_forwards_all_datasets_attacks.png`)
   - *Description*: Fig. 5 has three columns, the third a dataset appearing nowhere in the experimental setup, fully populated with detection and false-detection curves. Fig. 6's legend lists the same third dataset with accuracy curves well below the two evaluated benchmarks. That dataset is named exactly once in the entire source — in the conclusion's future-work list. The paper says the experiment has not been done while displaying its results.
   - *Impact*: Integrity-level. It also silently breaks every countable claim: "66 configurations" and "all six configurations" appear in the abstract, contributions, Sec. V-E and the conclusion, but the figures show 3×3 = 9 configurations and 99 runs. Aggravating context from the project's own records: a 2026-07-02 provenance audit found that this third dataset's results in an earlier draft were **not backed by real runs** — 30 of 33 sweep rows contradicted the `results.json` of the directories they referenced. It was removed from the narrative for that reason; leaving these curves re-imports exactly that data.
   - *Suggestion*: Regenerate both figures with the dataset list filtered to the two evaluated benchmarks. Fig. 7 already applies that filter correctly, so the pattern exists in the codebase.

3. **The detection benchmark is degenerate: ROC-AUC = 1.000 is a property of the attack implementations**
   - *Source*: Experimental Evaluator (log-verified) + Novelty Verifier (independently)
   - *Location*: Sec. V-E, Table V; mechanism admitted at lines 260, 278, 797
   - *Description*: LIE and Min-Max as implemented make every attacker submit the *identical* vector — the paper says so twice. The clone score is defined to saturate exactly on that condition, and line 797 explains the perfect AUC by that same condition. In the authors' logs at round 10 on MNIST, all 40 LIE attackers share a single distinct score value; all 40 Min-Max attackers share two values differing in the sixth decimal; benign scores span 0.015–0.229. There is no overlap to resolve.
   - *Impact*: Table V, Sec. V-D, the Sec. V-E ablation, and the headline negative result all inherit it. The ablation cannot distinguish "the temporal arm is useless" from "the benchmark saturates the statistical arm." Nothing in Sec. V establishes that the detector separates anything on a non-degenerate instance.
   - *Suggestion*: Add a de-cloned attack variant with per-attacker i.i.d. jitter — the standard sybil-evasion move from the FoolsGold literature the paper already cites. Report per-signal ROC-AUC and a β_L = 0 ablation. Frame symmetrically: if AUC survives, the paper is stronger than it currently argues.

4. **No adaptive attack against this detector, despite an omniscient threat model and an explicit resistance claim**
   - *Source*: Experimental Evaluator + Feasibility Assessor + Methodology Assessor (corroborated)
   - *Location*: Sec. III-C line 232; Sec. IV-G line 602; Table I "Adapt. ✓"; absent from Sec. V
   - *Description*: All three attacks are declared aggregator-agnostic and untailored (line 249). Sec. IV-G claims an update "must appear benign under *every* spatial statistic" — but eq. (25) is a fixed linear weighted sum, not a max; the max is over *arms*. A client saturating the clone score alone scores 0.10 against τ_abs = 0.40. The clone score is a hard cliff at a published τ_c = 0.95: below it the score is exactly zero. An omniscient adversary adds small per-client jitter, preserving the harmful common component while zeroing the signal the paper credits with catching two of its three attacks.
   - *Impact*: Publication-blocking at a security venue. Combined with C-003, the detector has been tested only against adversaries structurally guaranteed to be caught.
   - *Suggestion*: Minimize the weighted score subject to a displacement constraint — directly analogous to the Min-Max formulation already implemented. Absent this, remove the Table I check mark and soften Sec. IV-G to a hypothesis.

5. **The central "attackers cannot move the median" premise is false at the evaluated operating point**
   - *Source*: Methodology Assessor, **[VERIFIED]** by the reviewer numerically
   - *Location*: lines 52, 297, 306, 364, 604 — asserted five times, proved nowhere
   - *Description*: With N=100 and M=40, pushing all malicious values one way moves the coordinate-wise median to benign quantile 0.5/(1−ε) = 0.8333, a displacement of **0.967σ** — not zero. Sharper: because LIE and Min-Max emit one identical vector from all 40 clients, the median *is* the malicious value whenever the attack vector sits in benign-quantile band [11/60, 49/60] = [0.183, 0.817]. The paper's own eq. (17) places LIE at benign quantile 0.1833 — **exactly the lower edge of that band**. This is an algebraic identity, not a coincidence: LIE's "enough supporters" condition is literally the same inequality as median capture. All three numbers were reproduced by the reviewer.
   - *Impact*: The sole stated justification for the entire five-signal design is false as written, with no bound replacing it. Scope note in the paper's favour: this is proved in parameter space per coordinate, while the detector uses the *latent* median; whether the inversion transfers requires the authors' logs. But the assertion as printed is not defensible.
   - *Suggestion*: State and prove the displacement bound q_med = 0.5/(1−ε); acknowledge that eq. (17) calibrates LIE to exactly that limit; report the benign-quantile position of the attack vector per latent coordinate; then either re-derive the design around the real mechanism or adopt a higher-breakdown estimator.

6. **Fig. 1 depicts a system that contradicts the paper's central thesis**
   - *Source*: Visual Evaluator, **[VERIFIED]** by the reviewer opening the image
   - *Location*: `pipeline.png`, Sec. IV
   - *Description*: The only architecture figure runs Feature Extraction → ConvLSTM-EWMA scoring → a second box *also titled* "CONVLSTM-EWMA ANOMALY SCORING" → Robust Aggregation. **The five-signal statistical detector appears nowhere.** No median reference, no five signals, no fusion node. The filter box uses `e_{k,t}` where eq. (39) uses the fused `Φ_{k,t}`, and the τ_abs backstop — half of the "dual-threshold" contribution — is absent. Also "Quantile_{ns}" for q_s and "Broad cast global model".
   - *Impact*: A reader who reads the abstract and then looks at Fig. 1 sees a ConvLSTM paper. Since the headline contribution is a *negative result about the ConvLSTM*, the figure argues against the paper.
   - *Suggestion*: Redraw with the statistical screen as the primary arm, the ConvLSTM as a dashed "explored extension", an explicit max-pool fusion node, and the filter reading min(quantile, τ_abs). Use the unused vector `pipeline.pdf` already in the directory.

7. **The three closest competitors are cited, characterised, and never compared against**
   - *Source*: Experimental Evaluator + Literature Searcher + Novelty Verifier (corroborated)
   - *Location*: Sec. II-C lines 124, 150; Table IV; conceded in the conclusion at line 843
   - *Description*: SignGuard (line 124: signals "overlap"), DnC (the defense from the very paper supplying the Min-Max attack), and FLGuard (line 150: "the all-satisfied row is shared rather than ours alone") are all named and none is run. The newest baseline is FLDetector (2022). Once the paper concedes assumption parity, measured performance is the only differentiator — and it is unmeasured against every method sharing those assumptions.
   - *Impact*: Publication-blocking. There is no evidence the screen beats a 2022 method built from a subset of its own signals.
   - *Suggestion*: Run SignGuard and DnC at minimum. If SignGuard matches, reframe around assumption-lightness and per-round cost — but measure it.

8. **Single seed: the two headline claims are unsupported**
   - *Source*: Experimental Evaluator; disclosed by the authors at line 623
   - *Location*: Abstract; Sec. I lines 56, 63; Sec. V-C line 768; Table IV
   - *Description*: Assuming standard 10,000-example test sets, the SE of a difference is ≈0.0045 (MNIST) / 0.0062 (F-MNIST). On that floor: MNIST/Gaussian +0.003 is a tie; MNIST/LIE −0.007 is a loss; F-MNIST/LIE +0.007 and F-MNIST/Min-Max +0.008 are ties; only MNIST/Min-Max (+0.106) is decisive. "Best in five of six" reduces to one decisive win, one marginal, three ties, one loss. The worst-case column is a minimum over six single draws compared against another minimum over six single draws — the least stable order statistic available.
   - *Impact*: Both headline claims fail the first check a referee runs.
   - *Suggestion*: Without compute — restate three cells as ties and one as a loss, and replace the 14.9-point margin with the qualitative claim the data does support: *USTAD-FL is the only method with no catastrophic cell*, which survives seed noise because the contrast is with 40–70-point collapses. With compute — ≥5 seeds, mean ± std, paired tests.

9. **Dataset scope is below the venue bar**
   - *Source*: Experimental Evaluator + Literature Searcher
   - *Location*: Sec. V-A line 613
   - *Description*: Two 28×28 greyscale 10-class datasets are near-duplicates in structure, giving one axis of variation. Contemporary comparators: FedRefiner (IEEE TIFS 2026) uses 8 datasets, 10 attacks, 13 defenses; OptiGradTrust (Sci. Rep. 2025) — a methodologically near-identical defense with 4 of the same 5 signals — provides three seeds with paired tests, a white-box adaptive attacker, wall-clock, and a per-signal ablation.
   - *Impact*: Independently sufficient grounds for rejection at this venue, and it interacts with C-003: the narrow, easy scope is what makes the degeneracy invisible.
   - *Suggestion*: Add a harder benchmark and/or a higher-capacity model, chosen so benign clients disagree more sharply under heterogeneity. This does double duty — it may break the AUC saturation and make the ablation informative rather than tautological.

### Major Issues (Must be addressed for publication)

1. **A named contribution is not present in the recommended system**
   - *Source*: Novelty Verifier, **[VERIFIED]** by the reviewer against the equations
   - *Location*: Abstract; contribution bullet 2; Sec. III closing; conclusion — four sites
   - *Description*: The only two-sided score is ρ = tanh(0.6745|e−ẽ|/(2(MAD+ε))), computed on the **ConvLSTM EWMA**. All five statistical signals are one-sided; the distance score uses `max(0, d−d̃)` where ρ uses `|e−ẽ|` — identical structure, differing only in that operator. Four sites attach the two-sided normalization to the five-signal screen; only the pipeline stage list scopes it correctly. Sec. V-F recommends deleting the temporal arm, so the shipped system would contain neither the mechanism nor the LIE-polarity story built on it. The Sec. IV-G security argument loses two of its three pillars for the same reason.
   - *Impact*: Claim integrity. Note this does **not** invalidate the results: experiments ran on the fused system where ρ exists, and the statistical arm handles LIE anyway via the clone score. The system works; the description of why it works is wrong.
   - *Suggestion*: Rescope the four claim sites to the temporal arm. Then state explicitly how the *statistical* arm handles LIE — the honest answer appears to be the clone score.
   - *Supporting Evidence*: eq. (14) vs eq. (22) in the source

2. **τ_abs = 0.40 cannot fire on its own motivating case**
   - *Source*: Methodology Assessor, **[VERIFIED]** arithmetically by the reviewer
   - *Location*: line 532; eqs. (18), (25), (27)
   - *Description*: Since the five weights sum to one and each score is in [0,1], the maximum contribution of any single signal equals its weight: norm 0.30, distance 0.25, cosine 0.20, sign 0.15, clone 0.10. **None reaches 0.40.** τ_abs silently encodes "at least two axes must fire," which is never stated. Line 532 motivates τ_abs explicitly by "a large-magnitude Gaussian attack" — which drives the norm channel, capped at 0.30.
   - *Impact*: The absolute-threshold backstop is provably inert on the case the paper uses to justify it.
   - *Suggestion*: Recalibrate against the achievable range of eq. (18) and state the implied two-axis rule.

3. **Ablation over-claim**: the evidence supports "given these trajectories, the temporal arm adds no ranking value," not "the principled choice is to remove it." Two confounds are unconceded: the logged statistical scores come from trajectories filtered by the *fused* detector, and the filter threshold is a quantile of the fused distribution, so removing an arm moves the threshold and changes the trajectory. Run the stat-only system from scratch — the authors' own TODO already mandates it. *(Source: Methodology + Experimental)*

4. **Complexity conclusion is inverted**: the clone score is Θ(|S|²K), not O(|S|²) — normalizing removes the division, not the K-dimensional inner product. The basis-fit cost O(n_b²D) is omitted. The projection dominates by 2–3 orders of magnitude at every realistic size, so "the quadratic term would govern the scaling behavior" is practically backwards, and the proposed mitigation targets the cheaper term. *(Source: Methodology Assessor)*

5. **Per-client server state contradicts the paper's own "purely cross-sectional" claim**: line 60 claims the detector "needs no historical trajectory," but eq. (26) smooths r̄ recursively and eqs. (33)/(37) threshold on r̄, not r. The signals are stateless; the decision is not. Initialization of r̄ for a first-seen client is unspecified. *(Source: Feasibility Assessor)*

6. **Secure aggregation is precluded and never mentioned**: zero occurrences of "secure aggregation," "differential privacy," "homomorphic," or "encrypt" in 952 lines. The coordinate-wise median is not computable from a masked sum, and the pairwise clone score is the most SecAgg-hostile operation in the method. For a paper opening on FL as "a cornerstone of privacy-preserving machine learning," silently discarding its principal privacy mechanism is material. *(Source: Feasibility Assessor)*

7. **Contribution bullet contradicts Table IV**: line 63 claims USTAD-FL "matches or exceeds the strongest baseline in every configuration"; Table IV shows Multi-Krum 0.888 vs 0.881 on MNIST/LIE, and line 768 concedes the loss. *(Source: Experimental Evaluator)*

8. **Zero baseline hyperparameters, and two of three baselines fail in ways resembling misconfiguration**: no FLTrust root-dataset size, no FLDetector window, no Multi-Krum m. FLDetector at 0.100–0.103 in five of six cells is indistinguishable from no defense. No demonstration that any baseline reproduces its published behaviour. *(Source: Experimental Evaluator)*

9. **The harm–persistence argument is not sound as stated**: it equivocates on "predictable" (predictable *after onset* — yet round 6, the onset, is the only round the detector misses, and change-point detection is precisely a temporal tool). It also refutes a detector the paper did not build: ρ is explicitly two-sided and keys on anomalously *low* error, so if harm implies predictability implies low error, low error *is* the harm signature. *(Source: Methodology Assessor)*

10. **The strictness "severity" signal has the wrong sign**: φ^sev = mean/std with weight 0.30. A benign-only population gives ≈3.0 → 0.75; a 40%-attacked population gives ≈1.84 → 0.648. The signal *relaxes* the filter under attack, opposing φ^spr (weight 0.35) — the two largest weights in eq. (24) fight each other. *(Source: Methodology Assessor)*

11. **Four of seven figures have text far below the IEEE 8pt minimum**: Figs. 3/4 are 10in native placed at 2.28in (4.4× reduction), giving ~2.2pt tick labels; Fig. 2 ~3.2pt. Fig. 5 renders at 1.02× and is perfectly legible on the same page — it is the correct template. *(Source: Visual Evaluator)*

12. **Figs. 2, 3, 4 are neither colourblind- nor grayscale-safe**: tab10 red/green encode the key distinctions with colour as the sole channel, and blue/red are near-isoluminant, so "Our Method" and "Multi-Krum" merge in grayscale — exactly the two curves a reader must compare. *(Source: Visual Evaluator)*

13. **Sixteen hard-coded constants, zero sensitivity analyses**, with β anti-correlated with the paper's own account of which signal does the work: line 417 justifies the weights because norm and distance "respond to the largest number of attack families," while line 797 credits the *clone* score with catching LIE and Min-Max — and clone carries the lowest weight, 0.10. *(Source: Methodology Assessor)*

### Minor Issues (Recommended improvements)

1. **Live TODO in the source** — *Location*: lines 30–33. **[VERIFIED]** A `% TODO(salvage 2026-07-23)` comment concedes that headline numbers are from the fused system and that SignGuard/DnC are "mandated baselines" not yet run. Invisible in the PDF but visible in the source, which IEEE journals commonly require at submission. *Suggestion*: Delete before submission.
2. **Dead bibliography entries** — **[VERIFIED]** `elmhamdi2018bulyan` and `pillutla2022rfa` appear in `thebibliography` but are never cited. *Suggestion*: Remove or cite.
3. **Single-seed caveat is misplaced** — disclosed only in Sec. V-A, while the abstract and introduction state precise margins with no variance indication. *Suggestion*: Move a one-clause caveat into the abstract.
4. **eq. (17) uses the CDF where the quantile function is meant** — as written, Φ(49/60) = 0.793 is a probability, not a perturbation scale; Baruch et al. use Φ⁻¹(49/60) = 0.9027. *Suggestion*: Correct to Φ⁻¹.
5. **Table V never names its metric** — headers are bare method names with no "ROC-AUC" in caption, headers, or a units row. *Suggestion*: Add it to the caption along with the t ≥ 6 scope.
6. **Notation collisions** — B_t is both the benign set (line 249) and the scoring budget (eq. 21); η is both the learning rate and the ConvLSTM hidden width; T is the ConvLSTM window in Table II but "total rounds" in Algorithm 1, which also contradicts its own loop bound. *Suggestion*: Disambiguate and complete Table II.
7. **Three names for one quantity** — Fig. 7's legend says "ASR (Attack Success Rate)", the caption and line 828 say "evasion rate FN/N". "ASR" conventionally means backdoor success rate, which the paper defers to future work. *Suggestion*: Relabel to match the caption.
8. **Warm-up FDR mischaracterised** — line 784 says "approximately 0.46", but the curve declines from 0.46 at round 1 to ~0.35 by round 5. *Suggestion*: "starts at ≈0.46 and decays to ≈0.35."
9. **"Monotonically increasing accuracy from the first round onward"** (line 768) is contradicted by Figs. 3(b) and 3(c), which dip at round 6 — the paper's own documented lock-on latency. *Suggestion*: Say so; it reinforces the paper's mechanism.
10. **"Collapse to ≈0.10 across all attack types and datasets"** (line 758) — F-MNIST/Min-Max ends at 0.200, as Table IV itself records. *Suggestion*: "in five of the six settings."
11. **Legends say "Our Method", not "USTAD-FL"** — an anonymous-review artifact in camera-ready figures.
12. **Unspecified parameters block reproduction** — CNN architecture, σ_g, numeric z_max, batch size, seeds, Δ_acc, minimum buffer occupancy, test-set protocol, hardware.
13. **Float placement** — Fig. 1 cited p7, printed p15; Figs. 5–7 printed before their discussion. *Suggestion*: Demote Figs. 2 and 7 to single-column to free the `figure*` queue.
14. **Two overfull hboxes** — Table I (16.8pt) and Table III (9.7pt) overflow the column.

### Issue Index

| ID | Severity | Section | Issue Title |
|----|----------|---------|-------------|
| C-001 | Critical | Sec. IV-F / V-D | ~44% of honest clients discarded with no attacker present |
| C-002 | Critical | Figs. 5, 6 | Unevaluated third dataset displayed in two figures |
| C-003 | Critical | Sec. V-E | Degenerate benchmark; AUC 1.000 is an attack-implementation property |
| C-004 | Critical | Sec. IV-G / V | No adaptive attack despite omniscient threat model |
| C-005 | Critical | Sec. III-F / IV | Median-reference premise false at ε = 0.40 |
| C-006 | Critical | Fig. 1 | Architecture figure contradicts the paper's thesis |
| C-007 | Critical | Sec. V-C | SignGuard, DnC, FLGuard cited but never compared |
| C-008 | Critical | Abstract / Table IV | Single seed; four of six headline cells within noise |
| C-009 | Critical | Sec. V-A | Dataset scope below venue bar |
| M-001 | Major | Abstract / Sec. IV-D | Two-sided normalization credited to the wrong arm |
| M-002 | Major | Sec. IV-F | τ_abs = 0.40 cannot fire on its motivating case |
| M-003 | Major | Sec. V-E | Ablation over-claim |
| M-004 | Major | Sec. IV-F | Complexity conclusion inverted |
| M-005 | Major | Sec. I / IV-D | Per-client state contradicts "purely cross-sectional" |
| M-006 | Major | Whole method | Secure aggregation precluded, never mentioned |
| M-007 | Major | Sec. I | Contribution bullet contradicts Table IV |
| M-008 | Major | Sec. V-C | No baseline hyperparameters; baselines unvalidated |
| M-009 | Major | Sec. V-E | Harm–persistence argument unsound as stated |
| M-010 | Major | Sec. IV-F | Severity signal has the wrong sign |
| M-011 | Major | Figs. 2–4 | Text far below IEEE 8pt minimum |
| M-012 | Major | Figs. 2–4 | Not colourblind- or grayscale-safe |
| M-013 | Major | Sec. IV | Sixteen asserted constants, zero sensitivity analyses |
| m-001 | Minor | Preamble | Live TODO in source |
| m-002 | Minor | Bibliography | Dead entries |
| m-003 | Minor | Abstract | Single-seed caveat misplaced |
| m-004 | Minor | Sec. III-E | eq. (17) uses CDF for quantile function |
| m-005 | Minor | Table V | Metric never named |
| m-006 | Minor | Table II | Notation collisions |
| m-007 | Minor | Fig. 7 | Three names for one quantity |
| m-008 | Minor | Sec. V-D | Warm-up FDR mischaracterised |
| m-009 | Minor | Sec. V-C | Monotonicity claim contradicted by figures |
| m-010 | Minor | Sec. V-B | Collapse claim overstated |
| m-011 | Minor | Figs. 3–4 | "Our Method" in legends |
| m-012 | Minor | Sec. V-A | Unspecified parameters block reproduction |
| m-013 | Minor | Floats | Fig. 1 printed 8 pages after citation |
| m-014 | Minor | Tables I, III | Overfull hboxes |

## Missing References

### Must-Cite
1. **Krauß & Dmitrienko, "MESAS," ACM CCS 2023** — Metric-Cascades employs multiple detection metrics simultaneously against adaptive attackers. Directly refutes the "single-statistic screens" gap claim.
2. **Xie, Fang & Gong, "PoisonedFL," CVPR 2025** — states the harm–persistence mechanism from the attacker side one year earlier, and breaks 8 SOTA defenses via multi-round consistency.
3. **Kunjumon et al., "TMRGNN," IEEE OJ-COMS 2026** — direct counter-evidence for the temporal negative result on delayed/coordinated/slow-drift attacks.
4. **Yang et al., "ScaleSign / MSGuard," IEEE TIFS 2025** — reports 98.23% attack success against SignGuard by modifying sign statistics; threatens one of USTAD-FL's five signals and the Table I "Adapt." mark.
5. **Ding et al., Scientific Reports 2024** — independently reports that "temporal analysis alone struggles"; also uses SVD singular values as detection features.
6. **Kang et al., "MAD-Based Update Filtering," IEEE Access 2026** — states USTAD-FL's core mechanism nearly verbatim (coordinate-wise median reference, modified Z-score, per-round, no clean server dataset). Concurrent; not a citation failure, but must be engaged at revision.

### Should-Cite
1. **Badar et al., "ATTA-FL-Lite," IEEE IoTJ 2025/26** — median-referenced MAD + cosine multi-test screen on MNIST/Fashion-MNIST at up to 40% adversaries, with convergence theory.
2. **Yin & Du, "KL-CROSS," CSCWD 2026** and **Karami et al., "OptiGradTrust," Sci. Rep. 2025** — each carry 4 of USTAD-FL's 5 signals; OptiGradTrust at the same 40% non-IID regime with three seeds and a per-signal ablation.
3. **Younesi et al., "FLARE," arXiv 2025** — near-duplicate of the strictness-index construct; also introduces a Statistical Mimicry attack with persistent drift. Note the name collision with the manuscript's cited `qian2024flare`.
4. **Allouah et al., "ARC," ICLR 2025** — adaptive thresholding with convergence proofs.
5. **Benjamin et al., "FedSECA," CVPRW 2025** — its headline ("existing robust aggregators fail for at least some attacks") is USTAD-FL's consistency claim, already made.
6. **Shejwalkar, Houmansadr, Kairouz & Ramage, IEEE S&P 2022** — the threat-model realism correction from the same authors who supplied the Min-Max attack.

### Consider-Citing
1. **Li et al., "3DFed," IEEE S&P 2023** — evades FoolsGold, FLDetector, FLAME, DeepSight.
2. **Feng et al., "DPFLA," IEEE TSC 2024** — SVD dimensionality reduction for poison detection; the SVD-projection precedent.
3. **Zhou et al., "FLGuardian," IEEE TIFS 2025** and **Zhang et al., "FedRefiner," IEEE TIFS 2026** — target-venue SOTA establishing the current evaluation bar.
4. **Bonawitz et al., CCS 2017 / 2019** and **Kairouz et al. 2021** — required for the secure-aggregation and cross-device/cross-silo discussions that M-006 and C-001 make necessary.

## Questions for Authors

1. Table IV's "FedAvg (no attack)" row is undefended FedAvg. What is the accuracy of **USTAD-FL** on an attack-free federation, and what fraction of honest clients does it discard? Our reading of your own logs gives ~44% discarded, sustained to round 50.
2. Sec. V-D attributes the ≈0.46 warm-up false-detection rate to a "cold start." Since the same rate persists to round 50 in your attack-free runs, is the cold-start explanation still the one you would defend?
3. Figs. 5 and 6 plot a third dataset that the conclusion lists as future work and that appears nowhere in the experimental setup. Which is correct, and what is the provenance of those plotted runs?
4. Where is the two-sided robust normalization computed? Our reading is that ρ operates only on the ConvLSTM EWMA, and that all five statistical signals are one-sided. If the temporal arm is deleted as Sec. V-F recommends, what handles the LIE polarity inversion?
5. What is the coordinate-wise median's displacement bound at M/N = 0.40? Given eq. (17) places LIE at benign quantile 11/60 — exactly the edge of the band in which the malicious vector *is* the median — how do you reconcile the stated mechanism with the reported AUC of 1.000?
6. Do all 40 attackers submit bit-identical vectors in your LIE and Min-Max implementations? If so, what happens to the clone score, and to Table V, under per-client jitter that keeps max pairwise cosine at 0.94?
7. What is the CNN architecture, and what value of σ_g was used for the Gaussian attack?

## Verdict

**Recommendation**: **REJECT** (resubmission encouraged after substantial new experiments)

**Confidence Level**: **4** (High confidence — expert in this specific area)
- 1: Low confidence (outside expertise)
- 2: Some confidence (familiar with related work)
- 3: Moderate confidence (good understanding of the field)
- 4: High confidence (expert in this specific area)
- 5: Very high confidence (leading expert)

**Justification for Recommendation**:

The weighted score of 3.40 falls below the 5.0 rejection threshold, and nine critical issues far exceed
the two that trigger rejection under the decision matrix. But the arithmetic is not the reason. Three
findings are individually sufficient. First, the defense discards roughly 44% of honest clients on an
attack-free federation — verified in the authors' own logs, sustained through round 50, and structurally
invisible to this evaluation because the "no attack" reference row is undefended FedAvg and because a
100-client federation still converges on 56 clients. Second, the perfect detection scores that carry
Table V, Sec. V-D, and the headline negative result are traceable to an implementation property of the
attacks — all 40 attackers submit identical vectors, and the clone score is defined to saturate exactly
on that condition — so the benchmark cannot distinguish a working detector from a degenerate one. Third,
two figures display results for a third dataset the paper does not evaluate, and the project's own
provenance records show that dataset was removed precisely because an audit found those results
unbacked by real runs.

The paper's intellectual honesty is genuine and unusual, and this review has credited it throughout: it
publishes a negative result against its own architecture, names its overlap with SignGuard, discloses
its single seed unprompted, and diagnoses its own EWMA decay defect. Its numerical pipeline is
disciplined — all twelve Table IV cells match their figure endpoints exactly. That candor is why the
recommendation is reject-and-resubmit rather than a judgment about the work's seriousness. But candor is
disclosure, not mitigation. The authors' own stated priority list — run SignGuard and DnC, use multiple
seeds, apply significance testing — is precisely the mandatory-revision list, and a live TODO in the
source concedes that the headline numbers describe a configuration the paper recommends replacing.

Rejection rather than major revision is appropriate because the required work exceeds a revision cycle
and because two outcomes are currently unknowable. Whether the detector separates anything on a
non-degenerate instance is untested; whether the negative result survives a de-cloned attack is untested.
If the de-cloned experiment holds up, the paper becomes considerably stronger than it currently argues,
and the negative result is a real asset. If it collapses, the manuscript needs a different thesis rather
than a revision. That question should be settled before a venue commits reviewer time to a second round.

## Conditions for Acceptance (if REVISION)

### Required Changes (Non-negotiable)
1. Report a **defended, attack-free control** for both campaigns and correct the "cold start" attribution. The data is already on disk — this is the highest value per unit effort in the entire revision.
2. Regenerate Figs. 5 and 6 with the dataset list filtered to the two evaluated benchmarks, or promote the third dataset to a fully documented benchmark with verified provenance.
3. Add a **de-cloned attack variant** (per-attacker jitter keeping max pairwise cosine below τ_c) and report per-signal ROC-AUC plus a β_L = 0 ablation.
4. Add an **adaptive attack** minimizing the fused score subject to a displacement constraint, or remove the Table I "Adapt." check mark and soften Sec. IV-G to a hypothesis.
5. Compare against **SignGuard and DnC** at minimum.
6. Report **≥5 seeds with mean ± std and paired tests**, or restate the three tied cells and the one lost cell honestly and drop the 14.9-point margin in favour of the qualitative no-catastrophic-cell claim.
7. **Sweep M/N** over at least {0.05, 0.10, 0.20, 0.30, 0.40}.
8. State and prove the **median displacement bound**, and reconcile eq. (17)'s placement of LIE with the reported AUC.
9. **Rescope the two-sided normalization claim** to the temporal arm at all four sites, and state what handles LIE in the statistical arm.
10. **Redraw Fig. 1** to include the statistical screen, using the existing vector `pipeline.pdf`.
11. Specify the **CNN architecture, σ_g, z_max, batch size, seeds, and all baseline hyperparameters**; remove the live TODO at lines 30–33.

### Strongly Recommended
1. Add a paragraph on **secure aggregation and differential privacy** stating the incompatibility plainly and surveying MPC/quantile-based reconciliations.
2. Name the deployment regime (cross-device vs cross-silo) and commit to it.
3. Add **a harder third benchmark** (and/or a higher-capacity model) — it may break the AUC saturation and make the ablation informative.
4. Correct the complexity analysis (clone is Θ(|S|²K); include the basis-fit term; report the true crossover).
5. Fix the severity signal's sign, or replace it with a bimodality/gap statistic.
6. Regenerate Figs. 2–4 at final print size with colourblind- and grayscale-safe encoding; Fig. 5 is the correct template.
7. Cite and engage MESAS, PoisonedFL, TMRGNN, and ScaleSign.

### Optional Improvements
1. Add a per-signal ablation to justify the fusion weights empirically.
2. Add a sensitivity analysis for τ_abs, τ_c, and [q_min, q_max].
3. Report wall-clock and memory versus plain FedAvg for the stat-only configuration.
4. Move a single-seed caveat clause into the abstract.
5. Fix the notation collisions and complete Table II.
6. Resolve the float placement so Fig. 1 lands near Sec. IV.

## Detailed Comments for Authors

**On the shape of the contribution.** The most valuable thing in this manuscript is not the five-signal
screen — which the literature has largely converged on independently — but the willingness to publish a
negative result against your own architecture. That instinct is right and the field needs more of it.
The problem is that the negative result, as currently evidenced, is close to arithmetically forced: once
the statistical arm attains AUC = 1.000, max-fusion *cannot* improve it, so "the temporal arm never
helps" follows from the fusion rule rather than from any property of temporal modelling. The result
becomes genuinely informative only on a benchmark where the statistical arm is *not* saturated. That is
the single most valuable experiment available to you, and it is the same experiment that resolves C-003.

**On positioning.** Section II-C is the best-written part of the paper. It names its overlaps, concedes
that the all-satisfied row is shared with FLGuard, and correctly identifies mechanism parsimony rather
than capability as the contribution. The Introduction and abstract then assert three gaps that Section
II-C retracts, one of which ("single-statistic screens") is false against MESAS, SignGuard, and DnC. The
fix is not more literature — it is aligning Section I to Section II-C's own honest framing. The
defensible claim is already written at line 150: *among assumption-light detect-and-filter defenses,
USTAD-FL is the only one simultaneously training-free and purely per-round.*

**On the benign-rejection finding.** We want to be clear that this is not an accusation of concealment —
your Section V-D reports the ≈0.46 warm-up rate openly. What the evaluation could not show you is that
it never goes away, because with N = 100 and full participation the model converges comfortably on 56
clients. A defended-no-attack row would have surfaced it immediately, and adding one converts your most
damaging finding into a disclosed and bounded limitation. It also has a constructive resolution: on this
benchmark the absolute branch alone would have dropped exactly the 40 attackers and zero honest clients,
which suggests the adaptive-quantile machinery is contributing only false positives here.

**On the median argument.** The identity between LIE's supporter condition and the median-capture
condition is, we think, genuinely interesting and worth a subsection rather than a correction. LIE is
calibrated to sit exactly at the boundary where it becomes the median. That is a sharper statement about
why LIE is hard than anything currently in Section III, and it points toward higher-breakdown estimators
as the principled fix.

**On writing.** The prose is clear, well signposted, and consistently honest about uncertainty. The
abstract's numbers all trace to the body. The campaign-separation discipline in Section V-A is exemplary
and we would like to see it adopted more widely. The presentation problems are entirely in the figures,
and they are mechanical: four need regenerating at print size, two need a dataset filter, and one needs
redrawing. None requires new science.

**On resubmission.** The path is clear and most of it is cheap relative to the 66-configuration sweep you
have already run. In rough order of leverage: the defended-no-attack control (free — data on disk), the
de-cloned attack, SignGuard and DnC, the M/N sweep, then seeds. If the de-cloned experiment holds, you
have a considerably stronger paper than the one submitted, and we would look forward to reviewing it.
