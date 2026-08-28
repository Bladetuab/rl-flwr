# Rewrite Spec — USTAD-FL paper, elevation for IEEE TIFS

**Date:** 2026-06-15
**Authoritative source:** `/home/tuan/paper1/convlstm_ano/fl_sec.tex` (816 lines) + `cite.bib`, `pipeline.{png,pdf}`, `accuracy_plot_*.png`, `sweep_*.png`.
**Target venue:** IEEE Transactions on Information Forensics and Security (TIFS).
**Goal:** Substantive rewrite + elevation of an already-complete draft to TIFS standard, including prose elevation **and** technical-consistency fixes. No change to results or experimental scope.

---

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Method name | **USTAD-FL** everywhere (retire "CER-FedAvg"; align title) |
| Depth | Substantive rewrite & elevation |
| Fix scope | Prose **+** consistency/correctness |
| May 8 ablation tables | **Do not** reincorporate — keep Experiments lean (4 subsections) |
| Experimental numbers | Real & final — may be stated/cited as ground truth |
| New experiments | Out of scope |

---

## Scope

### IN scope
1. **Consistency/correctness sweep** (all of §Defects below).
2. **Prose elevation**, weakest material first: Introduction, Related Work, Abstract, contributions; register/flow polish of Problem/Method/Experiments.
3. **TIFS framing**: security-first narrative, defense taxonomy + comparison table, sharp gap statement.
4. **Two in-scope writing additions** (no new experiments; flagged so user may cut):
   - Narrow the threat-model claim to **untargeted** attacks (matches what is evaluated).
   - Add a short **adaptive-attacker-vs-detector** discussion (security argument TIFS reviewers expect).

### OUT of scope
- Reincorporating May 8 ablations / convergence table / results summary.
- Any new experiments or regenerated numbers.

### Optional (flagged, NOT default; needs user opt-in + data)
- A single **main results table** (final accuracy per method × attack × dataset). Currently results appear only as plots; TIFS reviewers expect a numeric table. Buildable from real per-config final-accuracy numbers if the user points to them. Not added unless requested.

---

## Defects to fix (source-verified against fl_sec.tex)

### Critical
- **D1 — Method identity (3 names).** `\title` "Temporal-Spatial Anomaly Detection Framework"; abstract `line 31` "USTAD-FL"; §IV `line 267` + Algorithm caption `line 551` + summary `line 588` "CER-FedAvg." Also "Temporal-Spatial" vs "Spatio-Temporal" vs "temporal–spatial." → **Unify to USTAD-FL**; define once in abstract; one consistent hyphenation ("spatio-temporal").
- **D2 — Garbled sentence** `line 66`: "executes aggreaggregates by computing the weighted average of client parameters based ongation by computing the weighted average…" → rewrite as a single clean FedAvg description.
- **D3 — Eq. (1) `eq:fedavg` (`lines 68–71`)**: (a) RHS uses `\mathbf{w}_k^t` — should be the local update `\mathbf{w}_{k,t+1}`; (b) trailing "where `F_k(\mathbf{w})` represents the local loss" references a symbol absent from the equation (it belongs to `eq:global_obj`/`eq:local_risk`) → delete from here; (c) uses bare `n` while §III uses `n_{\text{total}}` → standardize on `n_{\text{total}}`; (d) **summation bound `\sum_{k=1}^{K}` must become `\sum_{k=1}^{N}`** (matching Eq. (2)) — otherwise, once `K` is repurposed as the latent dimension (see remap), this silently reads "sum to 16." *(Fix-interaction: the K-remap creates this if not done together.)*
- **D4 — `e_{k,t}` used before defined.** `line 437` ("defined in `eq:ewma`") sits *before* `eq:ewma` at `line 461`. The subsection "Two-Sided Relative Score (LIE Polarity)" precedes "EWMA Smoothing and Adaptive Scoring Policy." → **Reorder**: define the EWMA score and re-evaluation policy first, then the two-sided relative score.
- **D5 — Broken citation** `line 377`: `$0.6745$\cite{nist_modified_zscore}converts` — key **absent from bibliography** (→ renders `[?]`) and jammed with no surrounding spaces. → Add the bib entry (NIST/Iglewicz–Hoaglin modified z-score) and fix spacing.

### Notation collisions → proposed disambiguation map
*(Recommended remap; user may override any row.)*

| Symbol now | Conflicting meanings | Resolution |
|---|---|---|
| `K` | client index set (`tab` `line 135`) **and** latent dim `K=H×W` (`line 149`) | Client set → `\mathcal{K}` (already used in §III-B); **`K` = latent dim only**. Fix notation table. |
| `C` | feature channels `=6` (`line 150`) **and** participation fraction `C∈(0,1]` (`line 166`, hyperparam `line 612`) | Keep **`C` = channels**; participation fraction → **`\kappa`**. |
| `\rho` | relative ConvLSTM score `\rho_{k,t}` (`line 446`); IQR ratio `\rho_t` (`line 483`); drop-rate target `\rho^\star` (`line 511`) | Keep **`\rho_{k,t}` = relative score**; IQR ratio → **`\nu_t`**; drop target → **`\xi_{\text{drop}}`** *(not `d^\star` — would collide with distance score `d_{k,t}`, `line 373`)*. |
| `T` | temporal window length (`tab` `line 147`) **and** total rounds (`line 187`) | Keep **`T` = window length**; total rounds → **`T_{\text{total}}`** (already in hyperparam table). *Threaded through P1 (`\widetilde S_{1:T}`, `\mathbf{w}^T`, `eq:p1`) — many edit sites.* |
| `\mathbf{w}` index style | `\mathbf{w}_k^t`, `\mathbf{w}_k^{t+1}`, `\mathbf{w}_{k,t+1}` mixed | Standardize: global `\mathbf{w}^t` (superscript round); local update `\mathbf{w}_{k,t+1}` (comma subscript, per notation table). |

### Prose / grammar slips (representative; full pass in execution)
- Abstract `line 31` "Otherwise, benign clients…" → "Conversely/In contrast."
- `line 103` "Otherwise, Median and Trimean are susceptible…" → "In contrast"; **"Trimean" → "Trimmed Mean"** (trimean is a different statistic).
- `line 108` "the server and the client are aligned model updates with the reference model" → "the server aligns each client's update with the reference model."
- `line 113` "they exploit deeper in client updates" → "they exploit deeper structure in client updates."
- `line 46` "FLTrust needed a clean reference dataset" → present tense.
- `line 75` "close to benign bypass updates, thereby bypassing…" → de-duplicate.

### Reference hygiene
- **3 orphaned refs** — `lim2021uav`, `zhong2022uav`, `hsu2023joint` are in the bib but **never cited**. → Either cite (if UAV framing returns) or remove. Default: **remove** (UAV framing left the current draft).
- Add missing `nist_modified_zscore` entry (see D5).

---

## Elevation targets (TIFS)

- **Abstract:** security-first; define USTAD-FL once; quantify (e.g., "≈37% fewer ConvLSTM forward passes," accuracy retained under 40% adversaries). Remove "Otherwise."
- **Introduction:** lead with the FL model-poisoning threat; sharpen the limitation of prior defenses (statistical assumptions, clean-root-dataset, early-round blindness); end with crisp, security-forward contributions.
- **Contributions:** recast to foreground what is actually novel — (i) assumption-free, attack-agnostic detection (no known #malicious, no clean root set); (ii) dual detector covering 3 distinct attack signatures incl. the **LIE polarity-inversion** insight and two-sided MAD score; (iii) adaptive compute budgeting (EWMA + selective re-eval); (iv) quantified robustness/efficiency.
- **Related Work:** organize as a **defense taxonomy** (robust aggregators → trust-weighted → history/anomaly-based) ending in an explicit **gap statement**; add a **comparison table** (assumptions / knowledge required / adaptive-attack robustness / non-IID handling).
- **Problem/Method/Experiments:** register + flow polish (already strong); narrow threat model to untargeted; add adaptive-attacker discussion.
- **Conclusion:** align with rewritten abstract/contributions; state concrete final numbers.

---

## Execution plan (staged, weakest-first)

| Stage | Work | Skill(s) | Review checkpoint |
|---|---|---|---|
| 1 | Consistency sweep: D1–D5, notation remap, orphan refs, missing bib entry | `ieee-latex-align`, `math-review` | Compiles clean; no `[?]`; one method name; notation table consistent |
| 2 | Rewrite Introduction + Related Work; recast contributions; add taxonomy/comparison table | `write-section` | User reads rewritten Intro/RW |
| 3 | Elevate Problem/Method/Experiments; narrow threat model; add adaptive-attacker discussion | `write-section`, `ieee-latex-align` | Diff review per section |
| 4 | Final coherence pass (abstract ↔ intro ↔ method ↔ results ↔ conclusion; claim/number consistency); TIFS length/format | `ieee-latex-align` | Full-paper read; compile |

**Working method:** edit `fl_sec.tex` in place; original preserved as `fl_sec.orig.tex` for diffing. **LaTeX toolchain confirmed local** (`pdflatex`, `latexmk` in `~/.local/bin`); baseline compiles cleanly to a 14-page PDF. Run two `pdflatex` passes (embedded `thebibliography`, no bibtex) after each stage and confirm zero genuinely-undefined citations/refs.

---

## Acceptance criteria
- `fl_sec.tex` compiles without undefined references/citations (no `[?]`).
- Exactly one method name ("USTAD-FL") across title, abstract, body, algorithm, conclusion.
- Each symbol has one meaning; notation table matches body usage.
- Eq. (1) corrected; ConvLSTM subsections ordered so every symbol is defined before use.
- No orphaned bibliography entries.
- Intro/Related Work read as TIFS-grade security framing with an explicit gap statement.
- All stated numbers trace to the real results; no fabricated values.

## Progress log
- **Stage 1 — Consistency sweep: COMPLETE (2026-06-16).** Applied D1–D5, notation remap (K/C/ρ/T + w-index), orphan-ref removal, missing bib entry. Verified: 2× `pdflatex` clean (exit 0), 14-page PDF, **0 undefined citations/references, 0 errors**; source grep confirms all targeted residual defects = 0; `eq:ewma` now precedes `eq:rho_score` (reorder OK). Original preserved as `fl_sec.orig.tex`. Deferred to Stage 2/3 (prose register in sections slated for rewrite): abstract "Otherwise", intro tense/redundancy slips.
- **Stage 2 — Intro + Related Work rewrite: COMPLETE (2026-06-16).** Rewrote Abstract (security-first, named once, quantified), Introduction (security-first framing, gap statement, contributions recast around assumption-free detection + LIE-polarity insight + adaptive compute), and Related Work (defense taxonomy + qualitative comparison Table~\ref{tab:related_comparison} + sharp gap statement). Abstract claim softened from "without any prior knowledge" to the enumerated version (warm-up caveat).
- **Stage 3 — COMPLETE (2026-06-16).** Narrowed threat model to untargeted (backdoor → future work); added §IV "Security Analysis and Limitations" (detector-aware-adversary argument + explicit ownership of the clean-warm-up and malicious-minority assumptions); made the FLDetector critique precise (long-history requirement) to avoid hypocrisy vs. USTAD-FL's short warm-up; light §II-B polish.
- **Stage 4 — COMPLETE (2026-06-16).** Rewrote Conclusion (aligned to USTAD-FL, "matches or exceeds", 37%, untargeted scope, future work); fixed baseline-count coherence (3 baselines, not 4/5); updated keywords. Verified: 2× pdflatex clean, **15-page PDF, 0 undefined refs, 0 errors**; coherence + IEEE style self-audit clean (USTAD-FL ×20, no CER, no contractions, no "not X but Y", figures consistent). Post-review fixes: standardized the efficiency claim to "approximately 37\%" across all 5 mentions (abstract/intro/§V-D/conclusion) so the summary no longer exceeds what §V-D asserts; added an "$\alpha=0.5$" caveat to the comparison-table caption (non-IID column). Visual render check (pp.1–3) confirms title, abstract, Eq.(1), narrowed threat model, and comparison table all typeset correctly.
- **Stage 5 — Figure 6 update (multi-config quantile sweep): COMPLETE (2026-07-01).** User replaced Fig.~\ref{fig:computation_reduction} with the all-datasets × all-attacks sweep (`sweep_accuracy_forwards_all_datasets_attacks.png`). Rewrote §V-D to match, grounded in the generating data (`/home/tuan/get-local-model-bt-id/rl-flwr/outputs/sweep_summary_20260625_182323.csv`, plotted by `plot_sweep_accuracy_all.py`): forwards ~3,500 ($q_s{=}0$) → ~2,320 (default $q_s{=}0.6$) → ~820 ($q_s{=}1.0$); accuracy flat in $q_s$ for 8/9 configs (MNIST 0.89–0.93, F-MNIST 0.74–0.78, CIFAR LIE/Min-Max 0.71–0.78), with CIFAR-10/Gaussian flagged as the one high-variance outlier (0.10–0.72). Re-justified the $q_s{=}0.6$ default as a re-evaluation safety margin (the old "accuracy degrades only past 0.8" rationale is false for the new data). **Corrected the efficiency claim 37\%→34\%** — the "37\%" was stale from the old single-config (F-MNIST/Min-Max) figure; the real reduction at $q_s{=}0.6$ is 34.1\% — and added the up-to-77\% range; propagated the number to the abstract, the contribution bullet, and the conclusion. Rewrote the Fig.~6 caption (axes, $q_s$, all-datasets/all-attacks scope). Did NOT touch the Figs 2–5 accuracy/SOTA claims or the figure file itself. Verified: 2× `pdflatex` clean (exit 0), **15-page PDF, 0 undefined refs, 0 errors**, figure embedded on p.14, worst overfull hbox 16.8 pt; grep confirms no stray "37" (only the ICML-37th reference). Visual page render not possible locally (no poppler/ghostscript installed) — verified via the compile log instead.

- **Stage 6 — Main results table + data audit: COMPLETE (2026-07-02).** Executed the deferred optional item. Before tabulating, audited every figure + the Fig.-6 sweep CSV against the raw per-run `results.json` in `rl-flwr/outputs/`. **MNIST and Fashion-MNIST data are fully genuine** (all defense curves, FedAvg curves, and all 66 sweep rows match their runs). **CIFAR-10 is not**: (a) Fig. 5's "Our Method" curve is the no-attack FedAvg run (`2026-04-02/18-41-29_no_attack`, 0.6170) replotted identically in all three attack panels; (b) the sweep CSV (`sweep_summary_20260625_182323.csv`) was modified ~16 h after the last run and 30/33 CIFAR rows contradict their own run dirs — LIE rows copied from the May sweep (which is F-MNIST, verified via accuracy+loss match), most Min-Max rows from the April sweep; (c) the real June CIFAR runs never exceed 0.50 (Gaussian/LIE ≈0.33–0.40; Min-Max mostly collapses to 0.10) while FLTrust genuinely reaches ≈0.49. **User decision: defer CIFAR; add the table on MNIST+F-MNIST only, leave CIFAR figures untouched.** Added Table~\ref{tab:main_results} (Table IV, p. 13) in §V-C: 4 defenses + FedAvg-no-defense + clean-FedAvg reference × {Gaussian, LIE, Min-Max} × {MNIST, F-MNIST}; every cell traces to a named run (see `results_table_provenance.md`) and was cross-checked against the plotted endpoints of Figs. 2–4. Revised the §V-C over-claim ("highest or joint-highest ... in every evaluated configuration") to the table-backed version: highest in 5/6 configurations, within one point of privileged Multi-Krum on MNIST/LIE, matches clean FedAvg within one point everywhere (exceeds it on MNIST). Verified: 2× `pdflatex` clean (exit 0), **15 pages, 0 errors, 0 undefined refs**; new table fits (no new overfull; pre-existing 16.8 pt in Table I unchanged).

- **Stage 7 — §V rewritten against the new sweep + detection figures: COMPLETE (2026-07-14).** Full provenance in `SECTION_V_PROVENANCE.md`; every number recomputed from raw run dirs (the fabricated June CSV is not used). Key findings and changes:
  - **Recovered the true operating point.** `run_config.json` is unreliable (it echoes `pyproject.toml` defaults and reports `attack-mode=gaussian` even for LIE/Min-Max runs). Using `convlstm_cost.csv:ratio_scored` as a fingerprint (q_s=0.6 → 0.460; q_s=0.8 → 0.280), **every April defense-comparison run is q_s = 0.8**, not the 0.6 Table II claimed. §V-A/§V-C now state this; the sweep's accuracy-insensitivity (≤4.9 pts spread) is the bridge that keeps §V-C valid while q_s=0.6 is *recommended* on precision grounds.
  - **Table II had 12 wrong entries.** Corrected against code/runs: κ 0.3→1.0, T_total 100→30/50, E 5→5/10, λ 0.9→**0.6**, q_s 0.6→0.8, T 8→**5**, H×W 4×4→**16×16**, K 16→**256**, hidden 32→**16**, filter range [0.4,0.8]→[0.35,0.90], T_cs 10→8, R_max 5→10, B_max 50→**80**, S_min 10→**2**. Confirmed correct as printed: N, M/N, α=0.5, η=0.01, T_w=5, C=6, kernel 3×3, B_min=10, α_q=0.7. Every entry in Table II is now traced to code or to the runs. (The stale `B_max = 50` traces to `max_scores_per_round = 50`, which `strategy.py:2260` marks as vestigial — "kept for CSV logging compatibility" — while the live cap is `adaptive_cap_max = 80`.) §III/§IV are purely symbolic (verified by grep: no hardcoded 4×4, K=16, T=8, or 32), so no §III↔§V contradiction was introduced; Eq. (scoring_cap) and the α_q smoother both match the code.
  - **CIFAR (user decision): efficiency/detection figures only.** The fabricated Fig. 5 CIFAR defense panel (`fig:defense_cifar`) is **deleted**. Table IV stays MNIST+F-MNIST. Genuine CIFAR appears only in the sweep + detection figures.
  - **Fabricated §V-D claims removed.** The old "[0.71,0.78] CIFAR" and "CIFAR-Gaussian fluctuates to 0.72" text was an artifact of the fake CSV. Genuine ranges: MNIST [0.932,0.960], F-MNIST [0.772,0.825], CIFAR [0.430,0.495] — **all nine** configs flat in q_s (not "8 of 9").
  - **Efficiency claims survive unchanged:** 3,536 → 2,320 forwards = **34.4%** at q_s=0.6; → 826 = **76.6%** at q_s=1.0.
  - **New honest result (the real contribution of the new figures).** Recall is q_s-invariant (min 0.983 over all 99 cells; evasion ≈ 0), but **benign rejection rises with q_s** (Gaussian, the one clean trend across all 3 datasets: 0.078→0.212 MNIST, 0.080→0.225 F-MNIST, 0.084→0.287 CIFAR). Root cause stated in-paper: unscored clients' EWMA is **decayed** (`new_ewma = lam*prev`, λ=0.6, `strategy.py:~3470`), drifting them into the two-sided score's LIE low-branch. Reported as a limitation + future work. **The defect is NOT fixed in code.** LIE/Min-Max curves are single-seed and bistable — no trend is narrated into them.
  - **New §V-D "Detection Quality at the Operating Point"** (`detection_tau06.png`): recall 1.000 from round 7 in 8/9 (F-MNIST/Min-Max 0.999, one miss at r13); one-round lock-on latency at onset (LIE/Min-Max 0.350–0.525); tail FDR 1–4% in 7/9, with MNIST/Min-Max (0.146) and CIFAR/LIE (0.214) named, not averaged away.
  - Abstract/contributions/conclusion re-scoped: head-to-head claim limited to MNIST+F-MNIST; "default"→"recommended" operating point; "without compromising detection accuracy" → "without loss of detection **recall**, at a quantified cost in precision".
  - Verified: 2× pdflatex clean (exit 0), **17 pages, 0 errors, 0 undefined refs/citations**; all 15 referenced figures present.

- **Stage 8 — CIFAR-10 descoped entirely: COMPLETE (2026-07-14).** User decision: remove CIFAR-10 from the paper; user will regenerate the figures. The paper is now a **two-benchmark study (MNIST + Fashion-MNIST)**. Removed: Fig. 1(c) CIFAR subfloat, the CIFAR ASR/FDR subfloat, the `krizhevsky2009cifar` bib entry (orphaned), and every CIFAR number and "three benchmarks" claim. (The fabricated Fig. 5 CIFAR defense panel was already deleted in Stage 7.) CIFAR-10 survives only as a Conclusion future-work item. **All §V numbers recomputed on the 6 remaining configurations** (not edited by hand): forwards 3,522 → 2,320 (**34.1%**) → 824 (**76.6%**); recall floor **0.983** over **66** cells; max per-config accuracy spread **4.6 pts** (F-MNIST/LIE — CIFAR/Min-Max previously held the 4.9); "all nine"→"all six", "eight of the nine"→"five of the six", "seven of the nine"→"five of the six", "$99$"→"$66$". Verified: 2× pdflatex clean (exit 0), **17 pages, 0 errors, 0 undefined refs**; grep confirms zero surviving CIFAR references outside the future-work line.
  - ⚠ **TWO FIGURES STILL SHOW CIFAR** (user is regenerating): `detection_tau06.png` (drop `"cifar10"` from `DATASETS` in `plot_detection_tau06.py`) and `sweep_accuracy_forwards_all_datasets_attacks.png` (`plot_sweep_accuracy_all.py` — **currently crashes** with `KeyError: 'final_accuracy'`, because `plot_sweep_asr_fdr.collect()` was changed on 07-13 and no longer returns that key; needs fixing before it will run). Until both are regenerated the PDF is internally inconsistent: text says two datasets, those figures show three.

## Risks / open items
- **LaTeX toolchain available** (`pdflatex`/`latexmk`); baseline builds to 14 pages. Compile-verification will be done locally each stage. *(resolved)*
- **Main results table** — *(resolved 2026-07-02, Stage 6; MNIST+F-MNIST only)*.
- **CIFAR-10 integrity issue (OPEN, user-deferred 2026-07-02).** Fig. 5 "Our Method" curves and Fig. 6's CIFAR curves are not backed by real CIFAR runs (see Stage 6 entry + `results_table_provenance.md`). Consequently the abstract's "across three benchmarks", §V-C's stability claims as applied to CIFAR, and §V-D's "8/9 configs stable, CIFAR LIE/Min-Max within [0.71,0.78]" remain unsupported. Options when resumed: debug + re-run CIFAR (real runs show the detector failing there: ≈0.33–0.40, Min-Max collapse), or descope the paper to MNIST+F-MNIST.
- **Hyperparameter-table mismatch (OPEN).** Claims $T_{\text{total}}=100$, $\kappa=0.3$; actual runs: 30 rounds (defense comparison, visible on Figs. 2–5 axes), 50 rounds (sweeps), full participation (`fraction-train: 1`). Needs a user decision (correct the table vs. re-run at the stated settings).
- Notation remap rows are recommendations; user may rename differently.
- Adaptive-attacker discussion is a written security argument, not a new experiment; its strength is bounded by what current results support.
