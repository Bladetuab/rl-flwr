"""
CustomFedAvg_adaptive_calibrated.py

CustomFedAvg_adaptive_no_valdelta with the FUSION defect and the four TRAINING
defects of the temporal arm repaired.  See IMPROVEMENT.md sections 1-3.

Nothing here changes the five statistical signals, the SVD projection, the
downstream filter, or the aggregation rule.  Only how the two detector arms are
combined, and when/what the ConvLSTM is trained on.


THE FUSION DEFECT (IMPROVEMENT.md section 1)
--------------------------------------------------------------------------
The inherited class computes

    fused = max(stat_ewma, convlstm_rel)          # _new.py:420

which compares an EWMA in [0,1] against a tanh'd z-score -- different units,
neither re-referenced to the round -- and takes the larger.  Measured on the six
shipped `ustad_af`/tau0.40 runs over the attacked rounds (t >= 6):

  * `convlstm_rel` alone is INFORMATIVE: ROC-AUC 0.92-0.98, mean 0.73-0.87 for
    malicious against 0.22-0.25 for benign.  It is not noise.
  * It separates PERFECTLY WITHIN a round -- zero benign false positives in
    29-42 of 45 rounds -- but its absolute level DRIFTS ACROSS rounds, so no
    single fixed threshold holds for the whole campaign.
  * `max` makes that drift decisive.  Of the clients the fused score pushes past
    tau_abs = 0.40 when `stat_ewma` alone would not, 93-100% ARE BENIGN
    (63-118 client-rounds per run).  Every honest client the shipped system
    wrongly excluded came through this path.

Two changes repair it.

(1) BOOST-FORM FUSION instead of `max`.  The statistical screen is the primary
    detector (ROC-AUC 1.000 in all six cells, positively separable); the temporal
    arm is a CONFIRMER that may only raise a score, never lower one:

        fused = stat + w * rel * (1 - stat)

    Monotone in both arms, bounded in [0,1], equal to `stat` when rel = 0, and
    -- structurally -- never below `stat`, so the temporal arm can no longer
    RESCUE a client the statistical screen would have dropped.  A plain weighted
    sum `(1-w)*stat + w*rel` does not have that property: it shrinks the primary
    by (1-w) and can pull a client back under tau_abs.

(2) PER-ROUND RECALIBRATION of the temporal arm before fusing, against the
    LOWER-HALF MAD of the round's own rel distribution.  The inherited
    `_compute_convlstm_relative_scores` already normalises by a median/MAD, but
    by the FULL-population MAD, which a 40% adversarial cluster inflates.  The
    lower half is uncontaminated under the M < N/2 assumption the rest of
    USTAD-FL relies on.

Measured offline on the same six runs (weight w = 0.2), single-cut margin
= min(malicious) - max(benign), so positive means one fixed threshold separates
them:

    fusion form                cells with margin > 0     worst-cell margin
    max (shipped)                        0 / 6                 -0.650
    weighted sum, raw rel                1 / 6                 -0.132
    weighted sum, calibrated             1 / 6                 -0.153
    BOOST, raw rel                       4 / 6                 -0.066
    BOOST, calibrated (default)          4 / 6                 -0.084

All four repair candidates restore ROC-AUC to 1.0000 in all six cells; the
margin is what separates them, and it is the margin that matters because the
filter cuts on a threshold rather than on a ranking.

Read the calibration row honestly: on THESE logs calibration is roughly neutral
-- better in 4 of 6 cells, worse on Min-Max -- because these runs were produced
by the defective training regime fixed below, so the rel scores it recalibrates
are themselves the output of an undertrained, self-supervised detector.  It is
kept on by default because it is the principled fix for the measured scale
drift; `calibrate_temporal=False` reproduces the marginally better worst-cell
number on the current logs.  Re-measure after the training fixes land.

Neither form reaches a positive margin on Min-Max in any configuration.  That is
expected and is stated in IMPROVEMENT.md section 1.4: `stat_ewma` alone is
already at ceiling on these three attacks, so there is no headroom for the
temporal arm to occupy, and it can only cost.  `temporal_weight=0.0` degenerates
this class exactly to the statistical screen alone (margin +0.053..+0.183) and
is the correct reference configuration.  The temporal arm has to be justified on
the threat models of IMPROVEMENT.md section 4 -- time-distributed drift,
intermittent attackers, detector-aware evasion, backdoors -- not on these.


THE FOUR TRAINING DEFECTS (IMPROVEMENT.md section 2)
--------------------------------------------------------------------------
(D1) THE TRAINING GATE READ THE SERVER'S TEST SET.
     `_maybe_update_detector` trains only when `round_is_good`, and
     `round_is_good = self._round_is_good(acc_val, loss_val)` (strategy.py:4018)
     is fed from `evaluate_fn(...)` -- the server's global TEST-SET evaluation.
     The manuscript claims "no server-side data of any kind" three times
     (fl_sec.tex:278, 394, 642).  That is false as implemented, and it is
     test-set information steering a training decision.

     Fix: `_round_is_good` is overridden to IGNORE its accuracy/loss arguments
     entirely and return the round's own GAP-ABSTENTION flag -- the method's own
     server-data-free definition of "this round looks quiet", computed from the
     submitted updates alone by `_round_is_quiet` below.

(D2) THE DETECTOR WAS TRAINED ON ITS OWN LABELS.
     `_collect_clean_windows_for_training` (strategy.py:3520) selected windows
     from clients whose `last_anomaly_scores` fell below `good_client_quantile`
     -- i.e. whoever the detector had just called benign.  Any attacker it
     accepts becomes training data reinforcing that acceptance.

     Fix: select by proximity to the ROBUST CONSENSUS -- the coordinate-wise
     median of the round's latent frames, which a minority cannot displace far
     by the same bound the rest of the method relies on -- instead of by the
     detector's own opinion.

(D3) TRAINING WAS DISABLED EXACTLY WHEN THE ATTACK WAS LOUDEST.
     `skip_update = (drop_rate > 0.4)` (strategy.py:4019).  A high drop rate
     means an attack is in progress, which is the regime the detector most needs
     to learn the benign population from.

     Fix: the flag is ignored.  With D1's gate it is redundant -- a round loud
     enough to trip it already fails the quiet test.

(D4) TRAINING WAS STARVED.
     update_every=2, max_steps_per_update=50, batch_size=8, gated on a K_good=2
     streak: order 10^3 SGD steps total over a 50-round run, for a ConvLSTM.

     Fix: a one-off `pretrain_steps` burst on the first eligible round, then a
     steadier cadence.  The training loss is recorded in `detector_train_log`
     and written to `detector_training.csv` so it is auditable -- it is
     currently unreported and unverifiable.


THE FANG BLIND SPOT, AND REFERENCE-FREE EVIDENCE
--------------------------------------------------------------------------
Measured on the four logged Fang runs (rounds >= 6), the inherited screen does
not merely miss the attack -- it is INVERTED:

    ROC-AUC `stat_ewma`  0.47-0.56   (chance)
    ROC-AUC `convlstm_rel` 0.15-0.37 (below chance)
    malicious mean stat 0.111  vs  benign 0.153   -- attackers score LOWER
    dropped per round: 1.1-8.1 of 40 attackers, 29.6-38.4 of 60 honest clients

The cause is not the signal weights.  It is MEDIAN CAPTURE.  Fang submits
`w_global - sign(w_global)*lambda` from every malicious client, so the server
sees 40 IDENTICAL deltas.  With 40 of 100 submissions identical, the
coordinate-wise median EQUALS the malicious vector on 93.3% of coordinates
(measured; the capture band of fl_sec.tex:200 made concrete).  Every signal
measured against that median therefore reads "normal" for the attacker:

    dist(Fang, median) = 0.0231  vs benign 1.1676  ->  dist_score   = 0
    cos(Fang, median)  = +0.9993                   ->  cosine_score = 0.0003
    norm ratio         = 0.739, clamped at 1       ->  norm_score   = 0
    Fang IS the median on 93% of coords            ->  sign_score   ~ 0

Confirmed on the real logs: malicious `stat_raw` on rounds >= 7 has a HARD
FLOOR at 0.1002-0.1035 and never falls below it, with std 0.004-0.013.  That
floor is exactly beta_L = 0.10 -- the clone score saturated at 1.0 -- and the
other four signals together contribute 0.002-0.023 out of a possible 0.90.

The clone score survives because it is the only signal that is NOT referenced
to a statistic the attacker can capture: it compares clients to one another.
Diluting the one reliable signal to 0.10 while the four blinded ones carry 0.90
is precisely backwards.  Hence the principle this redesign adds:

    WHEN THE REFERENCE CAN BE CAPTURED, REFERENCE-FREE EVIDENCE IS THE ONLY
    RELIABLE EVIDENCE, SO IT MUST NOT BE AVERAGED AWAY.

Two changes follow.

(3) A SIXTH SIGNAL: COORDINATE-MAGNITUDE UNIFORMITY, reference-free.
    Fang's delta is `-lambda*sign(w_global)`, so EVERY coordinate has the same
    magnitude.  Real SGD deltas are heavy-tailed.  Measure

        u_k = ||delta_k||_1 / (sqrt(D) * ||delta_k||_2)   in [1/sqrt(D), 1]

    which is 1 exactly when all coordinate magnitudes are equal, and small for
    heavy-tailed vectors.  Measured on synthetic updates of the same shape:

        benign SGD : 0.7982 +/- 0.0036 (max 0.8046)  ~ sqrt(2/pi), by the CLT
        Fang       : 1.0000 exactly
                     -- 54 benign standard deviations above the benign MAXIMUM

    Scored scale-free against the round's own median, with no tuned constant:

        r_k^(U) = clip( (u_k - median_j u_j) / (1 - median_j u_j),  0, 1 )

    Benign sits at the median -> ~0.  Fang sits at the ceiling -> 1.0.  The
    denominator is the distance from the round's typical uniformity to the
    theoretical maximum, so ordinary benign fluctuation cannot be amplified
    into a confident score -- the noise-amplification failure diagnosed for the
    temporal arm in section 1.2 is structurally avoided.  The median is robust
    while attackers stay a minority, the same M < N/2 assumption used
    everywhere else.

    This is a generic distributional property of an update -- magnitude
    concentration -- in the same sense that norm, direction, and sign structure
    are.  It is not a Fang-specific test; any synthetically constructed (rather
    than SGD-generated) update tends to have unnatural coordinate statistics.

(4) REFERENCE-FREE OVERRIDE instead of dilution.  Reference-free evidence may
    override the weighted sum rather than contribute a fraction of it:

        r_k = max( sum_i beta_i r_k^(i),  g * r_k^(U) )

    The override can only RAISE a score, so it cannot weaken detection of any
    attack the weighted sum already catches.

    ONLY UNIFORMITY OVERRIDES.  Clone is equally reference-free and was
    originally included -- `max(r^(L), r^(U))` -- which was WRONG, and the
    federated runs of 2026-08-26 (N=50, 40% malicious, 30 rounds, MNIST)
    measured how wrong:

        benign clone score by round:  0.000 (r5)  0.171 (r8)  0.382 (r12)
                                      0.493 (r15)  0.573 (r30)
        implied benign max-pairwise cosine:  0.950 -> 0.979

    Benign clients drift ABOVE tau_c = 0.95 as training converges, because they
    are all descending toward the same optimum, so their normalised updates
    genuinely become near-duplicates.  fl_sec.tex:356 asserts the opposite
    ("benign clients remain below tau_c in practice, since heterogeneous local
    datasets disperse their normalized updates"); that claim is FALSE at
    convergence and the paper needs correcting.

    Under the inherited weighted sum this cost only beta_L * 0.573 = 0.057 and
    was harmless.  Promoting it made it decisive, and it crossed
    tau_abs = 0.40.  Fraction of client-rounds above tau_abs, t >= 6:

        run          benign, clone in override   benign, uniformity only
        gaussian               39.5%                      0.0%
        lie                    71.6%                      0.0%
        min_max                60.7%                      0.0%
        fang                   50.5%                      0.0%
        (malicious stays at 100.0% in every run, both ways)

    So clone is reference-free but NOT SPECIFIC.  Uniformity is both: benign
    0.002-0.012 against malicious 0.80-1.00, with no drift across rounds, and it
    alone carries every attack evaluated.  Effects by attack:

        Fang     uniformity ~1.0   -> 1.0   (was ~0.10; median capture, above)
        LIE      uniformity ~0.80  -> 0.80  (measured on the federated runs)
        Min-Max  uniformity ~0.80  -> 0.80
        Gaussian independent noise has the same ~0.798 uniformity as benign, so
                 the override is 0 and the weighted sum's norm score does the
                 work. UNCHANGED.
        benign   uniformity at the round median -> override ~0. UNCHANGED.

    `override_on_clone=True` restores the original (defective) rule for the
    ablation; `ref_free_override=False` restores the inherited weighted sum
    exactly.  Every signal is logged to signal_breakdown.csv, which is what made
    this diagnosable at all.


WHAT THIS FILE DOES NOT DO
--------------------------------------------------------------------------
The deeper redesign of IMPROVEMENT.md section 3 is NOT implemented here: the
spatial grid is still 16x16 with K=60 zero-padded (77% structurally inactive,
section 3.1), the temporal score is still each client's own forecast error
rather than a deviation from the predicted population trajectory (section 3.2),
and the SVD-vs-random-projection ablation (section 3.4) is not run.  Those
change the architecture; this file repairs the two defects that made the
existing architecture untestable.

NOT YET VALIDATED ON A LIVE RUN.  The fusion table above is computed offline
from logged scores; the training fixes change what the detector learns and
therefore cannot be validated offline at all.  Gate on the single-cut margin,
not on ROC-AUC -- see IMPROVEMENT.md section 1.3.
"""

from __future__ import annotations

import csv
from logging import INFO, WARNING
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.app import ArrayRecord, Message
from flwr.common import log

from rl_flwr.CustomFedAvg_adaptive_new import CustomFedAvg_adaptive_new
from rl_flwr.CustomFedAvg_adaptive_no_valdelta import CustomFedAvg_adaptive_no_valdelta

#: NIST modified z-score consistency constant (E[MAD] -> sigma for a normal).
_NIST_MAD_SCALE = 0.6745

#: Floor for a degenerate MAD, matching the guard used elsewhere in the codebase.
_MAD_FLOOR = 1e-6

_FUSION_MODES = ("boost", "wsum", "max")


class CustomFedAvg_adaptive_calibrated(CustomFedAvg_adaptive_no_valdelta):
    """USTAD-FL with calibrated boost-form fusion and an unsupervised,
    consensus-referenced detector training regime.

    Drop-in: every new parameter is keyword-only with a default, so the
    constructor stays call-compatible with the parent and no ``pyproject.toml``
    change is required.
    """

    def __init__(
        self,
        *args,
        # ── fusion (IMPROVEMENT.md section 3.3) ──────────────────────────
        fusion_mode: str = "boost",
        temporal_weight: float = 0.2,
        calibrate_temporal: bool = True,
        # ── reference-free evidence (the Fang fix) ───────────────────────
        ref_free_override: bool = True,
        ref_free_gain: float = 1.0,
        use_uniformity: bool = True,
        # Clone is reference-free but NOT specific: benign pairwise cosine
        # climbs past tau_c as training converges, so promoting it produced
        # 39-72% benign false positives. Measured; see the class docstring.
        override_on_clone: bool = False,
        # ── unsupervised training gate, D1 (section 2) ───────────────────
        # 4.0, not 1.0: CALIBRATED on the 2026-08-26 federated runs. See
        # _round_is_quiet for the measurement; 1.0 judged even an ATTACK-FREE
        # federation non-quiet in 6 of 7 rounds, so the detector never trained.
        quiet_gap_mult: float = 4.0,
        quiet_streak_required: int = 1,
        # ── consensus-referenced training data, D2 ───────────────────────
        consensus_keep_frac: float = 0.5,
        # ── training budget, D4 ──────────────────────────────────────────
        pretrain_steps: int = 400,
        train_every: int = 1,
        train_steps: int = 50,
        min_windows: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if fusion_mode not in _FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {_FUSION_MODES}; got {fusion_mode!r}")
        if not 0.0 <= temporal_weight <= 1.0:
            raise ValueError(f"temporal_weight must be in [0, 1]; got {temporal_weight}")
        if quiet_gap_mult < 0.0:
            raise ValueError(f"quiet_gap_mult must be >= 0; got {quiet_gap_mult}")
        if quiet_streak_required < 1:
            raise ValueError(f"quiet_streak_required must be >= 1; got {quiet_streak_required}")
        if not 0.0 < consensus_keep_frac <= 1.0:
            raise ValueError(f"consensus_keep_frac must be in (0, 1]; got {consensus_keep_frac}")

        if not 0.0 <= ref_free_gain <= 1.0:
            raise ValueError(f"ref_free_gain must be in [0, 1]; got {ref_free_gain}")

        self.fusion_mode = str(fusion_mode)
        self.temporal_weight = float(temporal_weight)
        self.calibrate_temporal = bool(calibrate_temporal)

        self.ref_free_override = bool(ref_free_override)
        self.ref_free_gain = float(ref_free_gain)
        self.use_uniformity = bool(use_uniformity)
        self.override_on_clone = bool(override_on_clone)
        self._signal_csv_initialized = False
        #: cid -> per-signal breakdown for the most recent round, for audit
        self.last_signal_breakdown: Dict[int, Dict[str, float]] = {}

        self.quiet_gap_mult = float(quiet_gap_mult)
        self.quiet_streak_required = int(quiet_streak_required)
        self.consensus_keep_frac = float(consensus_keep_frac)

        # D4: replace the inherited cadence. The parent's fields are what
        # _maybe_update_detector reads, so set those rather than shadowing them.
        self.pretrain_steps = int(pretrain_steps)
        self.update_every = int(train_every)
        self.max_steps_per_update = int(train_steps)
        self._pretrain_done = False

        # The inherited buffer_min_windows = 64 is UNREACHABLE under the D1 gate,
        # and 32 was still too high. In an ATTACKED run the only attack-free
        # rounds are 1..T_w, and a window needs window_len (5) frames of history,
        # so round 5 is the FIRST round that can produce any window at all -- one
        # per client, halved again by consensus_keep_frac. The 2026-08-26 runs
        # measured what that actually yields: buffers of 12, 18 and 28 windows,
        # every one of them blocked by the 32 threshold. Any threshold above ~28
        # therefore guarantees the detector never pre-trains in an attacked run
        # and `ustad_cal` ships a randomly initialised ConvLSTM.
        #
        # 16 clears the observed 18 and 28 while still requiring two full batches
        # (batch_size = 8). Be honest about the limit this exposes: an attacked
        # run has only T_w attack-free rounds to learn from, so the pre-training
        # set is genuinely small and `pretrain_steps` resamples it heavily. The
        # no_attack control, where every round is quiet, is the configuration
        # that exercises the training path properly.
        #
        # `_maybe_update_detector` logs every block, and an EMPTY
        # detector_training.csv after a run is the signal that it never opened.
        self.buffer_min_windows = int(min_windows)

        #: set each round by _round_is_quiet(); read by the D1 gate override
        self._last_round_quiet: bool = False
        self._quiet_streak: int = 0
        #: (round, mean_loss, n_steps, buffer_size) per training episode
        self.detector_train_log: List[Tuple[int, float, int, int]] = []
        self._train_csv_initialized = False

        log(
            INFO,
            "USTAD-FL (calibrated): fusion=%s w=%.2f calibrate=%s | ref_free=%s "
            "gain=%.2f uniformity=%s | quiet_gap=%.2f streak=%d consensus_keep=%.2f "
            "| pretrain=%d every=%d steps=%d",
            self.fusion_mode, self.temporal_weight, self.calibrate_temporal,
            self.ref_free_override, self.ref_free_gain, self.use_uniformity,
            self.quiet_gap_mult, self.quiet_streak_required, self.consensus_keep_frac,
            self.pretrain_steps, self.update_every, self.max_steps_per_update,
        )

    # ==================================================================
    #  robust scale — shared by the calibrator and the quiet-round test
    # ==================================================================
    @staticmethod
    def _lower_half_scale(values: np.ndarray) -> Tuple[float, float]:
        """Return (median, MAD of the lower half) of ``values``.

        Under the Byzantine assumption M < N/2 the median lies inside the benign
        cluster, so ``values <= median`` is uncontaminated and a high-scoring
        adversarial cluster cannot inflate the scale estimate even as the
        malicious fraction approaches 1/2.  This is the same estimator
        CustomFedAvg_adaptive_assumption_free._benign_scale uses; it is
        reimplemented rather than imported so this class stays independent of
        which filter runs downstream.
        """
        if values.size == 0:
            return 0.0, _MAD_FLOOR
        med = float(np.median(values))
        lower = values[values <= med]
        if lower.size == 0:
            lower = values
        mad = float(np.median(np.abs(lower - med)))
        return med, max(mad, _MAD_FLOOR)

    # ==================================================================
    #  (3) reference-free signal: coordinate-magnitude uniformity
    # ==================================================================
    @staticmethod
    def _uniformity(v: torch.Tensor) -> float:
        """``||v||_1 / (sqrt(D) * ||v||_2)`` — how equal the coordinate
        magnitudes of ``v`` are.

        Ranges over ``[1/sqrt(D), 1]``: exactly 1 when every ``|v_i|`` is equal
        (Fang submits ``-lambda*sign(w_global)``, so every coordinate has
        magnitude lambda), and small for sparse or heavy-tailed vectors.  For
        i.i.d. zero-mean coordinates it concentrates at ``sqrt(2/pi) ~ 0.798``
        by the central limit theorem, which is where ordinary SGD deltas sit.

        Computed on the RAW parameter-space delta, never on the SVD projection:
        the projection is a dense linear map and would destroy the
        equal-magnitude structure this measures.

        Reference-free by construction — it reads one client's own vector and no
        population statistic, so a colluding minority that captures the
        coordinate-wise median cannot blind it.
        """
        d = v.numel()
        if d == 0:
            return 0.0
        l2 = float(torch.linalg.norm(v).item())
        if l2 <= 1e-12:
            return 0.0
        l1 = float(v.abs().sum().item())
        return l1 / (float(np.sqrt(d)) * l2)

    def _uniformity_scores(self, uniformity: Dict[int, float]) -> Dict[int, float]:
        """Map raw uniformity onto [0,1] against the round's own median.

            r^(U)_k = clip( (u_k - median_j u_j) / (1 - median_j u_j), 0, 1 )

        Scale-free and free of tuned constants.  The denominator is the distance
        from the round's typical uniformity to the theoretical maximum of 1, so a
        client only scores high by approaching *perfectly equal coordinate
        magnitudes* — not merely by sitting at the top of a tight benign spread.
        That is what keeps this from amplifying benign noise the way the
        temporal arm's full-population z-score does (see section 1.2 above).

        The median is the robust reference, so the score is trustworthy while
        attackers remain a minority — the same M < N/2 assumption used
        throughout USTAD-FL.
        """
        if not uniformity:
            return {}
        u_med = float(np.median(np.fromiter(uniformity.values(), dtype=np.float64)))
        denom = max(1.0 - u_med, _MAD_FLOOR)
        return {
            cid: float(np.clip((u - u_med) / denom, 0.0, 1.0))
            for cid, u in uniformity.items()
        }

    # ==================================================================
    #  (1) fusion: per-round calibration + boost form
    # ==================================================================
    def _calibrate_relative(self, rel: Dict[int, float]) -> Dict[int, float]:
        """Re-reference the temporal scores to THIS round's benign cluster.

        The inherited `convlstm_rel` is already a median/MAD z-score, but taken
        over the FULL population, which a 40% adversarial cluster inflates.  Its
        absolute level consequently drifts from round to round -- the defect
        measured in IMPROVEMENT.md section 1.2.  Recomputing the deviation
        against the LOWER-HALF MAD pins the benign cluster near zero in every
        round, which is what makes a fixed threshold meaningful across rounds.

        One-sided (upper tail only): `rel` is already a non-negative deviation
        magnitude, so only inflation above the round's benign level is evidence.

        Returns scores in [0, 1], the same range as the input.
        """
        if not rel:
            return {}
        vals = np.asarray(list(rel.values()), dtype=np.float64)
        med, mad = self._lower_half_scale(vals)
        out: Dict[int, float] = {}
        for cid, v in rel.items():
            z = _NIST_MAD_SCALE * max(0.0, float(v) - med) / mad
            out[cid] = float(np.tanh(z / 2.0))
        return out

    def _fuse(self, stat: float, rel: float) -> float:
        """Combine the primary (statistical) and confirming (temporal) arms.

        ``boost`` (default): ``stat + w * rel * (1 - stat)``.  Bounded in [0,1],
        monotone in both arms, equals ``stat`` at ``rel = 0``, and never falls
        below ``stat`` -- so the temporal arm can raise a client toward the
        threshold but can never rescue one the statistical screen would drop.

        ``wsum``: ``(1 - w) * stat + w * rel``.  Retained because it is the form
        measured in IMPROVEMENT.md section 1.3; it shrinks the primary by (1-w)
        and therefore CAN rescue, which is why it is not the default.

        ``max``: the inherited rule, kept so the defect is reproducible.
        """
        stat = float(stat)
        rel = float(rel)
        w = self.temporal_weight
        if self.fusion_mode == "boost":
            return stat + w * rel * (1.0 - stat)
        if self.fusion_mode == "wsum":
            return (1.0 - w) * stat + w * rel
        return max(stat, rel)

    # ==================================================================
    #  (4) five signals + uniformity, combined by reference-free override
    # ==================================================================
    def _compute_per_round_stat_scores(  # type: ignore[override]
        self,
        attacked_replies: List[Message],
        global_arrays: ArrayRecord,
    ) -> Dict[int, float]:
        """The parent's five signals, plus uniformity, plus the override.

        The five signals reproduce ``CustomFedAvg_adaptive_new`` exactly — this
        is a strict extension, and with ``ref_free_override=False`` the returned
        scores are identical to the parent's.  They are reimplemented rather
        than delegated to because the parent returns only the weighted sum,
        while the override needs the clone score on its own and the audit trail
        needs every component.  Keeping the breakdown is also what would have
        made the Fang diagnosis a one-line query instead of an investigation.
        """
        gsd = global_arrays.to_torch_state_dict()
        selected_names = self._get_selected_layer_names(gsd)

        cids: List[int] = []
        raw_norms: List[float] = []
        rep_vecs: Dict[int, torch.Tensor] = {}
        uniformity: Dict[int, float] = {}

        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()
            raw_v = self._get_raw_delta_vector(csd, gsd, selected_names)
            if raw_v is None:
                continue

            raw_norms.append(float(torch.linalg.norm(raw_v).item()))
            cids.append(cid)
            # Uniformity is measured on the RAW delta, before any projection.
            uniformity[cid] = self._uniformity(raw_v)

            if self.svd_ready:
                try:
                    rep_vecs[cid] = self._project_raw_vector_fixed_svd(raw_v)
                except Exception:
                    rep_vecs[cid] = raw_v[: min(512, raw_v.numel())].clone()
            else:
                rep_vecs[cid] = raw_v[: min(512, raw_v.numel())].clone()

        if not cids:
            return {}

        N = len(cids)
        norms_arr = np.array(raw_norms, dtype=np.float64)

        stacked = torch.stack([rep_vecs[cid] for cid in cids], dim=0)   # [N, D]
        ref_vec = stacked.median(dim=0).values
        ref_norm = float(torch.linalg.norm(ref_vec).item())
        ref_sign = torch.sign(ref_vec)
        ref_abs = ref_vec.abs()
        ref_abs_sum = float(ref_abs.sum().item()) + 1e-8

        trim_k = max(2, int(N * 0.80))
        trimmed_median_n = float(np.median(norms_arr[np.argsort(norms_arr)[:trim_k]]))

        dist_raw = np.zeros(N, dtype=np.float64)
        norm_scores: Dict[int, float] = {}
        cosine_scores: Dict[int, float] = {}
        sign_scores: Dict[int, float] = {}

        for idx, cid in enumerate(cids):
            v = stacked[idx]
            v_norm = float(torch.linalg.norm(v).item())

            # 1) norm — log2 ratio against the trimmed benign median
            norm_ratio = max(1.0, norms_arr[idx] / (trimmed_median_n + 1e-8))
            norm_scores[cid] = min(1.0, float(np.log2(norm_ratio)) / 5.0)

            # 2) cosine — angle from the coordinate-wise median
            if v_norm > 1e-8 and ref_norm > 1e-8:
                cos = float(np.clip(
                    float(torch.dot(v, ref_vec)) / (v_norm * ref_norm), -1.0, 1.0
                ))
            else:
                cos = 1.0
            cosine_scores[cid] = (1.0 - cos) / 2.0

            # 3) distance — held for post-loop MAD normalisation
            dist_raw[idx] = float(torch.linalg.norm(v - ref_vec).item())

            # 4) sign — amplitude-weighted disagreement with the median
            disagree = (torch.sign(v) != ref_sign).float()
            sign_scores[cid] = float((ref_abs * disagree).sum().item()) / ref_abs_sum

        median_dist = float(np.median(dist_raw))
        mad_dist = max(float(np.median(np.abs(dist_raw - median_dist))), 1e-6)
        dist_scores = {
            cid: float(np.tanh(
                (_NIST_MAD_SCALE * max(0.0, dist_raw[idx] - median_dist) / mad_dist) / 2.0
            ))
            for idx, cid in enumerate(cids)
        }

        # 5) clone — max pairwise cosine, hinged at tau_c. REFERENCE-FREE.
        clone_scores: Dict[int, float] = {cid: 0.0 for cid in cids}
        if N > 1:
            unit = stacked / torch.linalg.norm(stacked, dim=1).clamp(min=1e-8).unsqueeze(1)
            cos_mat = torch.mm(unit, unit.t()).clamp(-1.0, 1.0)
            cos_mat.fill_diagonal_(-2.0)
            max_sim = cos_mat.max(dim=1).values
            thr = self.clone_cos_threshold
            denom = max(1.0 - thr, 1e-9)
            for idx, cid in enumerate(cids):
                clone_scores[cid] = float(
                    np.clip((float(max_sim[idx].item()) - thr) / denom, 0.0, 1.0)
                )

        # 6) uniformity — equal-coordinate-magnitude structure. REFERENCE-FREE.
        unif_scores = (
            self._uniformity_scores(uniformity)
            if self.use_uniformity
            else {cid: 0.0 for cid in cids}
        )

        # ── combination ──────────────────────────────────────────────────────
        stat_scores: Dict[int, float] = {}
        self.last_signal_breakdown = {}
        for cid in cids:
            ns, cs = norm_scores.get(cid, 0.0), cosine_scores.get(cid, 0.0)
            ds, ss = dist_scores.get(cid, 0.0), sign_scores.get(cid, 0.0)
            cl, us = clone_scores.get(cid, 0.0), unif_scores.get(cid, 0.0)

            weighted = float(np.clip(
                0.30 * ns + 0.20 * cs + 0.25 * ds + 0.15 * ss + 0.10 * cl, 0.0, 1.0
            ))

            # The override may only RAISE a score, so it cannot weaken detection
            # of anything the weighted sum already catches. It exists because the
            # four median-referenced signals above are blinded exactly when a
            # colluding minority captures the median.
            #
            # UNIFORMITY ONLY by default. Clone is equally reference-free, but it
            # is not SPECIFIC: benign max-pairwise cosine climbs past tau_c = 0.95
            # as training converges (measured 0.979 by round 30), because honest
            # clients all descend toward the same optimum. Promoting it made that
            # decisive and produced 39-72% benign false positives. Uniformity has
            # no such drift -- benign 0.002-0.012 against malicious 0.80-1.00 --
            # and it alone carries every attack evaluated, Fang included.
            evidence = us
            if self.override_on_clone:
                evidence = max(evidence, cl)
            ref_free = self.ref_free_gain * evidence if self.ref_free_override else 0.0
            stat_scores[cid] = float(np.clip(max(weighted, ref_free), 0.0, 1.0))

            self.last_signal_breakdown[cid] = {
                "norm": ns, "cosine": cs, "dist": ds, "sign": ss,
                "clone": cl, "uniformity_raw": uniformity.get(cid, 0.0),
                "uniformity": us, "weighted": weighted,
                "ref_free": ref_free, "final": stat_scores[cid],
            }

        return stat_scores

    # ==================================================================
    #  (2) D1: unsupervised quiet-round test (the gap-abstention criterion)
    # ==================================================================
    def _round_is_quiet(self, scores: np.ndarray) -> bool:
        """True when the round's score distribution shows no separation.

        This is the abstention condition of the gap filter, computed locally so
        the class does not depend on which filter runs downstream: sort the
        scores, look for the largest gap in the UPPER HALF (M < N/2 means no
        correct cut removes more than half the population), and declare the
        round quiet when that gap does not exceed ``quiet_gap_mult`` robust
        scales of the benign cluster.

        A quiet round is one in which the detector may safely learn: there is no
        evidence of an adversarial cluster to learn FROM.  Crucially it consults
        no server-side data -- unlike the accuracy-based gate it replaces.

        CALIBRATION of ``quiet_gap_mult``.  It shipped at 1.0, which was far too
        tight: the largest gap in the upper half of N clients is an EXTREME-VALUE
        statistic, and against a lower-half MAD of a score distribution that is
        compressed near zero it clears 1x almost always.  On the 2026-08-26
        federated runs the gate opened on only 21 of 127 rounds, and -- the
        decisive evidence -- an ATTACK-FREE federation was judged non-quiet in 6
        of 7 rounds.  The detector consequently trained in one run of the sweep
        and ran at its random initialisation in every other.

        Re-measured on the corrected scores (uniformity-only override, since the
        clone override inflated the benign side and distorted the statistic):

            attack-free rounds (n=37):  median  1.54   p90  2.93   max  6.36
            attacked rounds   (n=127):  median 29.75   p10  5.35   min  0.90

        The two populations are separated by an empty band, [2.93, 5.35].  The
        default sits at its midpoint, which is the choice furthest from both
        failure modes -- too tight and the detector never trains, too loose and
        it trains on attacked rounds, reopening the D2 confirmation loop through
        a captured median.  Measured behaviour of the shipped value:

            quiet_gap_mult   attack-free -> quiet   attacked -> quiet
                       1.0                 16.2%                0.8%
                       3.0                 91.9%                5.5%
                   *** 4.0                 91.9%                5.5%
                       5.0                 97.3%                7.1%
                       8.0                100.0%               16.5%

        3.0 and 4.0 measure identically, so the default sits on a plateau: a
        calibration error in either direction changes nothing, which is the
        property to want here.
        """
        s = np.sort(np.asarray(scores, dtype=np.float64))
        n = s.size
        if n < 4:
            # Too few clients for the upper-half gap statistic to mean anything.
            # Treat as quiet: with no evidence of an attack, refusing to train
            # would starve the detector on small federations.
            return True

        _, mad = self._lower_half_scale(s)
        start = int(np.ceil(n / 2))
        if start >= n or start < 1:
            return True

        gaps = s[start:] - s[start - 1:-1]
        if gaps.size == 0:
            return True
        return bool(gaps.max() < self.quiet_gap_mult * mad)

    def _round_is_good(self, acc_val: float, loss_val: float) -> bool:  # type: ignore[override]
        """D1: the training gate, with the server-test-set dependency removed.

        ``acc_val`` and ``loss_val`` are accepted for signature compatibility
        with the caller in strategy.py and are DELIBERATELY IGNORED -- they come
        from ``evaluate_fn``, i.e. the server's global test set.  The gate is the
        unsupervised quiet-round flag computed during this round's scoring.

        Overriding this one method fixes both call sites at once: strategy.py
        gates ``_collect_clean_windows_for_training`` and
        ``_maybe_update_detector`` on the same return value.

        Pure predicate: the streak is advanced in ``_run_convlstm_analysis``
        where ``_last_round_quiet`` is set, once per round. Keeping the mutation
        out of here means a second caller in the same round cannot double-count
        the streak.
        """
        return self._quiet_streak >= self.quiet_streak_required

    # ==================================================================
    #  scoring: parent steps 1-3, then calibrated boost fusion
    # ==================================================================
    def _run_convlstm_analysis(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        global_arrays: ArrayRecord,
        client_val_deltas: Dict[int, float],
    ) -> None:
        """Two detectors in parallel, fused by the calibrated boost rule.

        Steps 1-3 are the parent's and are reproduced rather than delegated to,
        because the parent's step 4 writes the `max`-fused value into
        ``last_anomaly_scores`` and into stat_anomaly.csv.  Delegating and then
        overwriting would leave the CSV recording the defect rather than the fix,
        and that CSV is what analyze_fusion_defect.py audits.
        """
        # ── Step 1: per-round statistical scores, EWMA-smoothed ──────────────
        stat_scores_raw = self._compute_per_round_stat_scores(
            attacked_replies, global_arrays
        )
        lam = float(self.stat_decay)
        for cid, raw_s in stat_scores_raw.items():
            prev = self.ewma_stat.get(cid, None)
            self.ewma_stat[cid] = (
                raw_s if prev is None else lam * prev + (1.0 - lam) * raw_s
            )

        # ── Step 2: grandparent ConvLSTM pass (populates self.ewma_score) ────
        # Skips CustomFedAvg_adaptive_new's own body, which is exactly the
        # max-fusion step this class replaces.
        super(CustomFedAvg_adaptive_new, self)._run_convlstm_analysis(
            server_round,
            attacked_replies,
            clients_state,
            global_arrays,
            client_val_deltas,
        )

        # ── Step 3: two-sided relative ConvLSTM score ────────────────────────
        all_cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]
        convlstm_rel = self._compute_convlstm_relative_scores(all_cids)

        # ── Step 4: per-round recalibration of the temporal arm ──────────────
        rel_used = (
            self._calibrate_relative(convlstm_rel)
            if self.calibrate_temporal
            else convlstm_rel
        )

        # ── Step 5: boost-form fusion ────────────────────────────────────────
        fused_scores: Dict[int, float] = {}
        for cid in all_cids:
            stat_s = self.ewma_stat.get(cid, 0.0)
            rel_s = rel_used.get(cid, 0.0)
            fused = self._fuse(stat_s, rel_s)

            self.last_anomaly_scores[cid] = fused
            fused_scores[cid] = fused

            is_mal = bool(clients_state.get(cid, False))
            log(
                INFO,
                "[Round %s] client=%s | stat_ewma=%.4f  rel_raw=%.4f  rel_cal=%.4f"
                "  → fused=%.4f  malicious=%s",
                server_round, cid, stat_s,
                convlstm_rel.get(cid, 0.0), rel_s, fused, is_mal,
            )

            if hasattr(self, "save_path"):
                # Same schema as the parent, so analyze_fusion_defect.py reads
                # this class's runs unchanged. convlstm_rel is logged RAW so the
                # calibration can be re-derived offline; fused_score is the value
                # the filter actually thresholds.
                self._append_stat_csv(
                    server_round=server_round,
                    cid=cid,
                    stat_raw=stat_scores_raw.get(cid, 0.0),
                    stat_ewma=stat_s,
                    convlstm_rel=convlstm_rel.get(cid, 0.0),
                    fused=fused,
                    is_malicious=is_mal,
                )

            # Per-signal audit trail. Without this the Fang failure was only
            # diagnosable by inference from the aggregate score; with it, "which
            # signals fired" is a direct query.
            if hasattr(self, "save_path"):
                self._append_signal_csv(server_round, cid, is_mal)

        # ── Step 6: the unsupervised training gate for THIS round (D1) ───────
        if fused_scores:
            self._last_round_quiet = self._round_is_quiet(
                np.fromiter(fused_scores.values(), dtype=np.float64)
            )
            # Advance the streak here, once per round, so _round_is_good stays a
            # pure predicate the caller may evaluate more than once.
            self._quiet_streak = (
                self._quiet_streak + 1 if self._last_round_quiet else 0
            )
            log(
                INFO,
                "[Round %s] quiet-round gate: quiet=%s (streak=%d, required=%d)",
                server_round, self._last_round_quiet,
                self._quiet_streak, self.quiet_streak_required,
            )
        else:
            self._last_round_quiet = False
            self._quiet_streak = 0

    # ==================================================================
    #  D2: consensus-referenced training data
    # ==================================================================
    def _collect_clean_windows_for_training(  # type: ignore[override]
        self,
        replies: List[Message],
        *,
        max_windows_per_client: int = 3,
    ) -> None:
        """Collect training windows from clients CLOSEST TO THE ROBUST CONSENSUS.

        The inherited version ranked candidates by ``last_anomaly_scores`` -- the
        detector's own output -- so the detector was trained on whoever it had
        just decided was benign, and any attacker it accepted became training
        data reinforcing that acceptance (D2).

        Here the reference is the coordinate-wise MEDIAN of the round's latest
        latent frames, which a minority cannot displace far, and clients are
        ranked by L2 distance to it.  The detector's own opinion is not
        consulted, so the confirmation loop is broken by construction.
        """
        candidates: List[int] = []
        frames: List[torch.Tensor] = []
        for msg in replies:
            cid = int(msg.metadata.src_node_id)
            hist = self.client_feature_history.get(cid, [])
            if len(hist) < self.window_len:
                continue
            candidates.append(cid)
            frames.append(hist[-1].reshape(-1).to(torch.float32))

        if not candidates:
            return

        if len(candidates) == 1:
            keep = candidates
        else:
            stacked = torch.stack(frames, dim=0)               # [n, C*H*W]
            consensus = stacked.median(dim=0).values           # [C*H*W]
            dist = torch.linalg.norm(stacked - consensus, dim=1).cpu().numpy()
            # Keep the closest fraction; at least one client, never all of them
            # unless the fraction says so.
            n_keep = max(1, int(round(self.consensus_keep_frac * len(candidates))))
            order = np.argsort(dist)[:n_keep]
            keep = [candidates[i] for i in order]

        n_added = 0
        for cid in keep:
            windows = self._extract_sliding_windows(self.client_feature_history[cid])
            if not windows:
                continue
            # MOST RECENT windows only. The selection above ranks clients by the
            # distance of their LATEST frame to the consensus, but
            # _extract_sliding_windows returns every window in the history
            # (history_max=10, window_len=5 gives up to six), most of which do
            # not contain that frame. Sampling them uniformly would let a client
            # admitted for sitting at today's consensus contribute windows from
            # rounds where it was diverging -- which is precisely the profile of
            # the slow-drift adversary of IMPROVEMENT.md section 4.1, and would
            # reopen at the window level the confirmation loop D2 closes at the
            # client level. windows[-1] is the window ending at the frame the
            # client was actually selected on.
            for seq in windows[-max_windows_per_client:]:
                self.clean_window_buffer.append(seq.detach().cpu())
                n_added += 1

        if len(self.clean_window_buffer) > self.max_buffer:
            self.clean_window_buffer = self.clean_window_buffer[-self.max_buffer:]

        log(
            INFO,
            "Consensus window collection: %d/%d clients kept, %d windows added, "
            "buffer=%d",
            len(keep), len(candidates), n_added, len(self.clean_window_buffer),
        )

    # ==================================================================
    #  D3 + D4: training cadence and budget
    # ==================================================================
    def _maybe_update_detector(  # type: ignore[override]
        self,
        current_round: int,
        *,
        round_is_good: bool,
        skip_update: bool = False,
    ) -> None:
        """Train the ConvLSTM on quiet rounds, with a one-off pre-training burst.

        Differences from the inherited version:

        * ``skip_update`` is IGNORED (D3).  The caller sets it from
          ``drop_rate > 0.4``, which disables training exactly when an attack is
          in progress -- the regime in which the benign population is most
          worth modelling.  With the D1 gate it is redundant anyway: a round
          loud enough to trip it is not quiet.
        * The first eligible episode runs ``pretrain_steps`` rather than
          ``max_steps_per_update`` (D4), so the detector starts from a converged
          state instead of accumulating a few hundred steps over the campaign.
        * Every episode is recorded in ``detector_train_log`` and appended to
          ``detector_training.csv``, so the training actually performed is
          auditable rather than assumed.

        ``round_is_good`` arrives from the overridden ``_round_is_good``, i.e.
        it is the unsupervised quiet-round flag, not a test-set signal.
        """
        # Every early return is logged. A campaign in which the detector never
        # trains is a real possibility under the D1 gate (the adversary is active
        # from round 6, so almost no round is quiet), and it would otherwise be
        # invisible: the run completes normally with an untrained ConvLSTM.
        n_buf = len(self.clean_window_buffer)
        if not self.semi_online:
            return
        if not round_is_good:
            log(INFO, "[Round %s] detector: no training — round not quiet (buffer=%d)",
                current_round, n_buf)
            return
        if current_round - self.last_detector_update_round < self.update_every:
            return
        if n_buf < self.buffer_min_windows:
            log(WARNING,
                "[Round %s] detector: quiet round but buffer too small "
                "(%d < %d) — no training", current_round, n_buf, self.buffer_min_windows)
            return

        if self.detector_optimizer is None:
            self.detector_optimizer = torch.optim.Adam(
                self.detector.parameters(), lr=self.detector_lr
            )

        n_steps = self.max_steps_per_update
        phase = "steady"
        if not self._pretrain_done:
            n_steps = max(self.pretrain_steps, n_steps)
            phase = "pretrain"

        device = next(self.detector.parameters()).device
        self.detector.train()

        losses: List[float] = []
        buffer_size = len(self.clean_window_buffer)
        for _ in range(n_steps):
            if buffer_size < self.batch_size:
                break
            batch_ids = np.random.randint(0, buffer_size, size=(self.batch_size,))
            batch = torch.stack(
                [self.clean_window_buffer[i] for i in batch_ids], dim=0
            ).to(device)

            loss = self.detector.training_loss(batch)
            self.detector_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.detector_optimizer.step()
            losses.append(float(loss.item()))

        self.detector.eval()
        self.last_detector_update_round = current_round

        if not losses:
            log(
                WARNING,
                "[Round %s] detector %s episode ran 0 steps (buffer=%d < batch=%d)",
                current_round, phase, buffer_size, self.batch_size,
            )
            return

        self._pretrain_done = True
        mean_loss = float(np.mean(losses))
        self.detector_train_log.append(
            (current_round, mean_loss, len(losses), buffer_size)
        )
        log(
            INFO,
            "[Round %s] detector %s: %d steps, loss %.6f → %.6f (mean %.6f), buffer=%d",
            current_round, phase, len(losses), losses[0], losses[-1],
            mean_loss, buffer_size,
        )
        self._append_train_csv(current_round, phase, losses, buffer_size)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("detector/train_loss", mean_loss, current_round)
            self.tb_writer.add_scalar(
                "detector/buffer_size", float(buffer_size), current_round
            )

    # ==================================================================
    #  audit trail for the training that was actually performed (D4)
    # ==================================================================
    _SIGNAL_COLS = ("norm", "cosine", "dist", "sign", "clone",
                    "uniformity_raw", "uniformity", "weighted", "ref_free", "final")

    def _append_signal_csv(self, server_round: int, cid: int, is_malicious: bool) -> None:
        """Append the per-signal breakdown for one client to signal_breakdown.csv."""
        row = self.last_signal_breakdown.get(cid)
        if row is None:
            return
        csv_path = f"{self.save_path}/signal_breakdown.csv"
        mode = "a" if self._signal_csv_initialized else "w"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._signal_csv_initialized:
                writer.writerow(["round", "client_id", *self._SIGNAL_COLS, "is_malicious"])
                self._signal_csv_initialized = True
            writer.writerow([
                server_round, cid,
                *(f"{row.get(k, 0.0):.6f}" for k in self._SIGNAL_COLS),
                int(is_malicious),
            ])

    def _append_train_csv(
        self,
        server_round: int,
        phase: str,
        losses: List[float],
        buffer_size: int,
    ) -> None:
        """Append one row per training episode to detector_training.csv."""
        if not hasattr(self, "save_path"):
            return
        csv_path = f"{self.save_path}/detector_training.csv"
        mode = "a" if self._train_csv_initialized else "w"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._train_csv_initialized:
                writer.writerow([
                    "round", "phase", "n_steps",
                    "loss_first", "loss_last", "loss_mean", "buffer_size",
                ])
                self._train_csv_initialized = True
            writer.writerow([
                server_round, phase, len(losses),
                f"{losses[0]:.6f}", f"{losses[-1]:.6f}",
                f"{float(np.mean(losses)):.6f}", buffer_size,
            ])
