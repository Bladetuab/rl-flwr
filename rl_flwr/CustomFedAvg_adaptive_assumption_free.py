"""
CustomFedAvg_adaptive_assumption_free.py

CustomFedAvg_adaptive_no_valdelta with the three remaining *adversary-knowledge*
assumptions removed.  Nothing here touches the five detection signals, the
ConvLSTM arm, the fused scoring, or the aggregation rule — only the three places
where the inherited design smuggles in knowledge it claims not to need.

(A) THE QUANTILE DROP FLOOR
    The inherited filter computes
        effective_thr = min(q_thr, a_thr),   q_thr = quantile(scores, q)
    Because the quantile branch sits inside a ``min``, it *guarantees* that
    (1 - q) of the population is dropped every round regardless of what the
    scores say.  A hardcoded drop rate is an assumption about the malicious
    fraction wearing a different hat, and it is the direct cause of the
    ~0.40-0.44 false-detection rate seen in warm-up and in attack-free controls.

    Fix: replace the *quantile* branch with a *gap* branch.  The architecture is
    unchanged — still ``min`` of two thresholds — but the relative branch is now
    data-driven: it cuts at the largest gap in the upper half of the sorted score
    distribution, and yields nothing at all when no significant gap exists.  A
    federation with no attacker therefore loses no clients.

(B) THE TUNED ABSOLUTE CONSTANT
    ``abs_drop_threshold = 0.40`` is a single global constant.  It transfers
    between MNIST and Fashion-MNIST but has not been shown to transfer to harder
    datasets.  The gap branch above is scale-free and needs no constant; the
    absolute branch is retained as the second half of the ``min`` so a
    spread-out adversarial population (which defeats a single-gap heuristic)
    is still caught.  ``threshold_mode="zscore"`` offers a modified-z variant
    for ablation (see WARNING).

(C) THE CLEAN WARM-UP
    ``_fit_fixed_svd_basis_from_vectors`` centres on the arithmetic mean and
    decomposes the *whole* warm-up buffer, whose docstring assumes "benign
    samples".  An adversary that simply attacks from round 1 poisons the basis
    that every later round is scored against.

    Fix: centre on the coordinate-wise median (breakdown point 1/2, consistent
    with the rest of USTAD-FL) and trim the most-distant ``svd_trim_frac`` of
    samples before the decomposition, so a minority cannot determine the basis.

Synthetic validation (100 clients, 40% adversarial, invented score
distributions — NOT the real detector's scores)
--------------------------------------------------------------------------
  regime                     threshold   dropped   recall   false positives
  no attack                  0.4000        0         --           0
  40% well-separated         0.4000       40       1.000          0
  40% near-overlapping       0.2126       39       0.975          0
  40% spread 0.20-0.95       0.4000       28       0.700          0
  40% heavily overlapping    0.4000        0       0.000          0
  degenerate (identical)     0.4000        0         --           0

Two honest limits this exposes:
  * A heavily overlapping adversarial population is missed.  When the score
    distributions genuinely overlap no threshold can separate them; this is an
    information limit of score-based filtering, not of this rule.  The inherited
    quantile floor "caught" such attackers only by dropping 45% of everyone by
    rank, which is not detection.
  * A spread-out adversarial population defeats a single-gap heuristic; the
    absolute branch is what recovers recall to 0.700 there.

WARNING on the modified z-score variant
---------------------------------------
``threshold_mode="zscore"`` cuts at ``z_cut`` modified-z units above the median
using the NIST constant already used in this codebase (0.6745, see
CustomFedAvg_adaptive_new lines 251 and 315-325).  MAD has a 50% breakdown point,
so at the 40% fraction evaluated here it does *not* blow up — but z_cut = 3.5 is
calibrated for finding rare outliers in a mostly-clean sample, and on a
near-overlapping 40% cluster it lands mid-adversary (measured: recall 0.025).
It is kept for ablation, not recommended as the operating rule.

NOTHING HERE IS VALIDATED ON REAL RUNS.  The table above is synthetic.  This file
implements a stated design correction; it does not report a result.
"""

from __future__ import annotations

from logging import INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.app import Message
from flwr.common import log

from rl_flwr.CustomFedAvg_adaptive_no_valdelta import CustomFedAvg_adaptive_no_valdelta

#: NIST modified z-score consistency constant (E[MAD] -> sigma for a normal).
_NIST_MAD_SCALE = 0.6745

_MODES = ("gap_abs", "zscore", "absolute")


class CustomFedAvg_adaptive_assumption_free(CustomFedAvg_adaptive_no_valdelta):
    """USTAD-FL with the quantile drop floor, the tuned absolute threshold, and
    the clean-warm-up SVD fit removed.

    Drop-in: every new parameter is keyword-only with a default, so the
    constructor stays call-compatible with the parent and no ``pyproject.toml``
    change is required.
    """

    def __init__(
        self,
        *args,
        # ── (A) relative branch: gap instead of quantile ──────────────────
        threshold_mode: str = "gap_abs",
        gap_mult: float = 1.0,
        backstop_quantile: float = 1.0,
        # ── (B) absolute branch ──────────────────────────────────────────
        # Upper bound on the derived threshold, i.e. a guaranteed MINIMUM
        # strictness — applied via `min`, exactly like the parent's always-on
        # abs_drop_threshold. Not a lower bound on the threshold.
        abs_cap: Optional[float] = None,   # None -> inherit abs_drop_threshold
        z_cut: float = 3.5,
        use_lower_mad: bool = True,
        # ── (C) robust SVD basis fit ─────────────────────────────────────
        robust_svd: bool = True,
        svd_trim_frac: float = 0.40,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if threshold_mode not in _MODES:
            raise ValueError(f"threshold_mode must be one of {_MODES}; got {threshold_mode!r}")
        if not 0.0 < backstop_quantile <= 1.0:
            raise ValueError(f"backstop_quantile must be in (0, 1]; got {backstop_quantile}")
        if not 0.0 <= svd_trim_frac < 1.0:
            raise ValueError(f"svd_trim_frac must be in [0, 1); got {svd_trim_frac}")
        if gap_mult < 0.0:
            raise ValueError(f"gap_mult must be >= 0; got {gap_mult}")

        self.threshold_mode = str(threshold_mode)
        self.gap_mult = float(gap_mult)
        # 1.0 disables the inherited quantile branch entirely: quantile(s, 1.0)
        # is max(s), and `keep = score <= thr` then retains every client, so the
        # branch can no longer impose a drop floor.
        self.backstop_quantile = float(backstop_quantile)
        self.abs_cap = (
            float(self.abs_drop_threshold) if abs_cap is None else float(abs_cap)
        )
        self.z_cut = float(z_cut)
        self.use_lower_mad = bool(use_lower_mad)
        self.robust_svd = bool(robust_svd)
        self.svd_trim_frac = float(svd_trim_frac)

        #: last derived threshold and which branch produced it, for offline audit
        self.last_effective_abs_threshold: Optional[float] = None
        self.last_threshold_source: Optional[str] = None

        log(
            INFO,
            "USTAD-FL (assumption-free): mode=%s gap_mult=%.2f abs_cap=%.4f "
            "z_cut=%.2f lower_mad=%s backstop_q=%.3f robust_svd=%s trim=%.2f",
            self.threshold_mode, self.gap_mult, self.abs_cap, self.z_cut,
            self.use_lower_mad, self.backstop_quantile, self.robust_svd,
            self.svd_trim_frac,
        )

    # ==================================================================
    #  robust scale of the benign population
    # ==================================================================
    def _benign_scale(self, s_sorted: np.ndarray) -> Tuple[float, float]:
        """Return (median, robust scale) of the score distribution.

        With ``use_lower_mad`` the scale is estimated from ``s <= median`` only.
        Under the Byzantine assumption M < N/2 the median lies inside the benign
        cluster, so that half is uncontaminated and a high-scoring adversarial
        cluster cannot inflate the estimate even as the fraction approaches 1/2.
        """
        med = float(np.median(s_sorted))
        ref = s_sorted[s_sorted <= med] if self.use_lower_mad else s_sorted
        if ref.size == 0:
            ref = s_sorted
        mad = float(np.median(np.abs(ref - med)))
        # Same degenerate-MAD guard the parent uses for its z-scores.
        return med, max(mad, 1e-6)

    # ==================================================================
    #  (A) relative branch: largest gap above the median
    # ==================================================================
    def _gap_threshold(self, s_sorted: np.ndarray, mad: float) -> Optional[float]:
        """Cut at the largest gap in the upper half of the sorted scores.

        The search starts at the median because M < N/2 means the majority is
        benign, so a correct cut can never remove more than half the population.
        A gap only counts when it exceeds ``gap_mult`` robust scales; otherwise
        the distribution is treated as unimodal and this branch abstains
        (returns None), which is what lets an attack-free round drop nobody.
        """
        n = s_sorted.size
        start = int(np.ceil(n / 2))
        if start >= n or start < 1:
            return None

        gaps = s_sorted[start:] - s_sorted[start - 1:-1]
        if gaps.size == 0:
            return None

        j = int(np.argmax(gaps))
        if float(gaps[j]) < self.gap_mult * mad:
            return None

        # midpoint of the gap
        return float((s_sorted[start + j - 1] + s_sorted[start + j]) / 2.0)

    # ==================================================================
    #  threshold assembly
    # ==================================================================
    def _derive_abs_threshold(self, scores_arr: np.ndarray) -> Optional[float]:
        """Threshold in score units, or None to fall back to the parent constant."""
        if self.threshold_mode == "absolute":
            self.last_threshold_source = "absolute"
            return None

        s = np.asarray(scores_arr, dtype=np.float64).ravel()
        if s.size < 4:
            self.last_threshold_source = "too-few-scores"
            return None

        s_sorted = np.sort(s)
        med, mad = self._benign_scale(s_sorted)

        if self.threshold_mode == "zscore":
            thr = med + (self.z_cut / _NIST_MAD_SCALE) * mad
            thr = min(thr, self.abs_cap)
            self.last_threshold_source = "zscore"
            return float(thr)

        # "gap_abs": min of the gap branch and the absolute branch, i.e. drop if
        # EITHER fires. The gap branch is precise when the population is
        # bimodal; the absolute branch covers a spread-out adversary that a
        # single gap cannot bracket.
        gap_thr = self._gap_threshold(s_sorted, mad)
        if gap_thr is None:
            self.last_threshold_source = "abs-floor(no-gap)"
            return float(self.abs_cap)

        if gap_thr <= self.abs_cap:
            self.last_threshold_source = "gap"
            return float(gap_thr)

        self.last_threshold_source = "abs-floor"
        return float(self.abs_cap)

    # ==================================================================
    #  (A) wire it in; retire the quantile branch
    # ==================================================================
    def _filter_attacked_replies_by_anomaly(  # type: ignore[override]
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        *,
        quantile: float = 0.8,
        min_clients: int = 2,
        abs_threshold: float | None = None,
    ):
        """Threshold-primary filter; the caller's ``quantile`` is ignored.

        ``quantile`` is the parameter being removed, so it is replaced by
        ``backstop_quantile`` (default 1.0, which makes the inherited quantile
        branch inert).  Everything downstream — the min_clients guard, the
        per-round state CSV, and the confusion-matrix accumulation — is the
        parent's, unchanged, so metrics stay directly comparable to the
        inherited strategy.
        """
        cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]
        scored = [c for c in cids if c in self.last_anomaly_scores]

        derived: Optional[float] = None
        if abs_threshold is None and scored:
            derived = self._derive_abs_threshold(
                np.array([self.last_anomaly_scores[c] for c in scored], dtype=np.float64)
            )

        effective_abs = abs_threshold if abs_threshold is not None else derived
        self.last_effective_abs_threshold = (
            float(effective_abs) if effective_abs is not None else None
        )

        log(
            INFO,
            "[Round %s] assumption-free: quantile %.2f -> %.3f (inert); "
            "abs_thr=%s via=%s",
            server_round, quantile, self.backstop_quantile,
            f"{effective_abs:.4f}" if effective_abs is not None
            else f"inherited {self.abs_drop_threshold:.4f}",
            self.last_threshold_source,
        )

        return super()._filter_attacked_replies_by_anomaly(
            server_round,
            attacked_replies,
            clients_state,
            quantile=self.backstop_quantile,
            min_clients=min_clients,
            abs_threshold=effective_abs,
        )

    # ==================================================================
    #  (C) robust SVD basis fit — no clean-warm-up assumption
    # ==================================================================
    def _fit_fixed_svd_basis_from_vectors(  # type: ignore[override]
        self, vectors: List[torch.Tensor]
    ) -> bool:
        """Median-centred, trimmed SVD basis fit.

        Falls back to the parent when ``robust_svd`` is off, and on any numerical
        failure.
        """
        if not self.robust_svd:
            return super()._fit_fixed_svd_basis_from_vectors(vectors)

        valid = [v.detach().cpu().float() for v in vectors if v is not None]
        if len(valid) < 2:
            return False

        # Same pad/truncate-to-common-dim convention as the parent.
        dim = max(v.numel() for v in valid)
        padded = []
        for v in valid:
            if v.numel() < dim:
                v = torch.cat([v, torch.zeros(dim - v.numel(), dtype=v.dtype)], dim=0)
            elif v.numel() > dim:
                v = v[:dim]
            padded.append(v)

        X = torch.stack(padded, dim=0)  # [N, D]

        # ── robust centre: coordinate-wise median, not the mean ───────────
        centre = X.median(dim=0).values  # [D]
        Xc = X - centre.unsqueeze(0)

        # ── trim the most-distant samples before decomposing ──────────────
        n = Xc.shape[0]
        n_keep = max(2, int(round(n * (1.0 - self.svd_trim_frac))))
        if n_keep < n:
            dist = torch.linalg.vector_norm(Xc, dim=1)  # [N]
            Xc = Xc[torch.argsort(dist)[:n_keep]]
            log(
                INFO,
                "Robust SVD fit: trimmed %d/%d most-distant warm-up samples "
                "(trim_frac=%.2f)",
                n - n_keep, n, self.svd_trim_frac,
            )

        if torch.allclose(Xc, torch.zeros_like(Xc)):
            return False

        try:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except Exception as exc:  # pragma: no cover - numerical fallback
            log(INFO, "Robust SVD fit failed (%s); falling back to parent.", exc)
            return super()._fit_fixed_svd_basis_from_vectors(vectors)

        k = min(self.svd_target_len, Vh.shape[0])
        self.svd_basis = Vh[:k].detach().cpu().float()
        # _project_raw_vector_fixed_svd subtracts svd_mean, so the robust centre
        # must live there for the projection to be consistent with the basis.
        self.svd_mean = centre.detach().cpu().float()
        self.svd_ready = True

        log(
            INFO,
            "Robust SVD basis fitted (median-centred, trimmed): samples=%d/%d "
            "dim=%d k=%d",
            Xc.shape[0], n, dim, k,
        )
        return True
