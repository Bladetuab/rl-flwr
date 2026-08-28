"""
baseline_defenses.py — SignGuard and DnC baselines for head-to-head comparison.

Both classes subclass ``CustomFedAvg_adaptive_no_valdelta`` so they inherit the
*entire* experimental harness unchanged: malicious-client selection, attack
application, per-round CSV logging (``client_states_rounds.csv``), the
TP/FP/FN/TN confusion matrix, and the aggregation step. Only the
*client-selection rule* differs. That is deliberate — it is what makes the
accuracy / ASR / FDR numbers directly comparable to USTAD-FL rather than
merely adjacent.

What each subclass replaces
---------------------------
``_run_convlstm_analysis``        -> no-op (no SVD basis, no ConvLSTM, no EWMA)
``_get_adaptive_filter_quantile`` -> constant (neither defense uses a quantile dial)
``_filter_attacked_replies_by_anomaly`` -> the defense's own rule, then the
                                    inherited CSV + confusion-matrix logging

IMPORTANT — these are OUR implementations, written from the published
descriptions cited below, not the authors' released code. Parameter choices are
recorded in each class docstring so the comparison is auditable.

References
----------
SignGuard : J. Xu, S.-L. Huang, L. Song, T. Lan, "SignGuard: Byzantine-robust
            Federated Learning through Collaborative Malicious Gradient
            Filtering," IEEE ICDCS 2022.
DnC       : V. Shejwalkar, A. Houmansadr, "Manipulating the Byzantine:
            Optimizing Model Poisoning Attacks and Defenses for Federated
            Learning," NDSS 2021. (Divide-and-Conquer defense, Sec. VI.)
"""

from __future__ import annotations

from logging import INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.app import Message
from flwr.common.logger import log

from rl_flwr.CustomFedAvg_adaptive_no_valdelta import CustomFedAvg_adaptive_no_valdelta


# =============================================================================
#  Shared plumbing
# =============================================================================
class _BaselineDefense(CustomFedAvg_adaptive_no_valdelta):
    """Common scaffolding: disable the USTAD-FL detector, keep the harness."""

    #: human-readable name used in logs and the sweep summary
    DEFENSE_NAME = "baseline"

    # -- disable the USTAD-FL detector path -------------------------------
    def _run_convlstm_analysis(self, *args, **kwargs) -> None:  # type: ignore[override]
        """No-op. These baselines do not use the SVD projection, the ConvLSTM,
        or the EWMA state. ``last_anomaly_scores`` is cleared so that any
        incidental reader sees an empty dict rather than stale values."""
        self.last_anomaly_scores = {}

    def _get_adaptive_filter_quantile(self, *args, **kwargs) -> float:  # type: ignore[override]
        """Neither baseline uses an adaptive quantile; return a fixed value so
        the parent's plumbing has something well-defined to log."""
        return float(self.current_filter_quantile)

    # -- update extraction -------------------------------------------------
    def _updates_matrix(
        self, attacked_replies: List[Message]
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Return (cids, U) where U[i] is client i's flattened update vector.

        Uses the same layer selection and (client - global) convention as
        USTAD-FL, so all three defenses see byte-identical inputs.
        """
        gsd = self.round_global_arrays.to_torch_state_dict()
        selected = self._get_selected_layer_names(gsd)

        cids: List[int] = []
        rows: List[np.ndarray] = []
        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()
            v = self._get_raw_delta_vector(csd, gsd, selected)
            if v is None:
                continue
            cids.append(cid)
            rows.append(v.numpy().astype(np.float32))

        if not rows:
            return [int(m.metadata.src_node_id) for m in attacked_replies], None
        return cids, np.vstack(rows)

    # -- shared finish: apply the keep-set, log identically to USTAD-FL ----
    def _finalize(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        all_cids: List[int],
        keep_cids: List[int],
        min_clients: int,
    ):
        keep = set(keep_cids)
        if len(keep) < min_clients:
            # Never leave the aggregator with fewer than min_clients.
            for c in all_cids:
                if len(keep) >= min_clients:
                    break
                keep.add(c)
            log(INFO, "[Round %s] %s: min_clients guard raised keep-set to %d",
                server_round, self.DEFENSE_NAME, len(keep))

        filtered = [m for m in attacked_replies if int(m.metadata.src_node_id) in keep]
        detected_state = {cid: (cid not in keep) for cid in all_cids}

        log(INFO, "[Round %s] %s: kept=%d dropped=%d",
            server_round, self.DEFENSE_NAME, len(keep), len(all_cids) - len(keep))

        if hasattr(self, "save_path"):
            self._append_round_states_csv(server_round, all_cids, clients_state, detected_state)
        self._update_confusion_matrix(all_cids, clients_state, detected_state)
        return filtered, detected_state


# =============================================================================
#  SignGuard  (Xu et al., ICDCS 2022)
# =============================================================================
class SignGuardDefense(_BaselineDefense):
    """SignGuard: norm filtering AND sign-statistics clustering, intersected.

    Algorithm (Xu et al., Sec. IV):
      1. *Norm filter.* Let M be the median of the client update norms. Keep
         client i iff  lo <= ||g_i|| / M <= hi.  Paper uses lo=0.1, hi=3.0.
      2. *Sign filter.* Draw a random coordinate subset of size ``n_sub``.
         For each client compute the sign statistics
         (frac positive, frac zero, frac negative) on that subset, giving a
         3-D feature per client. Cluster the features (k-means, k=2) and keep
         the LARGER cluster — the honest majority.
      3. *Intersect.* Benign = norm-kept AND sign-kept.

    Assumption profile: needs no attacker count and no root dataset — the same
    profile USTAD-FL claims. This is what makes it the fair comparator.
    """

    DEFENSE_NAME = "SignGuard"

    def __init__(self, *args,
                 sg_norm_lo: float = 0.1,
                 sg_norm_hi: float = 3.0,
                 sg_n_sub: int = 1000,
                 sg_seed: int = 42,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sg_norm_lo = float(sg_norm_lo)
        self.sg_norm_hi = float(sg_norm_hi)
        self.sg_n_sub = int(sg_n_sub)
        self._sg_rng = np.random.default_rng(sg_seed)

    @staticmethod
    def _kmeans2(X: np.ndarray, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
        """Minimal 2-means. Returns a label array in {0,1}."""
        n = X.shape[0]
        if n < 2:
            return np.zeros(n, dtype=int)
        # k-means++ style seeding: farthest point from a random first centre
        c0 = X[rng.integers(n)]
        d0 = np.linalg.norm(X - c0, axis=1)
        c1 = X[int(np.argmax(d0))]
        centres = np.vstack([c0, c1])
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            d = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
            new = np.argmin(d, axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for k in (0, 1):
                if np.any(labels == k):
                    centres[k] = X[labels == k].mean(axis=0)
        return labels

    def _filter_attacked_replies_by_anomaly(  # type: ignore[override]
        self, server_round, attacked_replies, clients_state, *,
        quantile: float = 0.8, min_clients: int = 2, abs_threshold=None,
    ):
        all_cids, U = self._updates_matrix(attacked_replies)
        if U is None:
            return self._finalize(server_round, attacked_replies, clients_state,
                                  all_cids, all_cids, min_clients)

        # ---- 1. norm filter ------------------------------------------------
        norms = np.linalg.norm(U, axis=1)
        med = float(np.median(norms))
        if med <= 0:
            norm_keep = set(all_cids)
        else:
            ratio = norms / med
            norm_keep = {
                all_cids[i] for i in range(len(all_cids))
                if self.sg_norm_lo <= ratio[i] <= self.sg_norm_hi
            }

        # ---- 2. sign-statistics clustering ---------------------------------
        d = U.shape[1]
        n_sub = min(self.sg_n_sub, d)
        idx = self._sg_rng.choice(d, size=n_sub, replace=False)
        sub = U[:, idx]
        feats = np.stack([
            (sub > 0).mean(axis=1),
            (sub == 0).mean(axis=1),
            (sub < 0).mean(axis=1),
        ], axis=1).astype(np.float64)

        labels = self._kmeans2(feats, self._sg_rng)
        counts = np.bincount(labels, minlength=2)
        majority = int(np.argmax(counts))
        sign_keep = {all_cids[i] for i in range(len(all_cids)) if labels[i] == majority}

        keep = sorted(norm_keep & sign_keep)

        log(INFO,
            "[Round %s] SignGuard: median_norm=%.4f norm_kept=%d sign_kept=%d "
            "intersect=%d", server_round, med, len(norm_keep), len(sign_keep), len(keep))

        return self._finalize(server_round, attacked_replies, clients_state,
                              all_cids, keep, min_clients)


# =============================================================================
#  DnC  (Shejwalkar & Houmansadr, NDSS 2021)
# =============================================================================
class DnCDefense(_BaselineDefense):
    """Divide-and-Conquer: spectral outlier removal on randomly sparsified updates.

    Algorithm (Shejwalkar & Houmansadr, Sec. VI):
      For each of ``niters`` random dimension subsamples of size ``d_sub``:
        1. mu   = mean of the subsampled updates
        2. C    = U_sub - mu                          (centred)
        3. v    = top right singular vector of C
        4. s_i  = ((U_sub[i] - mu) . v)^2             (outlier score)
        5. drop the  ceil(c * m)  highest-scoring clients
      Keep the INTERSECTION of the per-iteration keep-sets, then aggregate.

    NOTE — privileged threat model. DnC requires an estimate ``m`` of the
    number of malicious clients, exactly as Multi-Krum does. We grant it the
    true count (``num_malicious``), which is the most generous setting, and
    record it as ``assumed_malicious`` in the sweep summary so the privilege is
    auditable rather than buried.
    """

    DEFENSE_NAME = "DnC"

    def __init__(self, *args,
                 dnc_niters: int = 5,
                 dnc_d_sub: int = 2000,
                 dnc_c: float = 1.0,
                 dnc_seed: int = 42,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dnc_niters = int(dnc_niters)
        self.dnc_d_sub = int(dnc_d_sub)
        self.dnc_c = float(dnc_c)
        self._dnc_rng = np.random.default_rng(dnc_seed)
        # the privileged input, recorded for the summary row
        self.assumed_malicious = int(getattr(self, "num_malicious", 0))

    def _filter_attacked_replies_by_anomaly(  # type: ignore[override]
        self, server_round, attacked_replies, clients_state, *,
        quantile: float = 0.8, min_clients: int = 2, abs_threshold=None,
    ):
        all_cids, U = self._updates_matrix(attacked_replies)
        if U is None:
            return self._finalize(server_round, attacked_replies, clients_state,
                                  all_cids, all_cids, min_clients)

        n, d = U.shape
        n_drop = int(np.ceil(self.dnc_c * self.assumed_malicious))
        n_drop = max(0, min(n_drop, max(0, n - min_clients)))

        keep = set(all_cids)
        score_sum = np.zeros(n, dtype=np.float64)
        n_iter_ok = 0
        for _ in range(self.dnc_niters):
            d_sub = min(self.dnc_d_sub, d)
            idx = self._dnc_rng.choice(d, size=d_sub, replace=False)
            sub = U[:, idx].astype(np.float64)

            mu = sub.mean(axis=0)
            C = sub - mu
            # top right singular vector of the centred matrix
            try:
                _, _, Vt = np.linalg.svd(C, full_matrices=False)
                v = Vt[0]
            except np.linalg.LinAlgError:
                continue

            scores = (C @ v) ** 2
            score_sum += scores
            n_iter_ok += 1
            if n_drop > 0:
                drop_idx = np.argsort(scores)[-n_drop:]
                iter_keep = {all_cids[i] for i in range(n) if i not in set(drop_idx.tolist())}
            else:
                iter_keep = set(all_cids)
            keep &= iter_keep

        # Faithful DnC intersects the per-iteration good sets. When the top
        # singular direction is unstable across subsamples -- which is exactly
        # what an attack designed against spectral filtering produces -- that
        # intersection can collapse far below the n - c*m clients the rule
        # intends to keep, leaving a degenerate aggregate. In that case we fall
        # back to ranking by the mean score across iterations and keeping
        # n - c*m clients. This is a documented deviation from the published
        # rule, applied only when it degenerates, and it is logged when it fires
        # so the comparison stays auditable.
        target_keep = max(min_clients, n - n_drop)
        if len(keep) < target_keep and n_iter_ok > 0:
            order = np.argsort(score_sum / n_iter_ok)
            keep = {all_cids[i] for i in order[:target_keep].tolist()}
            log(INFO,
                "[Round %s] DnC: intersection collapsed to %d < %d; fell back to "
                "mean-score ranking", server_round, len(keep), target_keep)

        keep_sorted = sorted(keep)
        log(INFO, "[Round %s] DnC: n=%d d=%d n_drop=%d (assumed_malicious=%d) kept=%d",
            server_round, n, d, n_drop, self.assumed_malicious, len(keep_sorted))

        return self._finalize(server_round, attacked_replies, clients_state,
                              all_cids, keep_sorted, min_clients)


DEFENSE_REGISTRY = {
    "ustad": CustomFedAvg_adaptive_no_valdelta,
    "signguard": SignGuardDefense,
    "dnc": DnCDefense,
}
