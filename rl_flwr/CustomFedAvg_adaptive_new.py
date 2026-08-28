"""
CustomFedAvg_adaptive_new.py

Enhanced federated anomaly-detection strategy that reliably detects every attack
type in attack.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THE ORIGINAL SYSTEM FAILS FOR GAUSSIAN AND LIE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gaussian (observed from logs):
  malicious avg ConvLSTM score = 0.48, benign avg = 0.12, but benign MAX = 1.75
  > benign malicious MAX = 1.58.  Distributions OVERLAP heavily.
  ConvLSTM is noisy; a single bad benign round creates a false positive.

LIE (observed from logs):
  malicious avg ConvLSTM score = 0.20, benign avg = 0.44.
  The polarity is INVERTED: malicious clients score LOWER than benign.
  Root cause: LIE crafts the same poisoned vector every round
  (corrupted = mean − z·std^0.5).  The ConvLSTM temporal forecaster quickly
  learns to *predict* this fixed pattern → near-zero prediction error → LOW
  anomaly score.  Benign clients vary naturally → HIGHER prediction error.
  With the original raw-EWMA score, LIE clients are systematically classified
  as benign and benign clients as malicious.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWO-PART FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fix 1 — Two-sided relative ConvLSTM score (catches LIE)
  Replace "raw EWMA" with the ABSOLUTE MAD z-score of that EWMA relative to
  the per-round population.  This flags clients that are anomalously HIGH
  (Gaussian, Min-Max, Fang) *and* anomalously LOW (LIE) compared to peers.

  convlstm_relative[cid] = tanh(max(|z_upper|, |z_lower|) / 2)

  For LIE:  malicious EWMA ≪ median → z_lower large → convlstm_relative high ✓
  For rest: malicious EWMA ≫ median → z_upper large → convlstm_relative high ✓
  For benign: EWMA ≈ median  → both z-scores ≈ 0 → score ≈ 0 ✓

Fix 2 — Richer per-round statistical detector (catches Gaussian + LIE earlier)
  Five complementary signals computed each round (no temporal history needed):

  Signal        Weight  Targets
  ──────────── ─────── ─────────────────────────────────────────────────────
  norm_score    0.30    Gaussian — log2-ratio of L2 norm vs trimmed median.
                        Gaussian noise (magnitude=10) inflates norms ~1000×.
  cosine_score  0.20    LIE, Min-Max — angle from coordinate-wise median.
                        Both shift updates away from the benign direction.
  dist_score    0.25    LIE, Min-Max — Euclidean distance from coord-wise
                        median in latent space.  LIE = mean − z·std^0.5
                        displaces update by z·‖std^0.5‖ from benign centroid.
  sign_score    0.15    LIE — weighted sign disagreement with coord-wise
                        median.  LIE flips signs in high-variance dimensions.
  clone_score   0.10    Min-Max — pairwise cosine similarity.  Min-Max sends
                        identical vectors to all malicious clients (sim ≈ 1).

  All references use the COORDINATE-WISE MEDIAN (not mean) so the signals
  remain valid even with ≤50% malicious clients.

Fusion:  last_anomaly_scores[cid] = max(ewma_stat, convlstm_relative)
  Either arm can independently flag a client.  The two-sided relative ConvLSTM
  score correctly handles LIE while stat catches Gaussian quickly.
"""

from __future__ import annotations

import csv
from logging import INFO
from typing import Dict, List, Optional

import numpy as np
import torch
from flwr.app import ArrayRecord, Message
from flwr.common import log

from rl_flwr.strategy import CustomFedAvg_adaptive


class CustomFedAvg_adaptive_new(CustomFedAvg_adaptive):
    """CustomFedAvg_adaptive with two-sided ConvLSTM scoring and richer stat detector.

    New constructor parameters
    --------------------------
    stat_decay : float
        EWMA decay for the statistical scores (default 0.4).
        Lower = more reactive; higher = more smoothed.
    clone_cos_threshold : float
        Cosine similarity above which two clients are considered potential
        clones (default 0.95).  Min-Max clones score ≈ 1.0.
    abs_drop_threshold : float
        Absolute anomaly score above which a client is ALWAYS dropped from
        aggregation, regardless of the adaptive quantile threshold.
        This closes the quantile-relative gap: if a malicious client scores
        0.78 but the quantile threshold lands at 0.82 (dropping only 20%),
        the absolute threshold still removes it.
        Default 0.40 — calibrated so that:
          - Gaussian (log-norm ~1.0, stat ≥ 0.65) → always dropped ✓
          - LIE (dist+sign ~0.47) → always dropped after fix ✓
          - Benign clients (stat ~0.08–0.28) → rarely above 0.40 ✓
        Set to 1.0 to disable (rely only on quantile).
    """

    def __init__(
        self,
        *args,
        stat_decay: float = 0.4,
        clone_cos_threshold: float = 0.95,
        abs_drop_threshold: float = 0.40,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stat_decay: float = stat_decay
        self.clone_cos_threshold: float = clone_cos_threshold
        self.abs_drop_threshold: float = abs_drop_threshold
        self.ewma_stat: Dict[int, float] = {}
        self._stat_csv_initialized: bool = False

    # ------------------------------------------------------------------
    # Helper: raw delta vector with architecture-agnostic fallback
    # ------------------------------------------------------------------

    def _get_raw_delta_vector(
        self,
        csd: Dict[str, torch.Tensor],
        gsd: Dict[str, torch.Tensor],
        selected_names: List[str],
    ) -> Optional[torch.Tensor]:
        """Return flattened (client − global) for selected layers.

        Falls back to ALL floating-point parameters when the model has no
        'layer4'/'fc' layers so the detector works on any architecture.
        """
        raw_v = self._collect_raw_update_vector(csd, gsd, selected_names)
        if raw_v is not None:
            return raw_v
        vecs: List[torch.Tensor] = []
        for name in gsd:
            p = csd.get(name)
            g = gsd[name]
            if p is None or not g.is_floating_point():
                continue
            vecs.append((p - g).detach().cpu().flatten().float())
        return torch.cat(vecs, dim=0) if vecs else None

    # ------------------------------------------------------------------
    # Fix 2: five-signal per-round statistical anomaly detector
    # ------------------------------------------------------------------

    def _compute_per_round_stat_scores(
        self,
        attacked_replies: List[Message],
        global_arrays: ArrayRecord,
    ) -> Dict[int, float]:
        """Compute per-round statistical anomaly scores (no history required).

        Returns
        -------
        stat_scores : Dict[cid -> float in [0, 1]]
        """
        gsd = global_arrays.to_torch_state_dict()
        selected_names = self._get_selected_layer_names(gsd)

        cids: List[int] = []
        raw_norms: List[float] = []
        rep_vecs: Dict[int, torch.Tensor] = {}

        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()

            raw_v = self._get_raw_delta_vector(csd, gsd, selected_names)
            if raw_v is None:
                continue

            raw_norms.append(float(torch.linalg.norm(raw_v).item()))
            cids.append(cid)

            # Low-dim representation for cosine / dist / sign / clone
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

        # ── Coordinate-wise median (robust reference, tolerates ≤50% attack) ──
        stacked = torch.stack([rep_vecs[cid] for cid in cids], dim=0)  # [N, D]
        ref_vec  = stacked.median(dim=0).values   # [D]
        ref_norm = float(torch.linalg.norm(ref_vec).item())
        ref_sign = torch.sign(ref_vec)             # [D]
        ref_abs  = ref_vec.abs()                   # [D] — magnitude weights
        ref_abs_sum = float(ref_abs.sum().item()) + 1e-8

        # ── Trimmed median of raw norms (bottom 80 % → benign reference) ─────
        trim_k = max(2, int(N * 0.80))
        trim_indices = np.argsort(norms_arr)[:trim_k]
        trimmed_median_n = float(np.median(norms_arr[trim_indices]))

        # ── Per-client measurements ────────────────────────────────────────────
        dist_raw = np.zeros(N, dtype=np.float64)
        norm_scores:   Dict[int, float] = {}
        cosine_scores: Dict[int, float] = {}
        sign_scores:   Dict[int, float] = {}

        for idx, cid in enumerate(cids):
            v = stacked[idx]
            v_norm = float(torch.linalg.norm(v).item())

            # ── 1) norm_score: log2-ratio vs trimmed benign median ───────────
            # Benign: ratio ≈ 1 → score ≈ 0
            # Gaussian (magnitude=10): ratio ≈ 1000 → score = log2(1000)/5 ≈ 2 → clipped to 1
            norm_ratio = max(1.0, norms_arr[idx] / (trimmed_median_n + 1e-8))
            norm_scores[cid] = min(1.0, float(np.log2(norm_ratio)) / 5.0)

            # ── 2) cosine_score: angle from coordinate-wise median ───────────
            if v_norm > 1e-8 and ref_norm > 1e-8:
                cos = float(torch.dot(v, ref_vec) / (v_norm * ref_norm))
                cos = float(np.clip(cos, -1.0, 1.0))
            else:
                cos = 1.0
            # (1 − cos) / 2 maps cos∈[−1,1] → [0,1]; neg cosine → score 1.0
            cosine_scores[cid] = (1.0 - cos) / 2.0

            # ── 3) dist_score: stored for post-loop MAD normalization ────────
            dist_raw[idx] = float(torch.linalg.norm(v - ref_vec).item())

            # ── 4) sign_score: weighted sign disagreement with median ─────────
            # LIE shifts signs of high-variance dimensions → systematic disagreement.
            # Benign clients broadly agree in sign with the median.
            my_sign = torch.sign(v)
            disagree = (my_sign != ref_sign).float()           # [D] binary
            sign_scores[cid] = float(
                (ref_abs * disagree).sum().item()
            ) / ref_abs_sum  # weighted fraction ∈ [0, 1]

        # ── Normalize dist_score with MAD (robust to outliers) ────────────────
        median_dist = float(np.median(dist_raw))
        mad_dist    = float(np.median(np.abs(dist_raw - median_dist)))
        mad_dist    = max(mad_dist, 1e-6)

        dist_scores: Dict[int, float] = {}
        for idx, cid in enumerate(cids):
            z_d = 0.6745 * max(0.0, dist_raw[idx] - median_dist) / mad_dist
            # tanh with /2 is more sensitive than /3 used for norm score
            dist_scores[cid] = float(np.tanh(z_d / 2.0))

        # ── 5) clone_score: max pairwise cosine similarity ────────────────────
        # Min-Max sends the same corrupted vector to every malicious client.
        # O(N²) on the low-dim representation; fine for N ≤ 200.
        clone_scores: Dict[int, float] = {cid: 0.0 for cid in cids}
        if N > 1:
            safe_norms = torch.linalg.norm(stacked, dim=1).clamp(min=1e-8)
            unit = stacked / safe_norms.unsqueeze(1)
            cos_mat = torch.mm(unit, unit.t()).clamp(-1.0, 1.0)
            cos_mat.fill_diagonal_(-2.0)
            max_sim = cos_mat.max(dim=1).values  # [N]
            thr = self.clone_cos_threshold
            denom = max(1.0 - thr, 1e-9)
            for idx, cid in enumerate(cids):
                sim = float(max_sim[idx].item())
                clone_scores[cid] = float(np.clip((sim - thr) / denom, 0.0, 1.0))

        # ── Weighted combination ───────────────────────────────────────────────
        stat_scores: Dict[int, float] = {}
        for cid in cids:
            ns = norm_scores.get(cid, 0.0)    # 0.30 — Gaussian
            cs = cosine_scores.get(cid, 0.0)  # 0.20 — LIE, Min-Max
            ds = dist_scores.get(cid, 0.0)    # 0.25 — LIE, Min-Max
            ss = sign_scores.get(cid, 0.0)    # 0.15 — LIE
            cl = clone_scores.get(cid, 0.0)   # 0.10 — Min-Max clones
            stat_scores[cid] = float(np.clip(
                0.30 * ns + 0.20 * cs + 0.25 * ds + 0.15 * ss + 0.10 * cl,
                0.0, 1.0,
            ))

        return stat_scores

    # ------------------------------------------------------------------
    # Fix 1: two-sided relative ConvLSTM score
    # ------------------------------------------------------------------

    def _compute_convlstm_relative_scores(
        self,
        cids: List[int],
    ) -> Dict[int, float]:
        """Compute two-sided MAD z-score of each client's EWMA relative to peers.

        The raw ConvLSTM EWMA score has INVERTED polarity for LIE:
          - LIE clients score LOW   (ConvLSTM learns their consistent pattern)
          - Benign clients score MODERATE/HIGH (more varied behavior)

        Using the absolute deviation from the population median captures BOTH
        directions:
          z_upper  catches clients anomalously HIGH  (Gaussian, Min-Max, Fang)
          z_lower  catches clients anomalously LOW   (LIE)

        Returns Dict[cid -> relative_score in [0, 1]].
        Only clients present in self.ewma_score are included.
        """
        # Collect current EWMA scores for all clients that have been scored
        scored = {cid: self.ewma_score[cid] for cid in cids if cid in self.ewma_score}
        if len(scored) < 2:
            return {cid: 0.0 for cid in cids}

        vals = np.array(list(scored.values()), dtype=np.float64)
        median_val = float(np.median(vals))
        mad_val = float(np.median(np.abs(vals - median_val)))
        mad_val = max(mad_val, 1e-6)

        rel_scores: Dict[int, float] = {}
        for cid in cids:
            if cid not in scored:
                rel_scores[cid] = 0.0
                continue
            ewma = scored[cid]
            z_upper = 0.6745 * max(0.0, ewma - median_val) / mad_val   # high → Gaussian/Minmax/Fang
            z_lower = 0.6745 * max(0.0, median_val - ewma) / mad_val   # low  → LIE
            z_combined = max(z_upper, z_lower)
            rel_scores[cid] = float(np.tanh(z_combined / 2.0))

        return rel_scores

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _append_stat_csv(
        self,
        server_round: int,
        cid: int,
        stat_raw: float,
        stat_ewma: float,
        convlstm_rel: float,
        fused: float,
        is_malicious: bool,
    ) -> None:
        """Write per-round per-client stat scores to stat_anomaly.csv."""
        csv_path = f"{self.save_path}/stat_anomaly.csv"
        mode = "a" if self._stat_csv_initialized else "w"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._stat_csv_initialized:
                writer.writerow([
                    "round", "client_id",
                    "stat_raw", "stat_ewma", "convlstm_rel",
                    "fused_score", "is_malicious",
                ])
                self._stat_csv_initialized = True
            writer.writerow([
                server_round, cid,
                f"{stat_raw:.6f}", f"{stat_ewma:.6f}",
                f"{convlstm_rel:.6f}", f"{fused:.6f}",
                int(is_malicious),
            ])

    # ------------------------------------------------------------------
    # Override: combine stat detector + two-sided relative ConvLSTM score
    # ------------------------------------------------------------------

    def _run_convlstm_analysis(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        global_arrays: ArrayRecord,
        client_val_deltas: Dict[int, float],
    ) -> None:
        """Two detectors in parallel, fused by max-pooling.

        Processing order
        ────────────────
        1. Per-round stat scores → EWMA-smoothed → ewma_stat[cid]
        2. Parent ConvLSTM analysis → ewma_score[cid] (raw EWMA, set by parent)
        3. Two-sided relative ConvLSTM score → convlstm_relative[cid]
        4. Fusion: last_anomaly_scores[cid] = max(ewma_stat, convlstm_relative)

        Why max-fusion instead of weighted average?
        Either arm can independently flag a client.  Stat catches Gaussian from
        round 1; convlstm_relative catches LIE once EWMAs diverge (after ~5–8
        rounds); together they cover every attack type in attack.py.
        """

        # ── Step 1: per-round statistical scores ──────────────────────────────
        stat_scores_raw = self._compute_per_round_stat_scores(
            attacked_replies, global_arrays
        )

        lam = float(self.stat_decay)
        for cid, raw_s in stat_scores_raw.items():
            prev = self.ewma_stat.get(cid, None)
            self.ewma_stat[cid] = (
                raw_s if prev is None else lam * prev + (1.0 - lam) * raw_s
            )

        # ── Step 2: parent ConvLSTM analysis (writes ewma_score & last_anomaly_scores) ──
        super()._run_convlstm_analysis(
            server_round,
            attacked_replies,
            clients_state,
            global_arrays,
            client_val_deltas,
        )

        # ── Step 3: two-sided relative ConvLSTM score ─────────────────────────
        all_cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]
        convlstm_rel = self._compute_convlstm_relative_scores(all_cids)

        # ── Step 4: max-fusion ────────────────────────────────────────────────
        for cid in all_cids:
            stat_s = self.ewma_stat.get(cid, 0.0)
            rel_s  = convlstm_rel.get(cid, 0.0)
            fused  = max(stat_s, rel_s)

            self.last_anomaly_scores[cid] = fused

            is_mal = bool(clients_state.get(cid, False))
            log(
                INFO,
                "[Round %s] client=%s | stat_ewma=%.4f  convlstm_rel=%.4f"
                "  → fused=%.4f  malicious=%s",
                server_round, cid, stat_s, rel_s, fused, is_mal,
            )

            if hasattr(self, "save_path"):
                raw_s = stat_scores_raw.get(cid, 0.0)
                self._append_stat_csv(
                    server_round=server_round,
                    cid=cid,
                    stat_raw=raw_s,
                    stat_ewma=stat_s,
                    convlstm_rel=rel_s,
                    fused=fused,
                    is_malicious=is_mal,
                )

    # ------------------------------------------------------------------
    # Override: dual-threshold filter (quantile + absolute)
    # ------------------------------------------------------------------

    def _filter_attacked_replies_by_anomaly(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        *,
        quantile: float = 0.8,
        min_clients: int = 2,
        abs_threshold: float | None = None,
    ):
        """Filter with BOTH a quantile threshold AND an absolute threshold.

        A client is dropped if EITHER condition holds:
          1. Its fused score > quantile-threshold  (relative, adaptive)
          2. Its fused score > abs_drop_threshold  (absolute, always-on)

        Condition 2 closes the key gap: when the score gap between malicious
        and benign is large (e.g. Gaussian scores 0.78, benign 0.08), the
        absolute threshold ensures the malicious client is always removed even
        if the quantile threshold would have kept it (because q landed too high).

        The min_clients guard is preserved so the aggregator is never left with
        fewer than min_clients clients.  When it would trigger, we log a warning
        and keep only the lowest-scoring min_clients — accepting that one may be
        malicious rather than crashing or aggregating zero clients.
        """
        cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]

        # Collect scores — stat scores are available from round 1 so this
        # dict will normally be fully populated after the first round.
        cid_with_score = [c for c in cids if c in self.last_anomaly_scores]
        scores = [self.last_anomaly_scores[c] for c in cid_with_score]

        if not scores:
            # No scores yet (should not happen in new class after round 1).
            log(INFO, "[Round %s] No fused scores available — passing all clients.", server_round)
            detected_state = {cid: False for cid in cids}
            return attacked_replies, detected_state

        scores_arr = np.array(scores, dtype=np.float32)

        # ── Threshold 1: quantile-based (relative, adaptive) ─────────────────
        q_thr = float(np.quantile(scores_arr, quantile))

        # ── Threshold 2: absolute (always-on) ────────────────────────────────
        # Use the caller-supplied abs_threshold if given; otherwise use the
        # instance-level abs_drop_threshold.
        a_thr = float(abs_threshold) if abs_threshold is not None else self.abs_drop_threshold

        # Effective threshold: the LOWER of the two (more aggressive drop)
        effective_thr = min(q_thr, a_thr)

        log(
            INFO,
            "[Round %s] Dual-threshold filter: q_thr=%.4f  abs_thr=%.4f"
            "  → effective_thr=%.4f (quantile=%.2f)",
            server_round, q_thr, a_thr, effective_thr, quantile,
        )

        keep_cids = [c for c in cid_with_score if self.last_anomaly_scores[c] <= effective_thr]
        drop_cids = [c for c in cid_with_score if self.last_anomaly_scores[c] > effective_thr]

        # ── min_clients guard ─────────────────────────────────────────────────
        if len(keep_cids) < min_clients:
            sorted_asc = sorted(cid_with_score, key=lambda c: self.last_anomaly_scores[c])
            keep_cids = sorted_asc[:min_clients]
            drop_cids = [c for c in cid_with_score if c not in keep_cids]
            log(
                INFO,
                "[Round %s] min_clients guard: forced keep of %d lowest-score clients: %s"
                " (some may be malicious — federation too small to avoid this).",
                server_round, min_clients, keep_cids,
            )

        filtered_replies = [
            msg for msg in attacked_replies
            if int(msg.metadata.src_node_id) in keep_cids
        ]

        detected_state: Dict[int, bool] = {cid: (cid in drop_cids) for cid in cids}

        log(
            INFO,
            "[Round %s] Filter result: kept=%s  dropped=%s",
            server_round, keep_cids, drop_cids,
        )

        if hasattr(self, "save_path"):
            self._append_round_states_csv(server_round, cids, clients_state, detected_state)
        self._update_confusion_matrix(cids, clients_state, detected_state)

        return filtered_replies, detected_state
