"""pytorch-example: A Flower / PyTorch app."""

import io
import json
import os
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional, Dict, List, Tuple

import torch
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, Krum
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info
from collections import deque
from dataclasses import dataclass, field
from rl_flwr.attack import (
    gaussian_attack_messages,
    lie_attack_messages,
    minmax_attack_messages,
    fang_attack_messages,
    no_attack_messages,
)

# ==== NEW ====
from rl_flwr.convlstm_detector import ConvLSTMForecaster


PROJECT_NAME = "FLOWER-advanced-pytorch"

class CustomFedAvg(FedAvg):

    def __init__(
        self,
        *args,
        num_malicious: int,
        warmup_rounds: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds

        # Existing fields
        self.clients_state = {}
        self.fixed_malicious_ids: list[int] | None = None
        self.all_client_ids: set[int] = set()

        self.round_log_initialized: bool = False
        self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.best_acc_so_far = 0.0
        self.best_arrays: Optional[ArrayRecord] = None

        # ==== NEW: ConvLSTM / anomaly state ====
        # Input spatial size
        self.feature_H = 16
        self.feature_W = 16
        # Number of channels in each frame (we define 6 below)
        self.feature_C = 6
        # Temporal window length
        self.window_len = 5
        # Number of bootstrap samples for visualization
        self.bootstrap_B = 20

        # Per-client history of frames: cid -> List[Tensor [C,H,W]]
        self.client_feature_history: Dict[int, List[torch.Tensor]] = {}

        # --- NEW: keep longer history to enable sliding windows ---
        self.history_max = 10  # can tune: 15~30

        # ConvLSTM detector
        self.detector = ConvLSTMForecaster(
            in_channels=self.feature_C,
            hidden_dim=16,
            kernel_size=3,
        )
        self.detector.eval()  # inference only here

        # CSV for anomaly visualization
        self.anomaly_csv_initialized = False

        # To store global model at start of current round for delta computation
        self.round_global_arrays: Optional[ArrayRecord] = None
        self.last_anomaly_scores: Dict[int, float] = {}
    # -------------------------------------------------------------------------
    # Existing methods (unchanged except where noted)
    # -------------------------------------------------------------------------
        # ===== Semi-online selective update (NEW) =====
        self.semi_online = True
        # round quality gating based on server eval
        self.prev_server_acc: Optional[float] = None
        self.prev_server_loss: Optional[float] = None
        self.good_streak = 0
        self.K_good = 2  # require K consecutive good rounds

        # thresholds for "good" (tune to your metrics)
        self.acc_improve_min = 0.0
        self.loss_improve_min = 0.0

        # client selection in good rounds
        self.good_client_quantile = 0.5  # keep lowest 50% anomaly scores

        # training schedule
        self.update_every = 2
        self.buffer_min_windows = 64
        self.max_steps_per_update = 50
        self.detector_lr = 1e-4
        self.batch_size = 8

        # clean window buffer
        self.clean_window_buffer: List[torch.Tensor] = []  # each is [T,C,H,W]
        self.max_buffer = 2000
        self.last_detector_update_round = 0
        self.detector_optimizer = None

        self.round_sequences: Dict[int, torch.Tensor] = {}
        # ==== TensorBoard logging ====
        self.tb_writer: Optional[SummaryWriter] = None

    def _extract_sliding_windows(self, hist: List[torch.Tensor]) -> List[torch.Tensor]:
        """hist: List[[C,H,W]] length L -> windows List[[T,C,H,W]] with T=window_len."""
        L = len(hist)
        T = self.window_len
        if L < T:
            return []
        windows: List[torch.Tensor] = []
        for start in range(0, L - T + 1):
            seq = torch.stack(hist[start:start + T], dim=0)  # [T,C,H,W]
            windows.append(seq)
        return windows


    def _collect_clean_windows_for_training(
        self,
        replies: List[Message],
        *,
        max_windows_per_client: int = 3,
    ) -> None:
        """
        Collect sliding windows [T,C,H,W] from clients that look benign
        (low anomaly) in a good round.
        """
        candidates: List[int] = []
        for msg in replies:
            cid = int(msg.metadata.src_node_id)
            if cid not in self.last_anomaly_scores:
                continue
            hist = self.client_feature_history.get(cid, [])
            if len(hist) < self.window_len:
                continue
            candidates.append(cid)

        if not candidates:
            return

        scores = np.array([self.last_anomaly_scores[c] for c in candidates], dtype=np.float32)
        thr = float(np.quantile(scores, self.good_client_quantile))

        for cid in candidates:
            if self.last_anomaly_scores[cid] > thr:
                continue

            hist = self.client_feature_history[cid]
            windows = self._extract_sliding_windows(hist)
            if not windows:
                continue

            # cap windows per client to avoid one client dominating the buffer
            if len(windows) > max_windows_per_client:
                idx = np.random.choice(len(windows), size=max_windows_per_client, replace=False)
                windows = [windows[i] for i in idx]

            for seq in windows:
                self.clean_window_buffer.append(seq.detach().cpu())

        # cap buffer size
        if len(self.clean_window_buffer) > self.max_buffer:
            self.clean_window_buffer = self.clean_window_buffer[-self.max_buffer:]



    def _maybe_update_detector(
        self,
        current_round: int,
        *,
        round_is_good: bool,
        skip_update: bool = False,
    ) -> None:
        """
        Semi-online selective update:
        - only update on good rounds
        - only update every N rounds
        - requires enough windows in buffer
        """
        if not self.semi_online or (not round_is_good) or skip_update:
            return
        if current_round - self.last_detector_update_round < self.update_every:
            return
        if len(self.clean_window_buffer) < self.buffer_min_windows:
            return

        # Create optimizer if needed
        if self.detector_optimizer is None:
            self.detector_optimizer = torch.optim.Adam(
                self.detector.parameters(),
                lr=self.detector_lr,
            )

        device = next(self.detector.parameters()).device
        self.detector.train()

        losses: List[float] = []
        buffer_size = len(self.clean_window_buffer)

        # Train for a bounded number of steps; sample batches with replacement
        for step in range(self.max_steps_per_update):
            if buffer_size < self.batch_size:
                break

            batch_ids = np.random.randint(0, buffer_size, size=(self.batch_size,))
            batch = torch.stack([self.clean_window_buffer[i] for i in batch_ids], dim=0)  # [B,T,C,H,W]
            batch = batch.to(device)

            loss = self.detector.training_loss(batch)

            self.detector_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.detector_optimizer.step()

            losses.append(float(loss.item()))

        self.detector.eval()
        self.last_detector_update_round = current_round

        # TensorBoard: log mean loss and buffer size
        if self.tb_writer is not None and losses:
            self.tb_writer.add_scalar("detector/train_loss_mean", float(np.mean(losses)), current_round)
            self.tb_writer.add_scalar("detector/buffer_size", float(len(self.clean_window_buffer)), current_round)


    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
        save_path: Path,
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        # ... [UNCHANGED: your original implementation] ...
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        if server_round <= self.warmup_rounds:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered_replies = sorted(
                train_replies, key=lambda m: m.metadata.src_node_id
            )
            log(
                INFO,
                "[Round %s] Warmup: no new malicious selection. fixed_malicious_ids=%s",
                server_round,
                malicious_ids,
            )
            self.clients_state = clients_state
            return ordered_replies, clients_state, malicious_ids

        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            if size > 0:
                self.fixed_malicious_ids = list(
                    np.random.choice(unique_ids, size=size, replace=False)
                )
            else:
                self.fixed_malicious_ids = []
            log(
                INFO,
                "[Round %s] Selected fixed malicious_ids=%s",
                server_round,
                self.fixed_malicious_ids,
            )

        malicious_ids = self.fixed_malicious_ids
        clients_state: Dict[int, bool] = {
            nid: (nid in malicious_ids) for nid in unique_ids
        }

        ordered_replies = sorted(
            train_replies,
            key=lambda m: m.metadata.src_node_id,
        )

        ordered_ids = [m.metadata.src_node_id for m in ordered_replies]
        log(INFO, "[Round %s] ordered node_ids = %s", server_round, ordered_ids)
        log(INFO, "[Round %s] clients_state = %s", server_round, clients_state)
        log(INFO, "[Round %s] malicious_ids (fixed) = %s", server_round, malicious_ids)

        self.clients_state = clients_state
        return ordered_replies, clients_state, malicious_ids

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

        # --- NEW: initialize TensorBoard writer ---
        log_dir = os.path.join(self.save_path, "tb_logs", self.run_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        logger.log(INFO, "TensorBoard logs will be written to %s", log_dir)

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        # ... [UNCHANGED: your original implementation] ...
        if os.path.exists(f"{self.save_path}/results.json"):
            with open(f"{self.save_path}/results.json", "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        last_train_metrics = dict(result.train_metrics_clientapp.get(current_round, {}))
        last_eval_client_metrics = dict(
            result.evaluate_metrics_clientapp.get(current_round, {})
        )
        last_eval_server_metrics = dict(
            result.evaluate_metrics_serverapp.get(current_round, {})
        )
        round_results = {
            "round": current_round,
            "train_metrics": last_train_metrics,
            "evaluate_metrics_clientapp": last_eval_client_metrics,
            "evaluate_metrics_serverapp": last_eval_server_metrics,
        }
        results.append(round_results)
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["lr"] = config["lr"]
        return super().configure_train(server_round, arrays, config, grid)

    def _append_round_states_csv(
        self,
        server_round: int,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        # ... [UNCHANGED] ...
        csv_path = f"{self.save_path}/client_states_rounds.csv"
        mode = "a" if self.round_log_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self.round_log_initialized:
                writer.writerow(["round", "client_id", "is_malicious", "predict_state"])
                self.round_log_initialized = True

            for cid in sorted(round_client_ids):
                is_mal = 1 if clients_state.get(cid, False) else 0
                is_pred = 1 if detected_state.get(cid, False) else 0
                writer.writerow([server_round, cid, is_mal, is_pred])

    def _update_confusion_matrix(
        self,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        # ... [UNCHANGED] ...
        for cid in round_client_ids:
            gt = bool(clients_state.get(cid, False))
            pred = bool(detected_state.get(cid, False))
            if gt and pred:
                self.confusion_matrix["TP"] += 1
            elif (not gt) and (not pred):
                self.confusion_matrix["TN"] += 1
            elif (not gt) and pred:
                self.confusion_matrix["FP"] += 1
            elif gt and (not pred):
                self.confusion_matrix["FN"] += 1

    # -------------------------------------------------------------------------
    # ==== NEW: helper for anomaly visualization CSV ====
    # -------------------------------------------------------------------------

    def _append_anomaly_csv(
        self,
        server_round: int,
        client_id: int,
        mean_score: float,
        var_score: float,
        is_malicious: bool,
    ) -> None:
        csv_path = f"{self.save_path}/convlstm_anomaly.csv"
        mode = "a" if self.anomaly_csv_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self.anomaly_csv_initialized:
                writer.writerow(
                    ["round", "client_id", "mean_score", "var_score", "is_malicious"]
                )
                self.anomaly_csv_initialized = True
            writer.writerow(
                [server_round, client_id, mean_score, var_score, int(is_malicious)]
            )

    # -------------------------------------------------------------------------
    # ==== NEW: _build_frame_for_client ====
    # -------------------------------------------------------------------------

    def _build_frame_for_client(
        self,
        cid: int,
        v: torch.Tensor,          # [D] flattened/padded delta
        idx: int,                 # index in stacked tensors (for norms, cos, val_deltas)
        stats: dict,
        val_delta_i: float,
    ) -> torch.Tensor:
        H = self.feature_H
        W = self.feature_W

        mu_v = stats["mu_v"]        # [D]
        sigma_v = stats["sigma_v"]  # [D]
        mu_m = stats["mu_m"]        # [D]
        sigma_m = stats["sigma_m"]  # [D]

        norms = stats["norms"]      # [N]
        mu_N = stats["mu_N"]
        sigma_N = stats["sigma_N"]

        cos = stats["cos"]          # [N]
        mu_C = stats["mu_C"]
        sigma_C = stats["sigma_C"]

        mu_L = stats["mu_L"]
        sigma_L = stats["sigma_L"]

        # ch1: normalized update values
        x1 = (v - mu_v) / sigma_v
        x1 = x1.view(1, H, W)

        # ch2: sign pattern
        x2 = torch.sign(v).view(1, H, W)

        # ch3: normalized magnitude
        m = v.abs()
        x3 = (m - mu_m) / sigma_m
        x3 = x3.view(1, H, W)

        # ch4: normalized update norm (scalar tiled)
        N_i = norms[idx]
        N_hat = (N_i - mu_N) / sigma_N
        x4 = torch.full((1, H, W), float(N_hat))

        # ch5: normalized cosine similarity (scalar tiled)
        C_i = cos[idx]
        C_hat = (C_i - mu_C) / sigma_C
        x5 = torch.full((1, H, W), float(C_hat))

        # ch6: normalized client-side validation delta (scalar tiled)
        L_i = torch.tensor(float(val_delta_i), dtype=torch.float32)
        L_hat = (L_i - mu_L) / sigma_L
        x6 = torch.full((1, H, W), float(L_hat))

        F = torch.cat([x1, x2, x3, x4, x5, x6], dim=0)  # [C, H, W]
        return F

    # -------------------------------------------------------------------------
    # ==== NEW: main ConvLSTM + bootstrap routine ====
    # -------------------------------------------------------------------------

    def _run_convlstm_analysis(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        global_arrays: ArrayRecord,
        client_val_deltas: Dict[int, float],  # cid -> ΔL_client (L_after - L_before)
    ) -> None:
        

        # 1) Build delta vectors [D] for each client
        target_len = self.feature_H * self.feature_W
        delta_vectors: Dict[int, torch.Tensor] = {}
        gsd = global_arrays.to_torch_state_dict()

        def layer_filter(name: str) -> bool:
            # Example: only use last block and FC; adjust to your model
            return ("layer4" in name) or ("fc" in name)

        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()

            vecs = []
            for name, p in csd.items():
                if not p.is_floating_point():
                    continue
                if not layer_filter(name):
                    continue
                delta = (p - gsd[name]).detach().cpu().flatten()
                vecs.append(delta)

            if not vecs:
                v = torch.zeros(target_len)
            else:
                v = torch.cat(vecs, dim=0)
                if v.numel() < target_len:
                    pad = torch.zeros(target_len - v.numel())
                    v = torch.cat([v, pad], dim=0)
                else:
                    v = v[:target_len]

            delta_vectors[cid] = v

        if not delta_vectors:
            return

        cids = list(delta_vectors.keys())
        all_vecs = torch.stack([delta_vectors[cid] for cid in cids], dim=0)  # [N, D]
        N, D = all_vecs.shape
        eps = 1e-6

        # 2) Per-round stats
        mu_v = all_vecs.mean(dim=0)
        sigma_v = all_vecs.std(dim=0, unbiased=False) + eps

        mag = all_vecs.abs()
        mu_m = mag.mean(dim=0)
        sigma_m = mag.std(dim=0, unbiased=False) + eps

        norms = torch.linalg.norm(all_vecs, dim=1)
        mu_N = norms.mean()
        sigma_N = norms.std(unbiased=False) + eps

        mean_vec = all_vecs.mean(dim=0)
        mean_vec_norm = torch.linalg.norm(mean_vec) + eps

        cos_list = []
        for i in range(N):
            v = all_vecs[i]
            c = torch.dot(v, mean_vec) / ((torch.linalg.norm(v) + eps) * mean_vec_norm)
            cos_list.append(c)
        cos = torch.stack(cos_list)
        mu_C = cos.mean()
        sigma_C = cos.std(unbiased=False) + eps

        val_deltas_list = []
        for cid in cids:
            dL = float(client_val_deltas.get(cid, 0.0))
            val_deltas_list.append(dL)
        val_deltas_tensor = torch.tensor(val_deltas_list, dtype=torch.float32)
        mu_L = val_deltas_tensor.mean()
        sigma_L = val_deltas_tensor.std(unbiased=False) + eps

        stats = {
            "mu_v": mu_v,
            "sigma_v": sigma_v,
            "mu_m": mu_m,
            "sigma_m": sigma_m,
            "norms": norms,
            "mu_N": mu_N,
            "sigma_N": sigma_N,
            "cos": cos,
            "mu_C": mu_C,
            "sigma_C": sigma_C,
            "mu_L": mu_L,
            "sigma_L": sigma_L,
        }

        # 3) Build frames and update histories
        for idx, cid in enumerate(cids):
            v = all_vecs[idx]
            val_delta_i = val_deltas_tensor[idx].item()
            F = self._build_frame_for_client(
                cid=cid,
                v=v,
                idx=idx,
                stats=stats,
                val_delta_i=val_delta_i,
            )
            hist = self.client_feature_history.setdefault(cid, [])
            hist.append(F)
            if len(hist) > self.history_max:
                hist.pop(0)

        # 4) ConvLSTM + bootstrap anomaly for visualization
        for cid, hist in self.client_feature_history.items():
            if len(hist) < self.window_len:
                continue

            seq = torch.stack(hist, dim=0)  # [T, C, H, W]
            T = seq.shape[0]
            scores = []

            for _ in range(self.bootstrap_B):
                idxs = torch.randint(low=0, high=T, size=(T,))
                boot_seq = seq[idxs]  # [T, C, H, W]
                score = self.detector.anomaly_score(boot_seq)
                scores.append(score)

            scores = np.array(scores, dtype=np.float32)
            mean_score = float(scores.mean())
            var_score = float(scores.var())

            is_malicious = bool(clients_state.get(cid, False))
            self._append_anomaly_csv(
                server_round=server_round,
                client_id=cid,
                mean_score=mean_score,
                var_score=var_score,
                is_malicious=is_malicious,
            )
            self.last_anomaly_scores[cid] = mean_score

    def _round_is_good(self, acc_val: float, loss_val: float) -> bool:
        """Return True if we have K_good consecutive 'good' rounds based on server eval."""
        if self.prev_server_acc is None or self.prev_server_loss is None:
            self.prev_server_acc = acc_val
            self.prev_server_loss = loss_val
            self.good_streak = 0
            return False

        d_acc = acc_val - self.prev_server_acc
        d_loss = self.prev_server_loss - loss_val  # >0 means loss decreased

        acc_improved = d_acc >= self.acc_improve_min
        loss_improved = d_loss >= self.loss_improve_min

        is_good = acc_improved or loss_improved
        self.good_streak = self.good_streak + 1 if is_good else 0

        self.prev_server_acc = acc_val
        self.prev_server_loss = loss_val

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("detector/good_streak", float(self.good_streak), int(getattr(self, "current_round_for_tb", 0)))
            self.tb_writer.add_scalar("detector/d_acc", float(d_acc), int(getattr(self, "current_round_for_tb", 0)))
            self.tb_writer.add_scalar("detector/d_loss", float(d_loss), int(getattr(self, "current_round_for_tb", 0)))

        return self.good_streak >= self.K_good

    def _filter_attacked_replies_by_anomaly(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        *,
        quantile: float = 0.8,
        min_clients: int = 2,
        abs_threshold: float | None = None,
    ) -> Tuple[List[Message], Dict[int, bool]]:
        """
        Filter out high-anomaly clients BEFORE aggregation.

        Returns:
            filtered_replies: list of messages used for aggregation
            detected_state: dict[cid -> True if predicted malicious (filtered out)]
        """

        # collect anomaly scores for the clients actually in this round
        cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]
        scores = []
        cid_with_score = []
        for cid in cids:
            if cid in self.last_anomaly_scores:
                scores.append(self.last_anomaly_scores[cid])
                cid_with_score.append(cid)

        # If we don't have anomaly scores (e.g., not enough history yet),
        # do not filter anything.
        if not scores:
            log(
                INFO,
                "[Round %s] No ConvLSTM scores available yet -> no filtering",
                server_round,
            )
            detected_state = {cid: False for cid in cids}
            return attacked_replies, detected_state

        scores_arr = np.array(scores, dtype=np.float32)

        # Decide threshold
        if abs_threshold is not None:
            thr = abs_threshold
        else:
            # drop top (1 - quantile) fraction by score (i.e., keep lowest 80% by default)
            thr = float(np.quantile(scores_arr, quantile))
        log(
            INFO,
            "[Round %s] Anomaly filtering: threshold=%.6f (quantile=%.2f)",
            server_round,
            thr,
            quantile,
        )

        # Initial keep/drop decision
        keep_cids = [cid for cid in cid_with_score if self.last_anomaly_scores[cid] <= thr]
        drop_cids = [cid for cid in cid_with_score if self.last_anomaly_scores[cid] > thr]

        # Make sure we keep at least min_clients
        if len(keep_cids) < min_clients:
            # sort by anomaly (low to high) and keep best min_clients
            sorted_by_score = sorted(
                cid_with_score,
                key=lambda c: self.last_anomaly_scores[c],
            )
            keep_cids = sorted_by_score[:min_clients]
            drop_cids = [cid for cid in cid_with_score if cid not in keep_cids]
            log(
                INFO,
                "[Round %s] Not enough clients after filtering, forcing keep of %d clients: %s",
                server_round,
                min_clients,
                keep_cids,
            )

        # Build filtered replies
        filtered_replies = [
            msg for msg in attacked_replies
            if int(msg.metadata.src_node_id) in keep_cids
        ]

        # Build detected_state: True if we decided to drop (treat as malicious)
        detected_state: Dict[int, bool] = {}
        for cid in cids:
            detected_state[cid] = cid in drop_cids

        # Optional: log some stats
        log(
            INFO,
            "[Round %s] Filtered clients by anomaly: kept=%s, dropped=%s",
            server_round,
            keep_cids,
            drop_cids,
        )

        # Optional: update confusion matrix / CSV
        self._append_round_states_csv(
            server_round=server_round,
            round_client_ids=cids,
            clients_state=clients_state,
            detected_state=detected_state,
        )
        self._update_confusion_matrix(
            round_client_ids=cids,
            clients_state=clients_state,
            detected_state=detected_state,
        )

        return filtered_replies, detected_state


    def _filter_evaluate_replies_by_anomaly(
        self,
        server_round: int,
        evaluate_replies: List[Message],
        *,
        quantile: float = 0.8,
        abs_threshold: float | None = None,
    ) -> List[Message]:
        """
        Filter out high-anomaly clients for evaluation.

        Uses self.last_anomaly_scores computed during the train phase
        of the same round.

        Returns:
            filtered_replies: list of messages used for aggregate_evaluate
        """

        # Collect CIDs and scores for this eval round
        cids = [int(msg.metadata.src_node_id) for msg in evaluate_replies]
        scores = []
        cid_with_score = []
        for cid in cids:
            if cid in self.last_anomaly_scores:
                scores.append(self.last_anomaly_scores[cid])
                cid_with_score.append(cid)

        # If we have no scores (e.g. early rounds, not enough history),
        # do not filter any eval replies.
        if not scores:
            log(
                INFO,
                "[Round %s] No ConvLSTM scores available for eval -> no filtering",
                server_round,
            )
            return evaluate_replies

        scores_arr = np.array(scores, dtype=np.float32)

        # Decide threshold (either absolute or quantile-based)
        if abs_threshold is not None:
            thr = abs_threshold
        else:
            thr = float(np.quantile(scores_arr, quantile))
        log(
            INFO,
            "[Round %s] Eval anomaly filtering: threshold=%.6f (quantile=%.2f)",
            server_round,
            thr,
            quantile,
        )

        # Keep only clients whose anomaly score <= threshold
        keep_cids = [
            cid for cid in cid_with_score
            if self.last_anomaly_scores[cid] <= thr
        ]
        drop_cids = [
            cid for cid in cid_with_score
            if self.last_anomaly_scores[cid] > thr
        ]

        filtered_replies = [
            msg for msg in evaluate_replies
            if int(msg.metadata.src_node_id) in keep_cids
        ]

        log(
            INFO,
            "[Round %s] Eval: kept clients=%s, dropped (high-anomaly)=%s",
            server_round,
            keep_cids,
            drop_cids,
        )

        # If everything got dropped (extreme case), fall back to no filtering
        if not filtered_replies:
            log(
                INFO,
                "[Round %s] Eval filtering dropped all clients, falling back to unfiltered eval",
                server_round,
            )
            return evaluate_replies

        return filtered_replies


    # -------------------------------------------------------------------------
    # start(): hook ConvLSTM into training loop
    # -------------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 20,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # For TensorBoard logging inside helper methods
            self.current_round_for_tb = current_round

            # Store global model at start of round (for delta computation)
            self.round_global_arrays = arrays

            # TRAIN
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            ordered_train_replies, clients_state, malicious_cids = self._select_malicious_and_order(
                current_round, train_replies, save_path=self.save_path
            )

            # Choose attack
            attacked_replies = minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_cids),
            )

            # --- Gaussian attack ---
            # attacked_replies = gaussian_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            #     magnitude=3,          
            # )

            # --- LIE attack ---
            # attacked_replies = lie_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            #     omniscent=True,         # hoặc False nếu bạn muốn
            # )

            # ==== NEW: gather client-side validation deltas (if available) ====
            # You must ensure the client attaches "val_loss_delta" in msg.content["metrics"].
            client_val_deltas: Dict[int, float] = {}
            for msg in attacked_replies:
                cid = int(msg.metadata.src_node_id)
                # adjust this depending on how you store metrics
                m = msg.content.get("metrics", {}) if isinstance(msg.content, dict) else {}
                dL = float(m.get("val_loss_delta", 0.0))
                client_val_deltas[cid] = dL

            # ==== NEW: run ConvLSTM analysis for visualization ====
            self._run_convlstm_analysis(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                global_arrays=self.round_global_arrays,
                client_val_deltas=client_val_deltas,
            )

            # Filter out high-anomaly clients BEFORE aggregation
            filtered_replies, detected_state = self._filter_attacked_replies_by_anomaly(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                quantile=0.6,      # keep lowest 80% anomaly scores
                min_clients=2,      # ensure at least 2 clients remain
                abs_threshold=None, # or set a fixed threshold if you want
            )
            drop_rate = 1.0
            if len(attacked_replies) > 0:
                drop_rate = 1.0 - (len(filtered_replies) / len(attacked_replies))

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                #attacked_replies,
                filtered_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # EVAL (clients)
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # ==== NEW: filter high-anomaly clients from eval ====
            filtered_eval_replies = self._filter_evaluate_replies_by_anomaly(
                server_round=current_round,
                evaluate_replies=evaluate_replies,
                quantile=0.6,       # keep lowest 80% anomaly scores
                abs_threshold=None, # or set a fixed absolute threshold
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                filtered_eval_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # EVAL (server)
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    metrics = res
                    acc_val = float(metrics.get("accuracy", 0.0))
                    loss_val = float(metrics.get("loss", 0.0))

                    # ===== Semi-online selective update (NEW) =====
                    round_is_good = self._round_is_good(acc_val, loss_val)

                    # skip updating detector if this round was unstable (too many dropped clients)
                    skip_update = (drop_rate > 0.4)  # tune threshold as needed

                    if round_is_good:
                        # Collect training windows from "attacked_replies" (less feedback loop),
                        # using low-anomaly quantile gating.
                        self._collect_clean_windows_for_training(filtered_replies, max_windows_per_client=3)

                    # Update detector (mini-batch)
                    self._maybe_update_detector(
                        current_round=current_round,
                        round_is_good=round_is_good,
                        skip_update=skip_update,
                    )
                    # --- NEW: TensorBoard logging for global metrics ---
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("global/accuracy", acc_val, current_round)
                        self.tb_writer.add_scalar("global/loss", loss_val, current_round)

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result

class CustomKrum(Krum):

    def __init__(
        self,
        *args,
        num_malicious: int,
        warmup_rounds: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds

        # Existing fields
        self.clients_state = {}
        self.fixed_malicious_ids: list[int] | None = None
        self.all_client_ids: set[int] = set()

        self.round_log_initialized: bool = False
        self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.best_acc_so_far = 0.0
        self.best_arrays: Optional[ArrayRecord] = None

        # ==== NEW: ConvLSTM / anomaly state ====
        # Input spatial size
        self.feature_H = 16
        self.feature_W = 16
        # Number of channels in each frame (we define 6 below)
        self.feature_C = 6
        # Temporal window length
        self.window_len = 5
        # Number of bootstrap samples for visualization
        self.bootstrap_B = 20

        # Per-client history of frames: cid -> List[Tensor [C,H,W]]
        self.client_feature_history: Dict[int, List[torch.Tensor]] = {}

        # ConvLSTM detector
        self.detector = ConvLSTMForecaster(
            in_channels=self.feature_C,
            hidden_dim=16,
            kernel_size=3,
        )
        self.detector.eval()  # inference only here

        # CSV for anomaly visualization
        self.anomaly_csv_initialized = False

        # To store global model at start of current round for delta computation
        self.round_global_arrays: Optional[ArrayRecord] = None
        self.last_anomaly_scores: Dict[int, float] = {}
    # -------------------------------------------------------------------------
    # Existing methods (unchanged except where noted)
    # -------------------------------------------------------------------------

    
    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
        save_path: Path,
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        # ... [UNCHANGED: your original implementation] ...
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        if server_round <= self.warmup_rounds:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered_replies = sorted(
                train_replies, key=lambda m: m.metadata.src_node_id
            )
            log(
                INFO,
                "[Round %s] Warmup: no new malicious selection. fixed_malicious_ids=%s",
                server_round,
                malicious_ids,
            )
            self.clients_state = clients_state
            return ordered_replies, clients_state, malicious_ids

        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            if size > 0:
                self.fixed_malicious_ids = list(
                    np.random.choice(unique_ids, size=size, replace=False)
                )
            else:
                self.fixed_malicious_ids = []
            log(
                INFO,
                "[Round %s] Selected fixed malicious_ids=%s",
                server_round,
                self.fixed_malicious_ids,
            )

        malicious_ids = self.fixed_malicious_ids
        clients_state: Dict[int, bool] = {
            nid: (nid in malicious_ids) for nid in unique_ids
        }

        ordered_replies = sorted(
            train_replies,
            key=lambda m: m.metadata.src_node_id,
        )

        ordered_ids = [m.metadata.src_node_id for m in ordered_replies]
        log(INFO, "[Round %s] ordered node_ids = %s", server_round, ordered_ids)
        log(INFO, "[Round %s] clients_state = %s", server_round, clients_state)
        log(INFO, "[Round %s] malicious_ids (fixed) = %s", server_round, malicious_ids)

        self.clients_state = clients_state
        return ordered_replies, clients_state, malicious_ids

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

        # --- NEW: initialize TensorBoard writer ---
        log_dir = os.path.join(self.save_path, "tb_logs", self.run_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        logger.log(INFO, "TensorBoard logs will be written to %s", log_dir)

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        # ... [UNCHANGED: your original implementation] ...
        if os.path.exists(f"{self.save_path}/results.json"):
            with open(f"{self.save_path}/results.json", "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        last_train_metrics = dict(result.train_metrics_clientapp.get(current_round, {}))
        last_eval_client_metrics = dict(
            result.evaluate_metrics_clientapp.get(current_round, {})
        )
        last_eval_server_metrics = dict(
            result.evaluate_metrics_serverapp.get(current_round, {})
        )
        round_results = {
            "round": current_round,
            "train_metrics": last_train_metrics,
            "evaluate_metrics_clientapp": last_eval_client_metrics,
            "evaluate_metrics_serverapp": last_eval_server_metrics,
        }
        results.append(round_results)
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["lr"] = config["lr"]
        return super().configure_train(server_round, arrays, config, grid)

    def _append_round_states_csv(
        self,
        server_round: int,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        # ... [UNCHANGED] ...
        csv_path = f"{self.save_path}/client_states_rounds.csv"
        mode = "a" if self.round_log_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self.round_log_initialized:
                writer.writerow(["round", "client_id", "is_malicious", "predict_state"])
                self.round_log_initialized = True

            for cid in sorted(round_client_ids):
                is_mal = 1 if clients_state.get(cid, False) else 0
                is_pred = 1 if detected_state.get(cid, False) else 0
                writer.writerow([server_round, cid, is_mal, is_pred])

    def _update_confusion_matrix(
        self,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        # ... [UNCHANGED] ...
        for cid in round_client_ids:
            gt = bool(clients_state.get(cid, False))
            pred = bool(detected_state.get(cid, False))
            if gt and pred:
                self.confusion_matrix["TP"] += 1
            elif (not gt) and (not pred):
                self.confusion_matrix["TN"] += 1
            elif (not gt) and pred:
                self.confusion_matrix["FP"] += 1
            elif gt and (not pred):
                self.confusion_matrix["FN"] += 1

    # -------------------------------------------------------------------------
    # ==== NEW: helper for anomaly visualization CSV ====
    # -------------------------------------------------------------------------

    def _append_anomaly_csv(
        self,
        server_round: int,
        client_id: int,
        mean_score: float,
        var_score: float,
        is_malicious: bool,
    ) -> None:
        csv_path = f"{self.save_path}/convlstm_anomaly.csv"
        mode = "a" if self.anomaly_csv_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self.anomaly_csv_initialized:
                writer.writerow(
                    ["round", "client_id", "mean_score", "var_score", "is_malicious"]
                )
                self.anomaly_csv_initialized = True
            writer.writerow(
                [server_round, client_id, mean_score, var_score, int(is_malicious)]
            )

    # -------------------------------------------------------------------------
    # ==== NEW: _build_frame_for_client ====
    # -------------------------------------------------------------------------

    def _build_frame_for_client(
        self,
        cid: int,
        v: torch.Tensor,          # [D] flattened/padded delta
        idx: int,                 # index in stacked tensors (for norms, cos, val_deltas)
        stats: dict,
        val_delta_i: float,
    ) -> torch.Tensor:
        H = self.feature_H
        W = self.feature_W

        mu_v = stats["mu_v"]        # [D]
        sigma_v = stats["sigma_v"]  # [D]
        mu_m = stats["mu_m"]        # [D]
        sigma_m = stats["sigma_m"]  # [D]

        norms = stats["norms"]      # [N]
        mu_N = stats["mu_N"]
        sigma_N = stats["sigma_N"]

        cos = stats["cos"]          # [N]
        mu_C = stats["mu_C"]
        sigma_C = stats["sigma_C"]

        mu_L = stats["mu_L"]
        sigma_L = stats["sigma_L"]

        # ch1: normalized update values
        x1 = (v - mu_v) / sigma_v
        x1 = x1.view(1, H, W)

        # ch2: sign pattern
        x2 = torch.sign(v).view(1, H, W)

        # ch3: normalized magnitude
        m = v.abs()
        x3 = (m - mu_m) / sigma_m
        x3 = x3.view(1, H, W)

        # ch4: normalized update norm (scalar tiled)
        N_i = norms[idx]
        N_hat = (N_i - mu_N) / sigma_N
        x4 = torch.full((1, H, W), float(N_hat))

        # ch5: normalized cosine similarity (scalar tiled)
        C_i = cos[idx]
        C_hat = (C_i - mu_C) / sigma_C
        x5 = torch.full((1, H, W), float(C_hat))

        # ch6: normalized client-side validation delta (scalar tiled)
        L_i = torch.tensor(float(val_delta_i), dtype=torch.float32)
        L_hat = (L_i - mu_L) / sigma_L
        x6 = torch.full((1, H, W), float(L_hat))

        F = torch.cat([x1, x2, x3, x4, x5, x6], dim=0)  # [C, H, W]
        return F

    # -------------------------------------------------------------------------
    # ==== NEW: main ConvLSTM + bootstrap routine ====
    # -------------------------------------------------------------------------

    def _run_convlstm_analysis(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        global_arrays: ArrayRecord,
        client_val_deltas: Dict[int, float],  # cid -> ΔL_client (L_after - L_before)
    ) -> None:
        

        # 1) Build delta vectors [D] for each client
        target_len = self.feature_H * self.feature_W
        delta_vectors: Dict[int, torch.Tensor] = {}
        gsd = global_arrays.to_torch_state_dict()

        def layer_filter(name: str) -> bool:
            # Example: only use last block and FC; adjust to your model
            return ("layer4" in name) or ("fc" in name)

        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()

            vecs = []
            for name, p in csd.items():
                if not p.is_floating_point():
                    continue
                if not layer_filter(name):
                    continue
                delta = (p - gsd[name]).detach().cpu().flatten()
                vecs.append(delta)

            if not vecs:
                v = torch.zeros(target_len)
            else:
                v = torch.cat(vecs, dim=0)
                if v.numel() < target_len:
                    pad = torch.zeros(target_len - v.numel())
                    v = torch.cat([v, pad], dim=0)
                else:
                    v = v[:target_len]

            delta_vectors[cid] = v

        if not delta_vectors:
            return

        cids = list(delta_vectors.keys())
        all_vecs = torch.stack([delta_vectors[cid] for cid in cids], dim=0)  # [N, D]
        N, D = all_vecs.shape
        eps = 1e-6

        # 2) Per-round stats
        mu_v = all_vecs.mean(dim=0)
        sigma_v = all_vecs.std(dim=0, unbiased=False) + eps

        mag = all_vecs.abs()
        mu_m = mag.mean(dim=0)
        sigma_m = mag.std(dim=0, unbiased=False) + eps

        norms = torch.linalg.norm(all_vecs, dim=1)
        mu_N = norms.mean()
        sigma_N = norms.std(unbiased=False) + eps

        mean_vec = all_vecs.mean(dim=0)
        mean_vec_norm = torch.linalg.norm(mean_vec) + eps

        cos_list = []
        for i in range(N):
            v = all_vecs[i]
            c = torch.dot(v, mean_vec) / ((torch.linalg.norm(v) + eps) * mean_vec_norm)
            cos_list.append(c)
        cos = torch.stack(cos_list)
        mu_C = cos.mean()
        sigma_C = cos.std(unbiased=False) + eps

        val_deltas_list = []
        for cid in cids:
            dL = float(client_val_deltas.get(cid, 0.0))
            val_deltas_list.append(dL)
        val_deltas_tensor = torch.tensor(val_deltas_list, dtype=torch.float32)
        mu_L = val_deltas_tensor.mean()
        sigma_L = val_deltas_tensor.std(unbiased=False) + eps

        stats = {
            "mu_v": mu_v,
            "sigma_v": sigma_v,
            "mu_m": mu_m,
            "sigma_m": sigma_m,
            "norms": norms,
            "mu_N": mu_N,
            "sigma_N": sigma_N,
            "cos": cos,
            "mu_C": mu_C,
            "sigma_C": sigma_C,
            "mu_L": mu_L,
            "sigma_L": sigma_L,
        }

        # 3) Build frames and update histories
        for idx, cid in enumerate(cids):
            v = all_vecs[idx]
            val_delta_i = val_deltas_tensor[idx].item()
            F = self._build_frame_for_client(
                cid=cid,
                v=v,
                idx=idx,
                stats=stats,
                val_delta_i=val_delta_i,
            )
            hist = self.client_feature_history.setdefault(cid, [])
            hist.append(F)
            if len(hist) > self.window_len:
                hist.pop(0)

        # 4) ConvLSTM + bootstrap anomaly for visualization
        for cid, hist in self.client_feature_history.items():
            if len(hist) < self.window_len:
                continue

            seq = torch.stack(hist, dim=0)  # [T, C, H, W]
            T = seq.shape[0]
            scores = []

            for _ in range(self.bootstrap_B):
                idxs = torch.randint(low=0, high=T, size=(T,))
                boot_seq = seq[idxs]  # [T, C, H, W]
                score = self.detector.anomaly_score(boot_seq)
                scores.append(score)

            scores = np.array(scores, dtype=np.float32)
            mean_score = float(scores.mean())
            var_score = float(scores.var())

            is_malicious = bool(clients_state.get(cid, False))
            self._append_anomaly_csv(
                server_round=server_round,
                client_id=cid,
                mean_score=mean_score,
                var_score=var_score,
                is_malicious=is_malicious,
            )
            self.last_anomaly_scores[cid] = mean_score


    def _filter_attacked_replies_by_anomaly(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        *,
        quantile: float = 0.8,
        min_clients: int = 2,
        abs_threshold: float | None = None,
    ) -> Tuple[List[Message], Dict[int, bool]]:
        """
        Filter out high-anomaly clients BEFORE aggregation.

        Returns:
            filtered_replies: list of messages used for aggregation
            detected_state: dict[cid -> True if predicted malicious (filtered out)]
        """

        # collect anomaly scores for the clients actually in this round
        cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]
        scores = []
        cid_with_score = []
        for cid in cids:
            if cid in self.last_anomaly_scores:
                scores.append(self.last_anomaly_scores[cid])
                cid_with_score.append(cid)

        # If we don't have anomaly scores (e.g., not enough history yet),
        # do not filter anything.
        if not scores:
            log(
                INFO,
                "[Round %s] No ConvLSTM scores available yet -> no filtering",
                server_round,
            )
            detected_state = {cid: False for cid in cids}
            return attacked_replies, detected_state

        scores_arr = np.array(scores, dtype=np.float32)

        # Decide threshold
        if abs_threshold is not None:
            thr = abs_threshold
        else:
            # drop top (1 - quantile) fraction by score (i.e., keep lowest 80% by default)
            thr = float(np.quantile(scores_arr, quantile))
        log(
            INFO,
            "[Round %s] Anomaly filtering: threshold=%.6f (quantile=%.2f)",
            server_round,
            thr,
            quantile,
        )

        # Initial keep/drop decision
        keep_cids = [cid for cid in cid_with_score if self.last_anomaly_scores[cid] <= thr]
        drop_cids = [cid for cid in cid_with_score if self.last_anomaly_scores[cid] > thr]

        # Make sure we keep at least min_clients
        if len(keep_cids) < min_clients:
            # sort by anomaly (low to high) and keep best min_clients
            sorted_by_score = sorted(
                cid_with_score,
                key=lambda c: self.last_anomaly_scores[c],
            )
            keep_cids = sorted_by_score[:min_clients]
            drop_cids = [cid for cid in cid_with_score if cid not in keep_cids]
            log(
                INFO,
                "[Round %s] Not enough clients after filtering, forcing keep of %d clients: %s",
                server_round,
                min_clients,
                keep_cids,
            )

        # Build filtered replies
        filtered_replies = [
            msg for msg in attacked_replies
            if int(msg.metadata.src_node_id) in keep_cids
        ]

        # Build detected_state: True if we decided to drop (treat as malicious)
        detected_state: Dict[int, bool] = {}
        for cid in cids:
            detected_state[cid] = cid in drop_cids

        # Optional: log some stats
        log(
            INFO,
            "[Round %s] Filtered clients by anomaly: kept=%s, dropped=%s",
            server_round,
            keep_cids,
            drop_cids,
        )

        # Optional: update confusion matrix / CSV
        self._append_round_states_csv(
            server_round=server_round,
            round_client_ids=cids,
            clients_state=clients_state,
            detected_state=detected_state,
        )
        self._update_confusion_matrix(
            round_client_ids=cids,
            clients_state=clients_state,
            detected_state=detected_state,
        )

        return filtered_replies, detected_state


    def _filter_evaluate_replies_by_anomaly(
        self,
        server_round: int,
        evaluate_replies: List[Message],
        *,
        quantile: float = 0.8,
        abs_threshold: float | None = None,
    ) -> List[Message]:
        """
        Filter out high-anomaly clients for evaluation.

        Uses self.last_anomaly_scores computed during the train phase
        of the same round.

        Returns:
            filtered_replies: list of messages used for aggregate_evaluate
        """

        # Collect CIDs and scores for this eval round
        cids = [int(msg.metadata.src_node_id) for msg in evaluate_replies]
        scores = []
        cid_with_score = []
        for cid in cids:
            if cid in self.last_anomaly_scores:
                scores.append(self.last_anomaly_scores[cid])
                cid_with_score.append(cid)

        # If we have no scores (e.g. early rounds, not enough history),
        # do not filter any eval replies.
        if not scores:
            log(
                INFO,
                "[Round %s] No ConvLSTM scores available for eval -> no filtering",
                server_round,
            )
            return evaluate_replies

        scores_arr = np.array(scores, dtype=np.float32)

        # Decide threshold (either absolute or quantile-based)
        if abs_threshold is not None:
            thr = abs_threshold
        else:
            thr = float(np.quantile(scores_arr, quantile))
        log(
            INFO,
            "[Round %s] Eval anomaly filtering: threshold=%.6f (quantile=%.2f)",
            server_round,
            thr,
            quantile,
        )

        # Keep only clients whose anomaly score <= threshold
        keep_cids = [
            cid for cid in cid_with_score
            if self.last_anomaly_scores[cid] <= thr
        ]
        drop_cids = [
            cid for cid in cid_with_score
            if self.last_anomaly_scores[cid] > thr
        ]

        filtered_replies = [
            msg for msg in evaluate_replies
            if int(msg.metadata.src_node_id) in keep_cids
        ]

        log(
            INFO,
            "[Round %s] Eval: kept clients=%s, dropped (high-anomaly)=%s",
            server_round,
            keep_cids,
            drop_cids,
        )

        # If everything got dropped (extreme case), fall back to no filtering
        if not filtered_replies:
            log(
                INFO,
                "[Round %s] Eval filtering dropped all clients, falling back to unfiltered eval",
                server_round,
            )
            return evaluate_replies

        return filtered_replies


    # -------------------------------------------------------------------------
    # start(): hook ConvLSTM into training loop
    # -------------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 20,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Store global model at start of round (for delta computation)
            self.round_global_arrays = arrays

            # TRAIN
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            ordered_train_replies, clients_state, malicious_cids = self._select_malicious_and_order(
                current_round, train_replies, save_path=self.save_path
            )

            # Choose attack
            attacked_replies = minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_cids),
            )

            # --- Gaussian attack ---
            # attacked_replies = gaussian_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            #     magnitude=3,          
            # )

            # --- LIE attack ---
            # attacked_replies = lie_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            #     omniscent=True,         # hoặc False nếu bạn muốn
            # )

            # ==== NEW: gather client-side validation deltas (if available) ====
            # You must ensure the client attaches "val_loss_delta" in msg.content["metrics"].
            client_val_deltas: Dict[int, float] = {}
            for msg in attacked_replies:
                cid = int(msg.metadata.src_node_id)
                # adjust this depending on how you store metrics
                m = msg.content.get("metrics", {}) if isinstance(msg.content, dict) else {}
                dL = float(m.get("val_loss_delta", 0.0))
                client_val_deltas[cid] = dL

            # ==== NEW: run ConvLSTM analysis for visualization ====
            self._run_convlstm_analysis(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                global_arrays=self.round_global_arrays,
                client_val_deltas=client_val_deltas,
            )

            # Filter out high-anomaly clients BEFORE aggregation
            filtered_replies, detected_state = self._filter_attacked_replies_by_anomaly(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                quantile=0.6,      # keep lowest 80% anomaly scores
                min_clients=2,      # ensure at least 2 clients remain
                abs_threshold=None, # or set a fixed threshold if you want
            )


            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                #attacked_replies,
                filtered_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # EVAL (clients)
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # ==== NEW: filter high-anomaly clients from eval ====
            filtered_eval_replies = self._filter_evaluate_replies_by_anomaly(
                server_round=current_round,
                evaluate_replies=evaluate_replies,
                quantile=0.6,       # keep lowest 80% anomaly scores
                abs_threshold=None, # or set a fixed absolute threshold
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                filtered_eval_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # EVAL (server)
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    metrics = res
                    acc_val = float(metrics.get("accuracy", 0.0))
                    loss_val = float(metrics.get("loss", 0.0))

                    # --- NEW: TensorBoard logging for global metrics ---
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("global/accuracy", acc_val, current_round)
                        self.tb_writer.add_scalar("global/loss", loss_val, current_round)

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    



class FLTrustCustomFedAvg(FedAvg):
    """
    CustomFedAvg style (W&B + save json + best ckpt + lr schedule),
    but training aggregation is replaced by FLTrust.

    FLTrust is implemented using DELTAS:
        local_delta = w_client - w_global
        central_delta = w_root_after - w_global

    Then run the original FLTrust scoring+clipping+weighted sum on deltas,
    and update:
        w_global += server_lr * trusted_delta
    """

    def __init__(
        self,
        *args,
        # FLTrust required server-side hooks:
        model_fn: Callable[[], torch.nn.Module],
        root_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
        server_lr: float = 1.0,
        root_epochs: int = 1,
        root_steps: Optional[int] = None,   # cap batches per epoch
        device: Optional[torch.device] = None,
        # Optional: restrict layers
        layer_filter: Optional[Callable[[str], bool]] = None,

        num_malicious: int,
        warmup_rounds: int,
        attack_mode: str = "lie",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_path: Optional[Path] = None
        self.run_dir: Optional[str] = None
        self.best_acc_so_far = 0.0

        # ---- Attack config ----
        self.num_malicious = int(num_malicious)
        self.warmup_rounds = int(warmup_rounds)
        self.attack_mode = attack_mode

        self.clients_state: Dict[int, bool] = {}
        self.fixed_malicious_ids: Optional[List[int]] = None
        self.all_client_ids: set[int] = set()

        # --- FLTrust config ---
        self.server_lr = float(server_lr)
        self.root_epochs = int(root_epochs)
        self.root_steps = root_steps
        self.layer_filter = layer_filter
        self.eps = 1e-12

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_fn = model_fn
        self.root_loader = root_loader
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        # server replica for root update
        self.root_model = self.model_fn().to(self.device)

    # ------------------------------------------------------------------
    # Original CustomFedAvg utilities
    # ------------------------------------------------------------------

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir


    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        if os.path.exists(f"{self.save_path}/results.json"):
            with open(f"{self.save_path}/results.json", "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        last_train_metrics = dict(result.train_metrics_clientapp.get(current_round, {}))
        last_eval_client_metrics = dict(result.evaluate_metrics_clientapp.get(current_round, {}))
        last_eval_server_metrics = dict(result.evaluate_metrics_serverapp.get(current_round, {}))

        round_results = {
            "round": current_round,
            "train_metrics": last_train_metrics,
            "evaluate_metrics_clientapp": last_eval_client_metrics,
            "evaluate_metrics_serverapp": last_eval_server_metrics,
        }
        results.append(round_results)

        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # same LR schedule as your baseline
        
        config["lr"] = config["lr"] * 0.5
            
        return super().configure_train(server_round, arrays, config, grid)


    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        if server_round <= self.warmup_rounds:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)
            self.clients_state = clients_state
            return ordered, clients_state, malicious_ids

        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            if size > 0:
                self.fixed_malicious_ids = list(np.random.choice(unique_ids, size=size, replace=False))
            else:
                self.fixed_malicious_ids = []

        malicious_ids = self.fixed_malicious_ids
        clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
        ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)

        self.clients_state = clients_state
        return ordered, clients_state, malicious_ids

    def _apply_attack(
        self,
        ordered_train_replies: List[Message],
        clients_state: Dict[int, bool],
        malicious_ids: List[int],
    ) -> List[Message]:
        mode = (self.attack_mode or "none").lower()

        if mode == "none":
            return no_attack_messages(ordered_train_replies, clients_state)

        if mode == "minmax":
            return minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_ids),
            )

        if mode == "gaussian":
            return gaussian_attack_messages(
                ordered_train_replies,
                clients_state,
                magnitude=3,
            )

        if mode == "fang":
            return fang_attack_messages(
                ordered_train_replies,
                clients_state,
            )

        # default: LIE
        return lie_attack_messages(
            ordered_train_replies,
            clients_state,
            omniscent=True,
        )
    # ------------------------------------------------------------------
    # FLTrust core helpers (based on your original fltrust() logic)
    # ------------------------------------------------------------------

    def _iter_float_keys(self, sd: Dict[str, torch.Tensor]) -> Iterable[str]:
        for k, v in sd.items():
            if not v.is_floating_point():
                continue
            if self.layer_filter is not None and (not self.layer_filter(k)):
                continue
            yield k

    def _sd_to_vec(self, sd: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for k in self._iter_float_keys(sd):
            parts.append(sd[k].detach().cpu().flatten())
        if not parts:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    def _delta_sd(self, client_sd: Dict[str, torch.Tensor], global_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d: Dict[str, torch.Tensor] = {}
        for k in self._iter_float_keys(global_sd):
            d[k] = (client_sd[k].detach().cpu() - global_sd[k].detach().cpu())
        return d

    def _compute_root_delta_sd(self, global_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute central_param (root update) = w_root_after - w_global."""
        self.root_model.load_state_dict(global_sd, strict=True)
        self.root_model.train()
        before = {k: v.detach().cpu().clone() for k, v in global_sd.items()}

        opt = self.optimizer_fn(self.root_model)

        for _ in range(self.root_epochs):
            steps = 0
            for batch in self.root_loader:
                # HF/Flower dataset format: dict with "image","label"
                xb = batch["image"]
                yb = batch["label"]

                # ensure float inputs
                if not torch.is_tensor(xb):
                    xb = torch.as_tensor(xb)
                if xb.dtype == torch.uint8:
                    xb = xb.float().div(255.0)
                else:
                    xb = xb.float()
                xb = xb.to(self.device)

                yb = torch.as_tensor(yb).long().to(self.device)

                opt.zero_grad(set_to_none=True)
                out = self.root_model(xb)
                loss = self.loss_fn(out, yb)
                loss.backward()
                opt.step()

                steps += 1
                if self.root_steps is not None and steps >= self.root_steps:
                    break

        after_sd = {k: v.detach().cpu().clone() for k, v in self.root_model.state_dict().items()}

        delta0: Dict[str, torch.Tensor] = {}
        for k in self._iter_float_keys(before):
            delta0[k] = after_sd[k] - before[k]
        return delta0

    def _fltrust_aggregate_arrays(
        self,
        global_arrays: ArrayRecord,
        train_replies: List[Message],
    ) -> Tuple[ArrayRecord, MetricRecord]:
        """
        Implements your original fltrust() logic on DELTAS:
          params -> list of local_delta_sd
          central_param -> root_delta_sd
          global_parameters -> global_sd (updated in-place conceptually)
        """
        global_sd = global_arrays.to_torch_state_dict()

        # central_param = root delta
        central_delta_sd = self._compute_root_delta_sd(global_sd)

        FLTrustTotalScore = 0.0
        score_list: List[float] = []

        central_v = self._sd_to_vec(central_delta_sd)
        central_norm = float(torch.linalg.norm(central_v) + self.eps)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        sum_delta: Optional[Dict[str, torch.Tensor]] = None

        # Build local deltas list (params)
        local_deltas: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for msg in train_replies:
            cid = int(msg.metadata.src_node_id)
            client_sd = msg.content["arrays"].to_torch_state_dict()
            local_delta_sd = self._delta_sd(client_sd, global_sd)
            local_deltas.append((cid, local_delta_sd))

        for cid, local_delta_sd in local_deltas:
            local_v = self._sd_to_vec(local_delta_sd)
            client_cos = float(cos(central_v, local_v))
            client_cos = max(client_cos, 0.0)

            local_norm = float(torch.linalg.norm(local_v) + self.eps)
            client_clipped_value = central_norm / local_norm

            score_list.append(client_cos)
            FLTrustTotalScore += client_cos

            if sum_delta is None:
                sum_delta = {k: (client_cos * client_clipped_value) * v.clone()
                             for k, v in local_delta_sd.items()}
            else:
                for k in sum_delta:
                    sum_delta[k] = sum_delta[k] + (client_cos * client_clipped_value) * local_delta_sd[k]

        # If all scores are 0 -> no update
        if FLTrustTotalScore <= self.eps or sum_delta is None:
            log(INFO, "FLTrustTotalScore == 0 -> no update. scores=%s", score_list)
            mr = MetricRecord({
                "fltrust/total_score": 0.0,
                "fltrust/mean_score": float(np.mean(score_list)) if score_list else 0.0,
                "fltrust/frac_pos": float(np.mean([1.0 if s > 0 else 0.0 for s in score_list])) if score_list else 0.0,
            })
            return global_arrays, mr

        # Update global_parameters[var] += temp * server_lr, where temp = sum_delta/total_score
        new_sd = dict(global_sd)
        for k in global_sd:
            if k.split(".")[-1] == "num_batches_tracked":
                # keep a valid value from first client if exists
                if train_replies:
                    new_sd[k] = train_replies[0].content["arrays"].to_torch_state_dict()[k]
                continue

            if (k not in new_sd) or (not global_sd[k].is_floating_point()):
                continue
            if self.layer_filter is not None and (not self.layer_filter(k)):
                continue

            temp = (sum_delta[k] / FLTrustTotalScore)
            # match dtype/device
            if new_sd[k].dtype != temp.dtype:
                temp = temp.to(new_sd[k].dtype)
            new_sd[k] = new_sd[k] + temp.to(new_sd[k].device) * self.server_lr

        agg_arrays = ArrayRecord.from_torch_state_dict(new_sd)

        # Metrics for W&B and json (mirror your existing logging pattern)
        mean_score = float(np.mean(score_list)) if score_list else 0.0
        frac_pos = float(np.mean([1.0 if s > 0 else 0.0 for s in score_list])) if score_list else 0.0

        mr = MetricRecord({
            "fltrust/total_score": float(FLTrustTotalScore),
            "fltrust/mean_score": mean_score,
            "fltrust/frac_pos": frac_pos,
        })
        log(INFO, "FLTrust scores=%s", score_list)
        return agg_arrays, mr

    # ------------------------------------------------------------------
    # start(): keep your baseline logic, replace aggregate_train block
    # ------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:


        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # ------------------- TRAIN -------------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )

            ordered_train_replies, clients_state, malicious_ids = self._select_malicious_and_order(
                current_round, list(train_replies)
            )
            attacked_replies = self._apply_attack(ordered_train_replies, clients_state, malicious_ids)
            # --------- FLTrust aggregation replaces FedAvg aggregate_train ---------
            agg_arrays, agg_train_metrics = self._fltrust_aggregate_arrays(
                global_arrays=arrays,
                train_replies=attacked_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays

            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # ------------------- EVAL (clients) -------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # ------------------- EVAL (server) -------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    



"""Flower ServerApp Strategy: FLTrust + Attack injection + LBFGS/HVP detection."""




# ---------------------------------------------------------------------
# LBFGS/HVP helpers (NumPy)
# ---------------------------------------------------------------------

def _flatten_arrayrecord(arrays: ArrayRecord) -> np.ndarray:
    """ArrayRecord -> (d, 1) float64 column vector."""
    nds = arrays.to_numpy_ndarrays()
    return np.concatenate([a.reshape(-1, 1) for a in nds], axis=0).astype(np.float64, copy=False)


def _lbfgs_compact(S_list: List[np.ndarray], Y_list: List[np.ndarray], v: np.ndarray) -> np.ndarray:
    """Compact L-BFGS product approximation (NumPy port of your MXNet-style function)."""
    if len(S_list) == 0 or len(Y_list) == 0:
        return np.zeros_like(v)

    S = np.concatenate(S_list, axis=1)  # (d, m)
    Y = np.concatenate(Y_list, axis=1)  # (d, m)

    Sty = S.T @ Y                       # (m, m)
    Sts = S.T @ S                       # (m, m)

    R = np.triu(Sty)
    L = Sty - R

    s_last = S_list[-1]
    y_last = Y_list[-1]

    denom = float(s_last.T @ s_last)
    sigma = 1.0 if denom <= 0 else float(y_last.T @ s_last) / denom

    D = np.diag(np.diag(Sty))

    upper = np.concatenate([sigma * Sts, L], axis=1)
    lower = np.concatenate([L.T, -D], axis=1)
    M = np.concatenate([upper, lower], axis=0)

    # Optional damping if inversion is unstable:
    # M = M + 1e-12 * np.eye(M.shape[0])

    M_inv = np.linalg.inv(M)

    approx = sigma * v
    p = np.concatenate([S.T @ (sigma * v), Y.T @ v], axis=0)  # (2m, 1)
    SY = np.concatenate([sigma * S, Y], axis=1)               # (d, 2m)
    approx = approx - (SY @ (M_inv @ p))
    return approx


@dataclass
class HVPConfig:
    array_key: str = "arrays"         # Message.content key for client arrays
    eta_server: float = 1.0           # g_t ≈ -(w_t - w_{t-1}) / eta_server
    warmup_rounds_hvp: int = 50       # enable hvp scoring only after this round
    max_hist: int = 10                # L-BFGS memory
    detect_topk: Optional[int] = None # if set, predict top-k as malicious (else use num_malicious)


@dataclass
class HVPState:
    S_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    Y_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    last_g: Optional[np.ndarray] = None
    last_client_delta: Dict[int, np.ndarray] = field(default_factory=dict)





class FedAvgWithLBFGSHVP(FedAvg):
    """
    CustomFedAvg-style start() + FLTrust aggregation + attack injection
    + LBFGS/HVP detection + comparison vs true malicious ids.
    """

    def __init__(
        self,
        *args,
        # ---- Attack config:
        num_malicious: int = 0,
        warmup_rounds_attack: int = 0,
        attack_mode: str = "none",

        # ---- HVP config:
        hvp_cfg: Optional[HVPConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save/logging
        self.save_path: Optional[Path] = None
        self.run_dir: Optional[str] = None
        self.best_acc_so_far: float = 0.0

        # ---- Attack config ----
        self.num_malicious = int(num_malicious)
        self.warmup_rounds_attack = int(warmup_rounds_attack)
        self.attack_mode = (attack_mode or "none").lower()

        self.clients_state: Dict[int, bool] = {}
        self.fixed_malicious_ids: Optional[List[int]] = None
        self.all_client_ids: set[int] = set()

        # ---- HVP config/state ----
        self.hvp_cfg = hvp_cfg or HVPConfig()
        self.hvp_state = HVPState(
            S_hist=deque(maxlen=self.hvp_cfg.max_hist),
            Y_hist=deque(maxlen=self.hvp_cfg.max_hist),
        )

    # ------------------------------------------------------------------
    # Basic utilities (save path, json)
    # ------------------------------------------------------------------

    def set_save_path_and_run_dir(self, path: Path, run_dir: str) -> None:
        self.save_path = path
        self.run_dir = run_dir

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        if self.save_path is None:
            return

        out = self.save_path / "results.json"
        if out.exists():
            with open(out, "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        round_results = {
            "round": current_round,
            "train_metrics": dict(result.train_metrics_clientapp.get(current_round, {})),
            "evaluate_metrics_clientapp": dict(result.evaluate_metrics_clientapp.get(current_round, {})),
            "evaluate_metrics_serverapp": dict(result.evaluate_metrics_serverapp.get(current_round, {})),
        }
        results.append(round_results)

        with open(out, "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    # ------------------------------------------------------------------
    # configure_train: example LR schedule (same structure as yours)
    # ------------------------------------------------------------------

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        config["lr"] = config["lr"] 
        return super().configure_train(server_round, arrays, config, grid)

    # ------------------------------------------------------------------
    # Attack: select malicious + apply attack
    # ------------------------------------------------------------------

    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        # During attack warmup: no malicious or fixed list (if already set)
        if server_round <= self.warmup_rounds_attack:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)
            self.clients_state = clients_state
            return ordered, clients_state, malicious_ids

        # Fix malicious IDs once (like your code)
        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            self.fixed_malicious_ids = list(np.random.choice(unique_ids, size=size, replace=False)) if size > 0 else []

        malicious_ids = self.fixed_malicious_ids
        clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
        ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)

        self.clients_state = clients_state
        return ordered, clients_state, malicious_ids

    def _apply_attack(
        self,
        ordered_train_replies: List[Message],
        clients_state: Dict[int, bool],
        malicious_ids: List[int],
    ) -> List[Message]:
        mode = (self.attack_mode or "none").lower()

        if mode == "none":
            return no_attack_messages(ordered_train_replies, clients_state)
        if mode == "minmax":
            return minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_ids),
            )

        if mode == "gaussian":
            return gaussian_attack_messages(
                ordered_train_replies,
                clients_state,
                magnitude=3,
            )

        if mode == "fang":
            return fang_attack_messages(
                ordered_train_replies,
                clients_state,
            )

        # default: LIE
        return lie_attack_messages(
            ordered_train_replies,
            clients_state,
            omniscent=True,
        )


    def _extract_client_arrays(self, msg: Message) -> Optional[ArrayRecord]:
        if msg.has_error() or msg.content is None:
            return None
        if self.hvp_cfg.array_key not in msg.content:
            return None
        return msg.content[self.hvp_cfg.array_key]  # type: ignore[return-value]

    def _update_lbfgs_and_get_hvp(
        self,
        *,
        current_round: int,
        global_prev: ArrayRecord,
        global_curr: Optional[ArrayRecord],
    ) -> Optional[np.ndarray]:
        if global_curr is None:
            return None

        w_prev = _flatten_arrayrecord(global_prev)
        w_curr = _flatten_arrayrecord(global_curr)

        s_t = w_curr - w_prev
        g_t = -s_t / float(self.hvp_cfg.eta_server)

        if self.hvp_state.last_g is not None:
            y_t = g_t - self.hvp_state.last_g
            self.hvp_state.Y_hist.append(y_t)
        self.hvp_state.S_hist.append(s_t)
        self.hvp_state.last_g = g_t

        if current_round <= self.hvp_cfg.warmup_rounds_hvp:
            return None
        if len(self.hvp_state.S_hist) == 0 or len(self.hvp_state.Y_hist) == 0:
            return None

        m = min(len(self.hvp_state.S_hist), len(self.hvp_state.Y_hist))
        S_list = list(self.hvp_state.S_hist)[-m:]
        Y_list = list(self.hvp_state.Y_hist)[-m:]
        return _lbfgs_compact(S_list, Y_list, v=s_t)

    def _hvp_detect(
        self,
        *,
        current_round: int,
        global_prev: ArrayRecord,
        global_curr: Optional[ArrayRecord],
        train_replies: List[Message],
    ) -> Tuple[Dict[int, float], Dict[str, float]]:
        """Return scores (node_id->distance) and summary stats."""
        w_prev = _flatten_arrayrecord(global_prev)
        hvp = self._update_lbfgs_and_get_hvp(
            current_round=current_round, global_prev=global_prev, global_curr=global_curr
        )

        scores: Dict[int, float] = {}
        for msg in train_replies:
            arrays_i = self._extract_client_arrays(msg)
            if arrays_i is None:
                continue

            node_id = int(msg.metadata.src_node_id)
            w_i = _flatten_arrayrecord(arrays_i)
            delta_i = w_i - w_prev  # Δ_i^t

            if hvp is not None and node_id in self.hvp_state.last_client_delta:
                pred_delta_i = self.hvp_state.last_client_delta[node_id] + hvp
                scores[node_id] = float(np.linalg.norm(pred_delta_i - delta_i))
            else:
                # baseline (optional)
                if node_id in self.hvp_state.last_client_delta:
                    scores[node_id] = float(np.linalg.norm(self.hvp_state.last_client_delta[node_id] - delta_i))

            self.hvp_state.last_client_delta[node_id] = delta_i

        if scores:
            vals = np.array(list(scores.values()), dtype=np.float64)
            stats = {
                "hvp/enabled": float(1.0 if hvp is not None else 0.0),
                "hvp/scored_clients": float(len(vals)),
                "hvp/score_mean": float(np.mean(vals)),
                "hvp/score_max": float(np.max(vals)),
                "hvp/score_min": float(np.min(vals)),
            }
        else:
            stats = {
                "hvp/enabled": float(1.0 if hvp is not None else 0.0),
                "hvp/scored_clients": 0.0,
                "hvp/score_mean": 0.0,
                "hvp/score_max": 0.0,
                "hvp/score_min": 0.0,
            }
        return scores, stats

    def _compare_hvp_to_truth(
        self,
        *,
        scores: Dict[int, float],
        malicious_ids: List[int],
        all_ids_in_round: List[int],
    ) -> Dict[str, float]:
        """
        Compare predicted malicious set (from hvp scores) vs ground truth malicious_ids.

        Prediction rule:
          - top-K highest scores are predicted malicious
          - K = hvp_cfg.detect_topk if set else len(malicious_ids) else num_malicious
        """
        truth = set(malicious_ids)
        ids = list(dict.fromkeys(all_ids_in_round))     # unique
        ids = [i for i in ids if i in scores]           # scored only

        if not ids:
            return {
                "hvp/precision": 0.0,
                "hvp/recall": 0.0,
                "hvp/fpr": 0.0,
                "hvp/fnr": 0.0,
                "hvp/tp": 0.0,
                "hvp/fp": 0.0,
                "hvp/tn": 0.0,
                "hvp/fn": 0.0,
            }

        k = self.hvp_cfg.detect_topk
        if k is None:
            k = len(malicious_ids) if len(malicious_ids) > 0 else self.num_malicious
        k = max(0, min(int(k), len(ids)))

        ranked = sorted(ids, key=lambda cid: scores[cid], reverse=True)
        pred_mal = set(ranked[:k])

        tp = len(pred_mal & truth)
        fp = len(pred_mal - truth)
        fn = len(truth - pred_mal)
        tn = len(set(ids) - truth - pred_mal)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "hvp/precision": float(precision),
            "hvp/recall": float(recall),
            "hvp/fpr": float(fpr),
            "hvp/fnr": float(fnr),
            "hvp/tp": float(tp),
            "hvp/fp": float(fp),
            "hvp/tn": float(tn),
            "hvp/fn": float(fn),
            "hvp/pred_k": float(k),
        }

    # ------------------------------------------------------------------
    # start(): FedAvg + attack + hvp detect + compare
    # ------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # ------------------- TRAIN -------------------
            train_replies_iter = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )
            train_replies = list(train_replies_iter)

            ordered, clients_state, malicious_ids = self._select_malicious_and_order(current_round, train_replies)
            attacked_replies = self._apply_attack(ordered, clients_state, malicious_ids)

            # FedAvg aggregation on attacked replies
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, attacked_replies)

            # HVP detection on attacked replies
            scores, hvp_stats = self._hvp_detect(
                current_round=current_round,
                global_prev=arrays,
                global_curr=agg_arrays,
                train_replies=attacked_replies,
            )

            ids_in_round = [m.metadata.src_node_id for m in attacked_replies]
            cmp_stats = self._compare_hvp_to_truth(
                scores=scores,
                malicious_ids=malicious_ids,
                all_ids_in_round=ids_in_round,
            )

            # Optional logging: top suspicious vs truth
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top = ranked[: min(10, len(ranked))]
                log(INFO, "\t└──> True malicious ids: %s", sorted(malicious_ids))
                log(INFO, "\t└──> HVP top suspicious (node_id, score): %s", top)

            # Update arrays
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays

            # Merge train metrics
            merged = {}
            if agg_train_metrics is not None:
                merged.update(dict(agg_train_metrics))
            merged.update(hvp_stats)
            merged.update(cmp_stats)

            if merged:
                mr = MetricRecord(merged)
                log(INFO, "\t└──> Aggregated MetricRecord: %s", mr)
                result.train_metrics_clientapp[current_round] = mr

            # ------------------- EVAL (clients) -------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )
            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # ------------------- EVAL (server) -------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        return result
    
















class FedAvgWithLBFGSHVP(FedAvg):
    """
    CustomFedAvg-style start() + FLTrust aggregation + attack injection
    + LBFGS/HVP detection + comparison vs true malicious ids.
    """

    def __init__(
        self,
        *args,
        # ---- Attack config:
        num_malicious: int = 0,
        warmup_rounds_attack: int = 0,
        attack_mode: str = "none",

        # ---- HVP config:
        hvp_cfg: Optional[HVPConfig] = None,

        suspicion_window: int = 10,        # N
        detect_from_round: int = 5,       # bài báo dùng mặc định 50
        gap_B: int = 5,                   # số sample cho Gap statistics
        gap_K: int = 2,                   # số cluster tối đa cho Gap statistics
        remove_detected_clients: bool = True,
        random_state: int = 0,

        # ---- Filtering config:
        keep_fraction: float = 0.6,   # fraction of least-suspicious clients to aggregate
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save/logging
        self.save_path: Optional[Path] = None
        self.run_dir: Optional[str] = None
        self.best_acc_so_far: float = 0.0

        # ---- Attack config ----
        self.num_malicious = int(num_malicious)
        self.warmup_rounds_attack = int(warmup_rounds_attack)
        self.attack_mode = (attack_mode or "none").lower()

        self.clients_state: Dict[int, bool] = {}
        self.fixed_malicious_ids: Optional[List[int]] = None
        self.all_client_ids: set[int] = set()

        # ---- HVP config/state ----
        self.hvp_cfg = hvp_cfg or HVPConfig()
        self.hvp_state = HVPState(
            S_hist=deque(maxlen=self.hvp_cfg.max_hist),
            Y_hist=deque(maxlen=self.hvp_cfg.max_hist),
        )

        self.suspicion_window = int(suspicion_window)
        self.detect_from_round = int(detect_from_round)
        self.gap_B = int(gap_B)
        self.gap_K = int(gap_K)
        self.remove_detected_clients = bool(remove_detected_clients)
        self._rng = np.random.default_rng(random_state)


        # ---- Filtering config ----
        self.keep_fraction = float(keep_fraction)


    def _get_hist(self, node_id: int) -> deque:
        hist = self.hvp_state.norm_dist_hist.get(node_id)
        if hist is None:
            hist = deque(maxlen=self.suspicion_window)
            self.hvp_state.norm_dist_hist[node_id] = hist
        return hist


    def _update_suspicious_scores(
        self,
        normalized_scores: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Mỗi round append d_hat_t[i] vào history của client i.
        suspicious_score[i] = mean(history_i)
        """
        for node_id, val in normalized_scores.items():
            self._get_hist(node_id).append(float(val))

        updated: Dict[int, float] = {}
        for node_id, hist in self.hvp_state.norm_dist_hist.items():
            if len(hist) == 0:
                continue
            updated[node_id] = float(np.mean(hist))

        self.hvp_state.suspicious_scores = updated
        return updated
    
    def _filter_replies_by_detected_set(
        self,
        train_replies: List[Message],
    ) -> List[Message]:
        if not self.remove_detected_clients or not self.hvp_state.detected_malicious:
            return train_replies

        kept = [
            msg for msg in train_replies
            if int(msg.metadata.src_node_id) not in self.hvp_state.detected_malicious
        ]
        dropped = len(train_replies) - len(kept)
        log(
            INFO,
            "\t└──> Removed %d previously detected malicious clients; kept %d replies",
            dropped,
            len(kept),
        )
        return kept

    # ------------------------------------------------------------------
    # Basic utilities (save path, json)
    # ------------------------------------------------------------------

    def set_save_path_and_run_dir(self, path: Path, run_dir: str) -> None:
        self.save_path = path
        self.run_dir = run_dir

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        if self.save_path is None:
            return

        out = self.save_path / "results.json"
        if out.exists():
            with open(out, "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        round_results = {
            "round": current_round,
            "train_metrics": dict(result.train_metrics_clientapp.get(current_round, {})),
            "evaluate_metrics_clientapp": dict(result.evaluate_metrics_clientapp.get(current_round, {})),
            "evaluate_metrics_serverapp": dict(result.evaluate_metrics_serverapp.get(current_round, {})),
        }
        results.append(round_results)

        with open(out, "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    # ------------------------------------------------------------------
    # configure_train: example LR schedule (same structure as yours)
    # ------------------------------------------------------------------

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        config["lr"] = config["lr"] 
        return super().configure_train(server_round, arrays, config, grid)

    # ------------------------------------------------------------------
    # Attack: select malicious + apply attack
    # ------------------------------------------------------------------

    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        # During attack warmup: no malicious or fixed list (if already set)
        if server_round <= self.warmup_rounds_attack:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)
            self.clients_state = clients_state
            return ordered, clients_state, malicious_ids

        # Fix malicious IDs once (like your code)
        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            self.fixed_malicious_ids = list(np.random.choice(unique_ids, size=size, replace=False)) if size > 0 else []

        malicious_ids = self.fixed_malicious_ids
        clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
        ordered = sorted(train_replies, key=lambda m: m.metadata.src_node_id)

        self.clients_state = clients_state
        return ordered, clients_state, malicious_ids

    def _apply_attack(
        self,
        ordered_train_replies: List[Message],
        clients_state: Dict[int, bool],
        malicious_ids: List[int],
        global_arrays: Optional[ArrayRecord] = None,
    ) -> List[Message]:
        mode = (self.attack_mode or "none").lower()

        if mode == "none":
            return no_attack_messages(ordered_train_replies, clients_state)
        if mode == "minmax":
            return minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_ids),
            )

        if mode == "gaussian":
            return gaussian_attack_messages(
                ordered_train_replies,
                clients_state,
                magnitude=10,
            )

        if mode == "fang":
            w_re = {k: v.cpu().numpy() for k, v in global_arrays.to_torch_state_dict().items()}
            return fang_attack_messages(
                ordered_train_replies,
                clients_state,
                w_re=w_re,
            )

        # default: LIE
        return lie_attack_messages(
            ordered_train_replies,
            clients_state,
            omniscent=True,
        )


    def _extract_client_arrays(self, msg: Message) -> Optional[ArrayRecord]:
        if msg.has_error() or msg.content is None:
            return None
        if self.hvp_cfg.array_key not in msg.content:
            return None
        return msg.content[self.hvp_cfg.array_key]  # type: ignore[return-value]

    def _update_lbfgs_and_get_hvp(
        self,
        *,
        current_round: int,
        global_prev: ArrayRecord,
        global_curr: Optional[ArrayRecord],
    ) -> Optional[np.ndarray]:
        if global_curr is None:
            return None

        w_prev = _flatten_arrayrecord(global_prev)
        w_curr = _flatten_arrayrecord(global_curr)

        s_t = w_curr - w_prev
        g_t = -s_t / float(self.hvp_cfg.eta_server)

        if self.hvp_state.last_g is not None:
            y_t = g_t - self.hvp_state.last_g
            self.hvp_state.Y_hist.append(y_t)
        self.hvp_state.S_hist.append(s_t)
        self.hvp_state.last_g = g_t

        if current_round <= self.hvp_cfg.warmup_rounds_hvp:
            return None
        if len(self.hvp_state.S_hist) == 0 or len(self.hvp_state.Y_hist) == 0:
            return None

        m = min(len(self.hvp_state.S_hist), len(self.hvp_state.Y_hist))
        S_list = list(self.hvp_state.S_hist)[-m:]
        Y_list = list(self.hvp_state.Y_hist)[-m:]
        hvp = _lbfgs_compact(S_list, Y_list, v=s_t)
        self.hvp_state.last_hvp = hvp   # cache for next round's pre-aggregation filtering
        return hvp

    def _hvp_detect(
        self,
        *,
        current_round: int,
        global_prev: ArrayRecord,
        global_curr: Optional[ArrayRecord],
        train_replies: List[Message],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]:
        """
        Return:
        raw_scores[node_id]        = ||pred_delta_i - delta_i||_2
        normalized_scores[node_id] = raw_scores / L1(raw_scores in this round)
        stats
        """
        w_prev = _flatten_arrayrecord(global_prev)
        hvp = self._update_lbfgs_and_get_hvp(
            current_round=current_round,
            global_prev=global_prev,
            global_curr=global_curr,
        )

        raw_scores: Dict[int, float] = {}
        for msg in train_replies:
            arrays_i = self._extract_client_arrays(msg)
            if arrays_i is None:
                continue

            node_id = int(msg.metadata.src_node_id)
            w_i = _flatten_arrayrecord(arrays_i)
            delta_i = w_i - w_prev

            if hvp is not None and node_id in self.hvp_state.last_client_delta:
                pred_delta_i = self.hvp_state.last_client_delta[node_id] + hvp
                raw_scores[node_id] = float(np.linalg.norm(pred_delta_i - delta_i))
            elif node_id in self.hvp_state.last_client_delta:
                # fallback khi HVP chưa sẵn sàng
                raw_scores[node_id] = float(
                    np.linalg.norm(self.hvp_state.last_client_delta[node_id] - delta_i)
                )

            self.hvp_state.last_client_delta[node_id] = delta_i

        normalized_scores: Dict[int, float] = {}
        l1 = float(sum(raw_scores.values()))
        if l1 > 0.0:
            normalized_scores = {cid: val / l1 for cid, val in raw_scores.items()}
        else:
            normalized_scores = {cid: 0.0 for cid in raw_scores}

        vals = np.array(list(raw_scores.values()), dtype=np.float64) if raw_scores else np.array([])
        stats = {
            "hvp/enabled": float(1.0 if hvp is not None else 0.0),
            "hvp/scored_clients": float(len(raw_scores)),
            "hvp/raw_score_mean": float(vals.mean()) if len(vals) else 0.0,
            "hvp/raw_score_max": float(vals.max()) if len(vals) else 0.0,
            "hvp/raw_score_min": float(vals.min()) if len(vals) else 0.0,
        }
        return raw_scores, normalized_scores, stats
    

    def _normalize_unit_interval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)


    def _kmeans_1d(
        self,
        x: np.ndarray,
        k: int,
        max_iter: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n = len(x)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64), 0.0

        uniq = np.unique(x)
        k = max(1, min(int(k), len(uniq)))

        if k == 1:
            center = np.array([float(np.mean(x))], dtype=np.float64)
            labels = np.zeros(n, dtype=int)
            W = float(np.sum((x - center[0]) ** 2))
            return labels, center, W

        # init bằng quantiles
        qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
        centers = np.quantile(x, qs)

        for _ in range(max_iter):
            d = np.abs(x[:, None] - centers[None, :])
            labels = np.argmin(d, axis=1)

            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = float(np.mean(x[mask]))

            if np.allclose(new_centers, centers, atol=1e-8):
                centers = new_centers
                break
            centers = new_centers

        d = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(d, axis=1)
        W = 0.0
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                W += float(np.sum((x[mask] - centers[j]) ** 2))

        return labels, centers, W


    def _gap_statistic_1d(
        self,
        suspicious_scores: Dict[int, float],
    ) -> Tuple[int, Dict[str, float]]:
        ids = sorted(suspicious_scores)
        x = np.array([suspicious_scores[i] for i in ids], dtype=np.float64)

        if len(x) < 2 or np.allclose(x, x[0]):
            return 1, {
                "detect/gap_k": 1.0,
                "detect/gap_value_k1": 0.0,
            }

        x = self._normalize_unit_interval(x)

        K = min(self.gap_K, len(np.unique(x)))
        if K <= 1:
            return 1, {
                "detect/gap_k": 1.0,
                "detect/gap_value_k1": 0.0,
            }

        eps = 1e-12
        gaps = np.zeros(K + 1, dtype=np.float64)
        s_vals = np.zeros(K + 1, dtype=np.float64)

        for k in range(1, K + 1):
            _, _, Wk = self._kmeans_1d(x, k)
            ref_logs = []

            for _ in range(self.gap_B):
                ref = self._rng.uniform(0.0, 1.0, size=len(x))
                _, _, Wkb = self._kmeans_1d(ref, k)
                ref_logs.append(np.log(Wkb + eps))

            ref_logs = np.asarray(ref_logs, dtype=np.float64)
            gaps[k] = float(np.mean(ref_logs) - np.log(Wk + eps))
            sd_k = float(np.sqrt(np.mean((ref_logs - np.mean(ref_logs)) ** 2)))
            s_vals[k] = float(np.sqrt((1.0 + self.gap_B) / self.gap_B) * sd_k)

        k_hat = K
        for k in range(1, K):
            if gaps[k] - gaps[k + 1] + s_vals[k + 1] >= 0.0:
                k_hat = k
                break

        stats = {
            "detect/gap_k": float(k_hat),
            "detect/gap_value_k1": float(gaps[1]),
            "detect/gap_value_khat": float(gaps[k_hat]),
        }
        return int(k_hat), stats
    

    def _detect_malicious_from_suspicion(
        self,
        current_round: int,
        active_node_ids: List[int],
    ) -> Tuple[Set[int], Dict[str, float]]:
        """
        active_node_ids: các client xuất hiện trong round hiện tại.
        Chỉ detect khi:
        - đủ round warmup
        - client có đủ suspicion_window history
        """
        if current_round < self.detect_from_round:
            return set(), {
                "detect/enabled": 0.0,
                "detect/eligible_clients": 0.0,
                "detect/detected_count": 0.0,
            }

        eligible_scores: Dict[int, float] = {}
        for node_id in active_node_ids:
            hist = self.hvp_state.norm_dist_hist.get(node_id)
            if hist is None or len(hist) < self.suspicion_window:
                continue
            eligible_scores[node_id] = float(np.mean(hist))

        if len(eligible_scores) < 2:
            return set(), {
                "detect/enabled": 1.0,
                "detect/eligible_clients": float(len(eligible_scores)),
                "detect/detected_count": 0.0,
            }

        k_hat, gap_stats = self._gap_statistic_1d(eligible_scores)

        detected: Set[int] = set()
        if k_hat > 1:
            ids = sorted(eligible_scores)
            x = np.array([eligible_scores[i] for i in ids], dtype=np.float64)
            x_norm = self._normalize_unit_interval(x)

            labels, centers, _ = self._kmeans_1d(x_norm, k=2)
            cluster_mean = {}
            for c in [0, 1]:
                mask = labels == c
                if np.any(mask):
                    cluster_mean[c] = float(np.mean(x[mask]))
                else:
                    cluster_mean[c] = -np.inf

            malicious_cluster = max(cluster_mean, key=cluster_mean.get)
            detected = {ids[i] for i, lab in enumerate(labels) if lab == malicious_cluster}

        stats = {
            "detect/enabled": 1.0,
            "detect/eligible_clients": float(len(eligible_scores)),
            "detect/detected_count": float(len(detected)),
            "detect/suspicion_mean": float(np.mean(list(eligible_scores.values()))) if eligible_scores else 0.0,
            "detect/suspicion_max": float(np.max(list(eligible_scores.values()))) if eligible_scores else 0.0,
            **gap_stats,
        }
        return detected, stats

    

    # ------------------------------------------------------------------
    # Pre-aggregation filtering helpers
    # ------------------------------------------------------------------

    def _score_clients_pre_agg(
        self,
        global_prev: ArrayRecord,
        train_replies: List[Message],
    ) -> Dict[int, float]:
        """Score clients using the cached HVP from last round (no aggregation needed).

        Clients with no history are not scored and will be kept by default.
        Returns node_id -> anomaly score (higher = more suspicious).
        """
        w_prev = _flatten_arrayrecord(global_prev)
        hvp = self.hvp_state.last_hvp   # may be None during warmup

        scores: Dict[int, float] = {}
        for msg in train_replies:
            arrays_i = self._extract_client_arrays(msg)
            if arrays_i is None:
                continue
            node_id = int(msg.metadata.src_node_id)
            w_i = _flatten_arrayrecord(arrays_i)
            delta_i = w_i - w_prev

            if node_id not in self.hvp_state.last_client_delta:
                continue  # no history yet — skip scoring, client is kept

            last_d = self.hvp_state.last_client_delta[node_id]
            pred = last_d + hvp if hvp is not None else last_d
            scores[node_id] = float(np.linalg.norm(pred - delta_i))

        return scores

    def _filter_replies_by_score(
        self,
        train_replies: List[Message],
        scores: Dict[int, float],
    ) -> List[Message]:
        """Keep the `keep_fraction` least-suspicious clients.

        Clients that couldn't be scored (no history) are treated as benign
        and are always included in the kept set.
        """
        scored, unscored = [], []
        for msg in train_replies:
            node_id = int(msg.metadata.src_node_id)
            if node_id in scores:
                scored.append((msg, scores[node_id]))
            else:
                unscored.append(msg)  # no history -> keep unconditionally

        # Sort scored clients ascending by suspicion score
        scored.sort(key=lambda x: x[1])
        keep_n = max(1, int(len(scored) * self.keep_fraction))
        kept_scored = [msg for msg, _ in scored[:keep_n]]

        kept = unscored + kept_scored
        dropped_n = len(scored) - keep_n
        log(
            INFO,
            "\t└──> Filtering: kept %d/%d scored clients (%.0f%%) + %d unscored | dropped %d most suspicious",
            keep_n, len(scored), self.keep_fraction * 100, len(unscored), dropped_n,
        )
        return kept

    # ------------------------------------------------------------------
    # start(): FedAvg + attack + hvp detect + compare
    # ------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # ------------------- TRAIN -------------------
            train_replies_iter = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )
            train_replies = list(train_replies_iter)

            ordered, clients_state, malicious_ids = self._select_malicious_and_order(current_round, train_replies)
            attacked_replies = self._apply_attack(ordered, clients_state, malicious_ids, arrays)

            # Score clients using cached HVP from last round, then filter to keep_fraction least suspicious
            pre_scores = self._score_clients_pre_agg(arrays, attacked_replies)
            filtered_replies = self._filter_replies_by_score(attacked_replies, pre_scores)

            # FedAvg aggregation on filtered (low-suspicion) clients only
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, filtered_replies)

            # HVP detection on ALL attacked replies (updates L-BFGS history + computes detection metrics)
            scores, hvp_stats = self._hvp_detect(
                current_round=current_round,
                global_prev=arrays,
                global_curr=agg_arrays,
                train_replies=attacked_replies,
            )

            ids_in_round = [m.metadata.src_node_id for m in attacked_replies]
            cmp_stats = self._compare_hvp_to_truth(
                scores=scores,
                malicious_ids=malicious_ids,
                all_ids_in_round=ids_in_round,
            )

            # Optional logging: top suspicious vs truth
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top = ranked[: min(10, len(ranked))]
                log(INFO, "\t└──> True malicious ids: %s", sorted(malicious_ids))
                log(INFO, "\t└──> HVP top suspicious (node_id, score): %s", top)

            # Update arrays
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays

            # Merge train metrics
            merged = {}
            if agg_train_metrics is not None:
                merged.update(dict(agg_train_metrics))
            merged.update(hvp_stats)
            merged.update(cmp_stats)

            if merged:
                mr = MetricRecord(merged)
                log(INFO, "\t└──> Aggregated MetricRecord: %s", mr)
                result.train_metrics_clientapp[current_round] = mr

            # ------------------- EVAL (clients) -------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )
            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # ------------------- EVAL (server) -------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        return result