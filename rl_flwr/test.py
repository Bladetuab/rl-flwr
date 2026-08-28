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
import os

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info
from rl_flwr.attack import gaussian_attack_messages, lie_attack_messages, minmax_attack_messages, fang_attack_messages, no_attack_messages


PROJECT_NAME = "FLOWER-advanced-pytorch"

class CustomFedAvg_attack(FedAvg):

    def __init__(
        self,
        *args,
        num_malicious: int ,
        warmup_rounds: int ,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds

        # để lưu state cross-round nếu cần
        self.clients_state = {}
        self.fixed_malicious_ids: list[int] | None = None
        self.all_client_ids: set[int] = set()

        self.round_log_initialized: bool = False
        self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.best_acc_so_far = 0.0
        self.best_arrays: Optional[ArrayRecord] = None


    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
        save_path: Path,
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        """Chọn malicious clients (cố định sau khi đã chọn) và sắp xếp replies.

        Returns
        -------
        ordered_replies : list[Message]
            List Message được sort theo node_id.
        clients_state : dict[node_id -> bool]
            True nếu node_id là malicious (ground truth).
        malicious_ids : list[int]
            Danh sách node_id malicious (global, không đổi sau khi set).
        """

        # 1) Lấy node_ids của các client tham gia round này
        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))

        # Cập nhật global set tất cả client id
        self.all_client_ids.update(unique_ids)

        # 2) Nếu vẫn trong warmup_rounds -> chưa attack, tất cả benign
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

        # 3) Hết warmup: nếu CHƯA chọn malicious cố định -> chọn MỘT LẦN
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

        # 4) Xây clients_state dựa trên fixed_malicious_ids (cố định từ giờ)
        malicious_ids = self.fixed_malicious_ids
        clients_state: Dict[int, bool] = {
            nid: (nid in malicious_ids) for nid in unique_ids
        }

        # 5) Sắp xếp replies theo node_id
        ordered_replies = sorted(
            train_replies,
            key=lambda m: m.metadata.src_node_id,
        )

        # 6) Log để kiểm tra
        ordered_ids = [m.metadata.src_node_id for m in ordered_replies]
        log(INFO, "[Round %s] ordered node_ids = %s", server_round, ordered_ids)
        log(INFO, "[Round %s] clients_state = %s", server_round, clients_state)
        log(INFO, "[Round %s] malicious_ids (fixed) = %s", server_round, malicious_ids)

        self.clients_state = clients_state
        return ordered_replies, clients_state, malicious_ids




    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        """Set the path where results and model checkpoints will be saved."""
        self.save_path = path
        self.run_dir = run_dir


    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        """Save the current results to a JSON file."""

        # Load current JSON if file exists
        if os.path.exists(f"{self.save_path}/results.json"):
            with open(f"{self.save_path}/results.json", "r", encoding="utf-8") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        # Extract metrics from current round
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
        # Save to JSON
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp)


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        config["lr"] = config["lr"] 
        return super().configure_train(server_round, arrays, config, grid)
        
    def _append_round_states_csv(
        self,
        server_round: int,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        """Ghi thêm 1 dòng cho mỗi client tham gia round này:

        round, client_id, is_malicious (ground truth theo clients_state), predict_state (RL).
        """
        csv_path = f"{self.save_path}/client_states_rounds.csv"
        mode = "a" if self.round_log_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Ghi header 1 lần duy nhất
            if not self.round_log_initialized:
                writer.writerow(["round", "client_id", "is_malicious", "predict_state"])
                self.round_log_initialized = True

            for cid in sorted(round_client_ids):
                # Ground truth: clients_state của round này
                is_mal = 1 if clients_state.get(cid, False) else 0
                # Pred: RL detector
                is_pred = 1 if detected_state.get(cid, False) else 0
                writer.writerow([server_round, cid, is_mal, is_pred])






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
        """Execute the federated learning strategy logging results to W&B and saving
        them to disk."""

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # --- NEW: chọn malicious và tạo ordered_replies ---
            ordered_train_replies, clients_state, malicious_cids = self._select_malicious_and_order(current_round, train_replies,save_path=self.save_path)



            # --- No attack ---
            # attacked_replies = no_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            # )

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

            # --- Min-Max attack ---
            attacked_replies = minmax_attack_messages(
                ordered_train_replies,
                clients_state,
                omniscent=True,
                threshold=1e-5,
                lambda_init=5.0,
                malicious_num=len(malicious_cids),
            )

        
            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                attacked_replies,
            )



            

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

                    # LẤY acc/loss và cập nhật cho RL
                    metrics = res  # MetricRecord là dict-like
                    acc_val = float(metrics.get("accuracy", 0.0))
                    loss_val = float(metrics.get("loss", 0.0))


            # Save metrics to disk as JSON
            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")


        return result