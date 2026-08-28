"""pytorch-example: CustomKrum strategy — same logic as CustomFedAvg but using Krum aggregation."""

import csv
import io
import json
import os
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Krum, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info
from rl_flwr.attack import (
    fang_attack_messages,
    gaussian_attack_messages,
    lie_attack_messages,
    minmax_attack_messages,
    no_attack_messages,
)


class CustomKrum(Krum):
    """Behaves like Krum but adds the same extra functionality as CustomFedAvg:

    1. Saves per-round results to a JSON file on disk.
    2. Saves a PyTorch checkpoint whenever a new best accuracy is found.
    3. Applies a simple learning-rate schedule (halve LR at round 5).
    """

    def set_save_path_and_run_dir(self, path: Path, run_dir: str) -> None:
        """Set the path where results and model checkpoints will be saved."""
        self.save_path = path
        self.run_dir = run_dir

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Save a model checkpoint if the current accuracy is a new best."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            file_name = f"model_state_acc_{accuracy}_round_{current_round}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            logger.log(INFO, "💾 New best model saved to disk: %s", file_name)

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        """Append the current round's metrics to a JSON file on disk."""
        json_path = f"{self.save_path}/results.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fp:
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

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training, halving LR at round 5."""
        if server_round == 5:
            config["lr"] = config["lr"] * 0.5
            logger.log(INFO, "⚙️ Adjusted learning rate to %f", config["lr"])
        return super().configure_train(server_round, arrays, config, grid)

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute federated learning with Krum aggregation, saving results and
        checkpoints to disk each round."""

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

        # Evaluate initial global parameters
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
            # TRAINING (CLIENTAPP-SIDE)
            # -----------------------------------------------------------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate with Krum
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # EVALUATION (CLIENTAPP-SIDE)
            # -----------------------------------------------------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # EVALUATION (SERVERAPP-SIDE)
            # -----------------------------------------------------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_acc(current_round, res["accuracy"], arrays)

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


class CustomKrum_attack(Krum):
    """Behaves like Krum but injects configurable Byzantine attacks, mirroring
    the logic of CustomFedAvg_attack from strategy.py."""

    def __init__(
        self,
        *args,
        num_malicious: int,
        warmup_rounds: int,
        attack_mode: str = "no_attack",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds
        self.attack_mode = attack_mode

        self.clients_state: Dict[int, bool] = {}
        self.fixed_malicious_ids: List[int] | None = None
        self.all_client_ids: set[int] = set()

        self.round_log_initialized: bool = False
        self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.best_acc_so_far = 0.0
        self.best_arrays: Optional[ArrayRecord] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List[Message],
        save_path: Path,
    ) -> Tuple[List[Message], Dict[int, bool], List[int]]:
        """Choose which clients are malicious (fixed after first selection) and
        return replies sorted by node_id."""

        node_ids = [msg.metadata.src_node_id for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        if server_round <= self.warmup_rounds:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered_replies = sorted(train_replies, key=lambda m: m.metadata.src_node_id)
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
        ordered_replies = sorted(train_replies, key=lambda m: m.metadata.src_node_id)

        ordered_ids = [m.metadata.src_node_id for m in ordered_replies]
        log(INFO, "[Round %s] ordered node_ids = %s", server_round, ordered_ids)
        log(INFO, "[Round %s] clients_state = %s", server_round, clients_state)
        log(INFO, "[Round %s] malicious_ids (fixed) = %s", server_round, malicious_ids)

        self.clients_state = clients_state
        return ordered_replies, clients_state, malicious_ids

    def set_save_path_and_run_dir(self, path: Path, run_dir: str) -> None:
        """Set the path where results and model checkpoints will be saved."""
        self.save_path = path
        self.run_dir = run_dir

    def save_metrics_as_json(self, current_round: int, result: Result) -> None:
        """Append the current round's metrics to a JSON file on disk."""
        json_path = f"{self.save_path}/results.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fp:
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

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training (LR unchanged, matching CustomFedAvg_attack)."""
        config["lr"] = config["lr"]
        return super().configure_train(server_round, arrays, config, grid)

    def _append_round_states_csv(
        self,
        server_round: int,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        """Append one row per client to client_states_rounds.csv."""
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

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
        """Execute federated learning with Krum aggregation and Byzantine attack injection."""

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

            # -----------------------------------------------------------------
            # TRAINING (CLIENTAPP-SIDE)
            # -----------------------------------------------------------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            ordered_train_replies, clients_state, malicious_cids = (
                self._select_malicious_and_order(
                    current_round, train_replies, save_path=self.save_path
                )
            )

            if self.attack_mode == "no_attack":
                attacked_replies = no_attack_messages(ordered_train_replies, clients_state)
            elif self.attack_mode == "gaussian":
                attacked_replies = gaussian_attack_messages(
                    ordered_train_replies, clients_state, magnitude=3,
                )
            elif self.attack_mode == "lie":
                attacked_replies = lie_attack_messages(
                    ordered_train_replies, clients_state, omniscent=True,
                )
            elif self.attack_mode == "min_max":
                attacked_replies = minmax_attack_messages(
                    ordered_train_replies, clients_state, omniscent=True,
                    threshold=1e-5, lambda_init=5.0, malicious_num=len(malicious_cids),
                )
            else:
                raise ValueError(f"Unknown attack_mode: {self.attack_mode}")

            # Aggregate with Krum
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                attacked_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # EVALUATION (CLIENTAPP-SIDE)
            # -----------------------------------------------------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # EVALUATION (SERVERAPP-SIDE)
            # -----------------------------------------------------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    metrics = res
                    acc_val = float(metrics.get("accuracy", 0.0))
                    loss_val = float(metrics.get("loss", 0.0))

            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
