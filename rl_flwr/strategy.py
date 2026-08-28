"""pytorch-example: A Flower / PyTorch app."""

import io
import json
import os
import re
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional, Dict, List, Tuple, Set, Deque

import torch
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader




from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, Krum, MultiKrum
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
    """A class that behaves like FedAvg but has extra functionality added into the
    `start` method. It also overrides `configure_train` to implement a simple learning
    rate schedule.

    This strategy: (1) saves results to the filesystem each rounds, (2)
    saves a checkpoint of the global model when a new best is found,
    (3) logs results to W&B.
    """

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        """Set the path where results and model checkpoints will be saved."""
        self.save_path = path
        self.run_dir = run_dir

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Track the best accuracy seen so far.

        This used to also write `model_state_acc_<acc>_round_<r>.pth` on every new best,
        which produced one checkpoint file per improvement -- dozens of multi-megabyte
        files per run, none of which any analysis or plotting script reads. Checkpointing
        is removed; the accuracy itself is still tracked, and every round's accuracy is
        already persisted by `save_metrics_as_json`, so nothing downstream is lost.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)

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

    def write_run_summary_csv(self) -> None:
        """Write a one-row `run_summary.csv` in the sweep-summary schema.

        Same columns, same order and same semantics as `outputs/sweep_summary_*.csv`, so a
        single run can be concatenated with a sweep or read by the same analysis code
        without a special case.

        Provenance comes from the run-DIRECTORY name, which in this project is the record
        of what a run actually was: `<time>_<dataset>_<defense>_<attack>_tau<q>`.
        run_config.json is deliberately not consulted -- inside a sweep it reports one
        combination for every run and would mislabel the row.
        """
        if getattr(self, "save_path", None) is None:
            return

        fields = [
            "dataset", "defense", "attack_mode", "decorrelate_cos", "assumed_malicious",
            "tau_s_quantile", "final_accuracy", "asr", "fdr", "recall",
            "total_convlstm_forwards", "mean_ratio_scored", "total_convlstm_time_ms",
            "save_path", "error",
        ]
        nan = float("nan")
        row = {k: "" for k in fields}
        row.update({
            "dataset": "", "defense": "", "attack_mode": "",
            "decorrelate_cos": getattr(self, "decorrelate_cos", ""),
            "assumed_malicious": int(getattr(self, "num_malicious", 0) or 0),
            "tau_s_quantile": nan, "final_accuracy": nan,
            "asr": nan, "fdr": nan, "recall": nan,
            "total_convlstm_forwards": int(getattr(self, "total_convlstm_forwards", 0) or 0),
            "mean_ratio_scored": nan, "total_convlstm_time_ms": nan,
            "save_path": str(self.save_path), "error": "",
        })

        m = re.match(
            r"^\d\d-\d\d-\d\d_(?P<dataset>mnist|fashion_mnist|cifar10)_"
            r"(?P<defense>.+?)_(?P<attack>gaussian|lie|min_max|no_attack|fang)"
            r"_tau(?P<tau>[0-9.]+)$",
            Path(self.save_path).name,
        )
        if m:
            row["dataset"] = m.group("dataset")
            row["defense"] = m.group("defense")
            row["attack_mode"] = m.group("attack")
            row["tau_s_quantile"] = float(m.group("tau"))

        # Final accuracy: the last round's server-side global accuracy, i.e. exactly the
        # quantity the sweep's own collector reports.
        results_file = Path(self.save_path) / "results.json"
        if results_file.exists():
            try:
                data = json.load(open(results_file))
                if isinstance(data, list) and data:
                    acc = data[-1].get("evaluate_metrics_serverapp", {}).get("accuracy")
                    if acc is not None:
                        row["final_accuracy"] = round(float(acc), 4)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Detection metrics, if this strategy tracked any. Plain FedAvg excludes nobody,
        # so a strategy without a confusion matrix correctly leaves these NaN rather than
        # reporting a spurious 0.
        cm = getattr(self, "confusion_matrix", None) or {}
        TP, FP, FN, TN = (int(cm.get(k, 0)) for k in ("TP", "FP", "FN", "TN"))
        total = TP + FP + FN + TN
        if total:
            row["asr"] = round(FN / total, 4)
            row["fdr"] = round(FP / total, 4)
            row["recall"] = round(TP / (TP + FN), 4) if (TP + FN) else nan

        out = Path(self.save_path) / "run_summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow(row)
        logger.log(INFO, "📄 Run summary written to %s", out)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Perform basic learning rate scheduling
        if server_round == 5:  # half LR at round 5
            config["lr"] = config["lr"] * 0.5
            logger.log(INFO, "⚙️ Adjusted learning rate to %f", config["lr"])
        # Continue with standard FedAvg configure_train
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

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
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
                    # Maybe save to disk if new best is found
                    self._update_best_acc(current_round, res["accuracy"], arrays)

            # Save metrics to disk as JSON
            self.save_metrics_as_json(current_round=current_round, result=result)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        self.write_run_summary_csv()

        return result



class _DetectionLogMixin:
    """Confusion-matrix + per-round CSV logging for the published baselines.

    The USTAD-FL family (CustomFedAvg_adaptive and its descendants) has always
    recorded which clients it excluded, which is what `_collect_metrics` turns
    into asr/fdr/recall. Multi-Krum, FLTrust and FLDetector did not, so those
    columns came out `nan` and no detection comparison was possible.

    This mixin supplies the same two sinks, byte-compatible with the adaptive
    family's, so `client_states_rounds.csv` from any defense can be read by the
    same analysis code:

        confusion_matrix           -> pooled asr/fdr/recall in the sweep CSV
        client_states_rounds.csv   -> per-round, which is what lets a caller
                                      restrict to the attacked rounds t >= 6

    Both matter: the pooled figures mix in the warm-up rounds, where no attacker
    exists and every exclusion is a false positive by construction, so they are
    not comparable across defenses with different warm-up behaviour.

    Each defense supplies its own notion of "detected"; see the call sites.
    """

    def _init_detection_log(self) -> None:
        if not hasattr(self, "confusion_matrix"):
            self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        if not hasattr(self, "round_log_initialized"):
            self.round_log_initialized = False

    def _update_confusion_matrix(
        self,
        round_client_ids: List[int],
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        for cid in round_client_ids:
            gt = bool(clients_state.get(cid, False))
            pred = bool(detected_state.get(cid, False))
            if gt and pred:
                self.confusion_matrix["TP"] += 1
            elif (not gt) and (not pred):
                self.confusion_matrix["TN"] += 1
            elif (not gt) and pred:
                self.confusion_matrix["FP"] += 1
            else:
                self.confusion_matrix["FN"] += 1

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

    def _record_detection(
        self,
        server_round: int,
        clients_state: Dict[int, bool],
        detected_state: Dict[int, bool],
    ) -> None:
        """Record one round of ground-truth vs. predicted exclusions."""
        self._init_detection_log()
        cids = list(clients_state.keys())
        self._update_confusion_matrix(cids, clients_state, detected_state)
        if getattr(self, "save_path", None) is not None:
            self._append_round_states_csv(
                server_round, cids, clients_state, detected_state
            )


class CustomKrum_attack(_DetectionLogMixin, MultiKrum):
    """Behaves like Krum but injects configurable Byzantine attacks, mirroring
    the logic of CustomFedAvg_attack from strategy.py."""

    def __init__(
        self,
        *args,
        num_malicious_nodes: int,
        warmup_rounds: int,
        attack_mode: str = "no_attack",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious_nodes = num_malicious_nodes
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
            size = min(self.num_malicious_nodes, len(unique_ids))
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

    def _record_multikrum_detection(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
    ) -> None:
        """Log which clients Multi-Krum's selection excluded this round.

        NOTE on what this measures. Multi-Krum keeps a FIXED number of clients
        (`num_nodes_to_select`), so it excludes exactly
        `n - num_nodes_to_select` participants every round whether or not an
        attack is under way -- the same assumed-M-as-drop-rate property DnC has.
        A false-detection rate of 0 on attacked rounds is therefore arithmetic,
        not discrimination; read it together with the warm-up rounds, where the
        same fixed count falls entirely on honest clients.
        """
        from flwr.serverapp.strategy.multikrum import select_multikrum

        valid = [m for m in attacked_replies if not m.has_error()]
        if not valid:
            return
        # GUARD. select_multikrum() ends in `argsort(scores)[:num_nodes_to_select]`,
        # which does NOT clamp: if the retained-set size is >= the number of
        # replies it returns every one of them and Multi-Krum silently becomes
        # plain FedAvg over all attackers. That failure is invisible in accuracy
        # under an attack strong enough to collapse the model either way (Fang),
        # so fail loudly here -- this runs on the same inputs, one call before
        # aggregate_train's own selection, so it guards the aggregation too.
        if self.num_nodes_to_select >= len(valid):
            raise ValueError(
                f"num_nodes_to_select={self.num_nodes_to_select} >= "
                f"{len(valid)} participating clients: Multi-Krum would retain "
                "every reply and degenerate into FedAvg. Set it to N - M "
                f"(here {len(valid) - self.num_malicious_nodes})."
            )
        contents = [m.content for m in valid]
        selected = select_multikrum(
            contents,
            num_malicious_nodes=self.num_malicious_nodes,
            num_nodes_to_select=self.num_nodes_to_select,
        )
        # select_multikrum returns `[contents[i] for i in best_indices]`, i.e.
        # the same objects. Identity matching is therefore exact -- assert it
        # rather than trust it, because a failure here would silently produce a
        # complete but wrongly-aligned detection table.
        sel_ids = {id(c) for c in selected}
        assert sel_ids <= {id(c) for c in contents}, (
            "select_multikrum returned objects that are not the inputs; "
            "identity matching is invalid and detected_state would be wrong"
        )
        detected_state = {
            m.metadata.src_node_id: (id(m.content) not in sel_ids) for m in valid
        }
        self._record_detection(server_round, clients_state, detected_state)

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
                    ordered_train_replies, clients_state, magnitude=10,
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
            elif self.attack_mode == "fang":
                w_re = {k: v.cpu().numpy() for k, v in arrays.to_torch_state_dict().items()}
                attacked_replies = fang_attack_messages(
                    ordered_train_replies, clients_state, w_re=w_re,
                )
            else:
                raise ValueError(f"Unknown attack_mode: {self.attack_mode}")

            # Which clients did Multi-Krum exclude this round?
            # "Detected" = not in the selected set. We call the same
            # select_multikrum() that MultiKrum.aggregate_train() calls, on the
            # same inputs, rather than re-deriving the score: that cannot drift
            # from Flower's own selection. It costs one extra selection pass per
            # round, which is negligible beside 100 clients x 10 local epochs.
            self._record_multikrum_detection(
                current_round, attacked_replies, clients_state
            )

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





# ══════════════════════════════════════════════════════════════════════════════
#  Attack-mode normalisation
# ══════════════════════════════════════════════════════════════════════════════
# The sweep drivers spell these "min_max" and "no_attack"; two _apply_attack
# implementations below historically tested the un-underscored "minmax" and
# "none". The mismatch was silent in the worst way: FedAvgWithLBFGSHVP fell
# through to its unconditional LIE fallback, so a run labelled min_max actually
# executed LIE, and FLTrustCustomFedAvg returned None and crashed the caller
# with "'NoneType' object is not iterable".
#
# Resolve every spelling here, and treat an unrecognised mode as a hard error:
# silently substituting a different attack corrupts results in a way that looks
# like a finding. The sweep driver catches the exception and records an error
# row, so raising is both safe and visible.
_ATTACK_MODE_ALIASES = {
    "":          "none",
    "none":      "none",
    "no_attack": "none",
    "noattack":  "none",
    "min_max":   "minmax",
    "minmax":    "minmax",
    "gaussian":  "gaussian",
    "fang":      "fang",
    "lie":       "lie",
}


def normalize_attack_mode(attack_mode: Optional[str]) -> str:
    """Canonicalise an attack-mode string; raise on anything unrecognised."""
    key = (attack_mode or "none").strip().lower()
    try:
        return _ATTACK_MODE_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown attack_mode {attack_mode!r}. "
            f"Known: {sorted(set(_ATTACK_MODE_ALIASES))}. "
            "Refusing to silently fall back to a different attack."
        ) from None


class FLTrustCustomFedAvg(_DetectionLogMixin, FedAvg):
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
        server_lr: float = 0.01,
        root_epochs: int = 5,
        root_steps: Optional[int] = None,   # cap batches per epoch
        root_train_repeats: int = 50,        # independent server trains per round (deltas are averaged)
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
        self.root_train_repeats = max(1, int(root_train_repeats))
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
        
        config["lr"] = config["lr"] 
            
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
        global_arrays: Optional[ArrayRecord] = None,
    ) -> List[Message]:
        mode = normalize_attack_mode(self.attack_mode)

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
            w_re = {k: v.cpu().numpy() for k, v in global_arrays.to_torch_state_dict().items()}
            return fang_attack_messages(
                ordered_train_replies,
                clients_state,
                w_re=w_re,
            )

        # LIE is the only remaining mode; normalize_attack_mode() has already
        # rejected anything else, so this can no longer fall through to None.
        assert mode == "lie", mode
        return lie_attack_messages(
            ordered_train_replies,
            clients_state,
            omniscent=True,
        )
    # ------------------------------------------------------------------
    # FLTrust core helpers (based on your original fltrust() logic)
    # ------------------------------------------------------------------

    # BatchNorm running statistics are updated by forward passes, not gradients.
    # Including them in delta vectors corrupts cosine similarity scoring in FLTrust.
    _BN_STAT_SUFFIXES = {"running_mean", "running_var"}

    def _iter_float_keys(self, sd: Dict[str, torch.Tensor]) -> Iterable[str]:
        for k, v in sd.items():
            if not v.is_floating_point():
                continue
            if k.split(".")[-1] in self._BN_STAT_SUFFIXES:
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
                xb = batch["image"] #MNIST or FAMNIST
                #xb = batch["img"] #CIFAR10
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

        # central_param = root delta (averaged over root_train_repeats independent runs)
        if self.root_train_repeats <= 1:
            central_delta_sd = self._compute_root_delta_sd(global_sd)
        else:
            repeats = [self._compute_root_delta_sd(global_sd) for _ in range(self.root_train_repeats)]
            central_delta_sd = {
                k: torch.mean(torch.stack([d[k] for d in repeats]), dim=0)
                for k in repeats[0]
            }

        FLTrustTotalScore = 0.0
        score_list: List[float] = []

        central_v = self._sd_to_vec(central_delta_sd)
        central_norm = float(torch.linalg.norm(central_v)) + self.eps  # ‖central‖ + eps (matches original norm_clip)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        sum_delta: Optional[Dict[str, torch.Tensor]] = None

        # Build local deltas list (params)
        local_deltas: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for msg in train_replies:
            cid = int(msg.metadata.src_node_id)
            client_sd = msg.content["arrays"].to_torch_state_dict()
            local_delta_sd = self._delta_sd(client_sd, global_sd)
            local_deltas.append((cid, local_delta_sd))

        # Trust score per client id, for detection logging. Keyed by cid inside
        # this loop rather than zipped against a separate id list afterwards, so
        # the mapping cannot be misaligned by a reordering elsewhere.
        self.last_trust_scores: Dict[int, float] = {}

        for cid, local_delta_sd in local_deltas:
            local_v = self._sd_to_vec(local_delta_sd)
            client_cos = float(cos(central_v, local_v))
            client_cos = max(client_cos, 0.0)

            # norm_clip: ‖central‖ / ‖local‖  (original: vnum / norm2 + 1e-9)
            local_norm = float(torch.linalg.norm(local_v))
            client_clipped_value = central_norm / (local_norm + self.eps)

            score_list.append(client_cos)
            self.last_trust_scores[cid] = client_cos
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
        # Keys not in sum_delta (num_batches_tracked, running_mean, running_var, non-float)
        # are left unchanged — matching the original FLTrust model2vector exclusion logic.
        new_sd = dict(global_sd)
        for k in sum_delta:
            if self.layer_filter is not None and (not self.layer_filter(k)):
                continue

            temp = sum_delta[k] / FLTrustTotalScore
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
            attacked_replies = self._apply_attack(ordered_train_replies, clients_state, malicious_ids, arrays)
            # --------- FLTrust aggregation replaces FedAvg aggregate_train ---------
            agg_arrays, agg_train_metrics = self._fltrust_aggregate_arrays(
                global_arrays=arrays,
                train_replies=attacked_replies,
            )

            # Which clients did FLTrust reject this round?
            #
            # FLTrust DOWNWEIGHTS rather than excludes, so "detected" is not as
            # crisp as it is for a filter. We take the one non-arbitrary point
            # available: the trust score is a ReLU'd cosine, and a client scoring
            # exactly 0 contributes nothing to the aggregate -- FLTrust's own
            # rejection mechanism, not a threshold we chose.
            #
            # Two caveats travel with any FDR computed from this. A client at
            # cos = 0.001 contributes ~nothing yet counts as "not detected", so
            # the figure understates rejection; and under non-IID data a benign
            # client can legitimately land at negative cosine, be clipped to 0,
            # and count as a false positive. It is not commensurate with the
            # false-detection rate of a hard filter.
            self._record_detection(
                current_round,
                clients_state,
                {cid: (score <= 0.0)
                 for cid, score in getattr(self, "last_trust_scores", {}).items()},
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
    last_hvp: Optional[np.ndarray] = None   # cached HVP from last round, used for pre-agg filtering

     # NEW: lưu lịch sử normalized distances cho từng client
    norm_dist_hist: Dict[int, Deque[float]] = field(default_factory=dict)

    # NEW: suspicious score hiện tại của từng client
    suspicious_scores: Dict[int, float] = field(default_factory=dict)

    # NEW: tập client đã bị detector gắn cờ
    detected_malicious: Set[int] = field(default_factory=set)


class FedAvgWithLBFGSHVP(_DetectionLogMixin, FedAvg):
    """
    FedAvg-style strategy + optional attack injection
    + FLDetector-style malicious-client detection.

    Main differences vs the old version:
      - No keep_fraction ranking filter
      - No comparison against ground-truth malicious ids
      - Uses suspicious score = moving average of normalized distances
      - Uses Gap statistics + k-means to detect malicious clients
      - Can remove detected clients and optionally restart training
    """

    def __init__(
        self,
        *args,
        # ---- Attack config ----
        num_malicious: int = 0,
        warmup_rounds_attack: int = 0,
        attack_mode: str = "none",

        # ---- HVP / L-BFGS config ----
        hvp_cfg: Optional["HVPConfig"] = None,

        # ---- FLDetector-style config ----
        suspicion_window: int = 10,     # N
        detect_from_round: int = 50,    # start detection after warmup
        gap_B: int = 20,                # number of reference samples
        gap_K: int = 10,                # max clusters for gap statistic
        remove_detected_clients: bool = True,
        restart_on_detection: bool = False,
        max_restarts: int = 1,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Save/logging
        self.save_path: Optional[Path] = None
        self.run_dir: Optional[str] = None
        self.best_acc_so_far: float = 0.0

        # Attack config
        self.num_malicious = int(num_malicious)
        self.warmup_rounds_attack = int(warmup_rounds_attack)
        self.attack_mode = (attack_mode or "none").lower()

        self.clients_state: Dict[int, bool] = {}
        self.fixed_malicious_ids: Optional[List[int]] = None
        self.all_client_ids: Set[int] = set()

        # HVP / L-BFGS config
        self.hvp_cfg = hvp_cfg or HVPConfig()
        self.hvp_state = HVPState(
            S_hist=deque(maxlen=self.hvp_cfg.max_hist),
            Y_hist=deque(maxlen=self.hvp_cfg.max_hist),
        )

        # FLDetector config
        self.suspicion_window = int(suspicion_window)
        self.detect_from_round = int(detect_from_round)
        self.gap_B = int(gap_B)
        self.gap_K = int(gap_K)
        self.remove_detected_clients = bool(remove_detected_clients)
        self.restart_on_detection = bool(restart_on_detection)
        self.max_restarts = int(max_restarts)
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    def set_save_path_and_run_dir(self, path: Path, run_dir: str) -> None:
        self.save_path = path
        self.run_dir = run_dir

    def save_metrics_as_json(self, current_round: int, result: "Result") -> None:
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

    def configure_train(
        self, server_round: int, arrays: "ArrayRecord", config: "ConfigRecord", grid: "Grid"
    ) -> Iterable["Message"]:
        config["lr"] = config["lr"]
        return super().configure_train(server_round, arrays, config, grid)

    # ------------------------------------------------------------------
    # Attack helpers
    # ------------------------------------------------------------------

    def _select_malicious_and_order(
        self,
        server_round: int,
        train_replies: List["Message"],
    ) -> Tuple[List["Message"], Dict[int, bool], List[int]]:
        node_ids = [int(msg.metadata.src_node_id) for msg in train_replies]
        unique_ids = sorted(set(node_ids))
        self.all_client_ids.update(unique_ids)

        # During attack warmup: no malicious or fixed list (if already set)
        if server_round <= self.warmup_rounds_attack:
            malicious_ids: List[int] = self.fixed_malicious_ids or []
            clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
            ordered = sorted(train_replies, key=lambda m: int(m.metadata.src_node_id))
            self.clients_state = clients_state
            return ordered, clients_state, malicious_ids

        # Fix malicious IDs once
        if self.fixed_malicious_ids is None:
            size = min(self.num_malicious, len(unique_ids))
            self.fixed_malicious_ids = (
                list(self._rng.choice(unique_ids, size=size, replace=False))
                if size > 0 else []
            )

        malicious_ids = self.fixed_malicious_ids
        clients_state = {nid: (nid in malicious_ids) for nid in unique_ids}
        ordered = sorted(train_replies, key=lambda m: int(m.metadata.src_node_id))

        self.clients_state = clients_state
        return ordered, clients_state, malicious_ids

    def _apply_attack(
        self,
        ordered_train_replies: List["Message"],
        clients_state: Dict[int, bool],
        malicious_ids: List[int],
        global_arrays: Optional["ArrayRecord"] = None,
    ) -> List["Message"]:
        mode = normalize_attack_mode(self.attack_mode)

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
            if global_arrays is None:
                return ordered_train_replies
            w_re = {k: v.cpu().numpy() for k, v in global_arrays.to_torch_state_dict().items()}
            return fang_attack_messages(
                ordered_train_replies,
                clients_state,
                w_re=w_re,
            )

        # LIE is the only remaining mode; normalize_attack_mode() rejects
        # anything else, so this is no longer a catch-all that would silently
        # run LIE for a mode it did not recognise.
        assert mode == "lie", mode
        return lie_attack_messages(
            ordered_train_replies,
            clients_state,
            omniscent=True,
        )

    # ------------------------------------------------------------------
    # Array / HVP helpers
    # ------------------------------------------------------------------

    def _extract_client_arrays(self, msg: "Message") -> Optional["ArrayRecord"]:
        if msg.has_error() or msg.content is None:
            return None
        if self.hvp_cfg.array_key not in msg.content:
            return None
        return msg.content[self.hvp_cfg.array_key]  # type: ignore[return-value]

    def _update_lbfgs_and_get_hvp(
        self,
        *,
        current_round: int,
        global_prev: "ArrayRecord",
        global_curr: Optional["ArrayRecord"],
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
        self.hvp_state.last_hvp = hvp
        return hvp

    def _hvp_detect(
        self,
        *,
        current_round: int,
        global_prev: "ArrayRecord",
        global_curr: Optional["ArrayRecord"],
        train_replies: List["Message"],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]:
        """
        Compute FLDetector-style raw distances and per-round L1-normalized distances.

        raw_scores[node_id]        = ||pred_delta_i - delta_i||_2
        normalized_scores[node_id] = raw_scores / sum_j raw_scores[j]
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
                # fallback before HVP becomes available
                prev_delta_i = self.hvp_state.last_client_delta[node_id]
                raw_scores[node_id] = float(np.linalg.norm(prev_delta_i - delta_i))

            self.hvp_state.last_client_delta[node_id] = delta_i

        normalized_scores: Dict[int, float] = {}
        l1 = float(sum(raw_scores.values()))
        if l1 > 0.0:
            normalized_scores = {cid: (val / l1) for cid, val in raw_scores.items()}
        else:
            normalized_scores = {cid: 0.0 for cid in raw_scores.keys()}

        vals = np.array(list(raw_scores.values()), dtype=np.float64) if raw_scores else np.array([])
        stats = {
            "hvp/enabled": float(1.0 if hvp is not None else 0.0),
            "hvp/scored_clients": float(len(raw_scores)),
            "hvp/raw_score_mean": float(np.mean(vals)) if len(vals) else 0.0,
            "hvp/raw_score_max": float(np.max(vals)) if len(vals) else 0.0,
            "hvp/raw_score_min": float(np.min(vals)) if len(vals) else 0.0,
        }
        return raw_scores, normalized_scores, stats

    # ------------------------------------------------------------------
    # Suspicious score helpers
    # ------------------------------------------------------------------

    def _get_norm_hist(self, node_id: int) -> Deque[float]:
        hist = self.hvp_state.norm_dist_hist.get(node_id)
        if hist is None:
            hist = deque(maxlen=self.suspicion_window)
            self.hvp_state.norm_dist_hist[node_id] = hist
        return hist

    def _update_suspicious_scores(
        self,
        normalized_scores: Dict[int, float],
    ) -> Dict[int, float]:
        for node_id, val in normalized_scores.items():
            self._get_norm_hist(node_id).append(float(val))

        updated: Dict[int, float] = {}
        for node_id, hist in self.hvp_state.norm_dist_hist.items():
            if len(hist) == 0:
                continue
            updated[node_id] = float(np.mean(hist))

        self.hvp_state.suspicious_scores = updated
        return updated

    # ------------------------------------------------------------------
    # Gap statistic + 1D k-means
    # ------------------------------------------------------------------

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

        # deterministic quantile init
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
        ids = sorted(suspicious_scores.keys())
        x = np.array([suspicious_scores[i] for i in ids], dtype=np.float64)

        if len(x) < 2 or np.allclose(x, x[0]):
            return 1, {
                "detect/gap_k": 1.0,
                "detect/gap_value_k1": 0.0,
                "detect/gap_value_khat": 0.0,
            }

        x = self._normalize_unit_interval(x)
        K = max(1, min(self.gap_K, len(np.unique(x))))
        if K == 1:
            return 1, {
                "detect/gap_k": 1.0,
                "detect/gap_value_k1": 0.0,
                "detect/gap_value_khat": 0.0,
            }

        eps = 1e-12
        gaps = np.zeros(K + 1, dtype=np.float64)
        s_vals = np.zeros(K + 1, dtype=np.float64)

        for k in range(1, K + 1):
            _, _, Wk = self._kmeans_1d(x, k)
            ref_logs: List[float] = []

            for _ in range(self.gap_B):
                ref = self._rng.uniform(0.0, 1.0, size=len(x))
                _, _, Wkb = self._kmeans_1d(ref, k)
                ref_logs.append(float(np.log(Wkb + eps)))

            ref_logs_arr = np.asarray(ref_logs, dtype=np.float64)
            gaps[k] = float(np.mean(ref_logs_arr) - np.log(Wk + eps))
            sd_k = float(np.sqrt(np.mean((ref_logs_arr - np.mean(ref_logs_arr)) ** 2)))
            s_vals[k] = float(np.sqrt((1.0 + self.gap_B) / self.gap_B) * sd_k)

        # choose smallest k such that Gap(k) - Gap(k+1) + s_{k+1} >= 0
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
        *,
        current_round: int,
        active_node_ids: List[int],
    ) -> Tuple[Set[int], Dict[str, float]]:
        if current_round < self.detect_from_round:
            return set(), {
                "detect/enabled": 0.0,
                "detect/eligible_clients": 0.0,
                "detect/detected_count": 0.0,
            }

        eligible_scores: Dict[int, float] = {}
        for node_id in sorted(set(active_node_ids)):
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
            ids = sorted(eligible_scores.keys())
            x = np.array([eligible_scores[i] for i in ids], dtype=np.float64)
            x_norm = self._normalize_unit_interval(x)

            labels, _, _ = self._kmeans_1d(x_norm, k=2)

            cluster_mean: Dict[int, float] = {}
            for c in [0, 1]:
                mask = labels == c
                cluster_mean[c] = float(np.mean(x[mask])) if np.any(mask) else -np.inf

            malicious_cluster = max(cluster_mean, key=cluster_mean.get)
            detected = {ids[i] for i, lab in enumerate(labels) if lab == malicious_cluster}

        stats = {
            "detect/enabled": 1.0,
            "detect/eligible_clients": float(len(eligible_scores)),
            "detect/detected_count": float(len(detected)),
            "detect/suspicion_mean_active": float(np.mean(list(eligible_scores.values()))) if eligible_scores else 0.0,
            "detect/suspicion_max_active": float(np.max(list(eligible_scores.values()))) if eligible_scores else 0.0,
            **gap_stats,
        }
        return detected, stats

    # ------------------------------------------------------------------
    # Client removal / reset helpers
    # ------------------------------------------------------------------

    def _filter_removed_clients(self, replies: List["Message"]) -> List["Message"]:
        if not self.hvp_state.removed_client_ids:
            return replies

        kept = [
            msg for msg in replies
            if int(msg.metadata.src_node_id) not in self.hvp_state.removed_client_ids
        ]
        dropped = len(replies) - len(kept)
        if dropped > 0:
            log(
                INFO,
                "\t└──> Skipped %d replies from previously removed clients",
                dropped,
            )
        return kept

    def _register_detected_clients(self, detected: Set[int]) -> None:
        if not self.remove_detected_clients or not detected:
            return
        self.hvp_state.removed_client_ids.update(detected)

    def _reset_detector_state(self, keep_removed_clients: bool = True) -> None:
        removed = set(self.hvp_state.removed_client_ids) if keep_removed_clients else set()
        self.hvp_state = HVPState(
            S_hist=deque(maxlen=self.hvp_cfg.max_hist),
            Y_hist=deque(maxlen=self.hvp_cfg.max_hist),
        )
        self.hvp_state.removed_client_ids = removed

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    def start(
        self,
        grid: "Grid",
        initial_arrays: "ArrayRecord",
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional["ConfigRecord"] = None,
        evaluate_config: Optional["ConfigRecord"] = None,
        evaluate_fn: Optional[Callable[[int, "ArrayRecord"], Optional["MetricRecord"]]] = None,
    ) -> "Result":
        self.best_acc_so_far = 0.0
        self._reset_detector_state(keep_removed_clients=False)

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        # Evaluate initial global model once
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        # We separate:
        # - phase_round: FL round inside the current training phase
        # - global_step: monotonically increasing counter for result logging
        phase_round = 1
        global_step = 0
        restart_count = 0

        while phase_round <= num_rounds:
            global_step += 1

            log(INFO, "")
            log(
                INFO,
                "[GLOBAL STEP %s | PHASE ROUND %s/%s | RESTARTS %s]",
                global_step,
                phase_round,
                num_rounds,
                restart_count,
            )

            # ------------------- TRAIN -------------------
            train_replies_iter = grid.send_and_receive(
                messages=self.configure_train(phase_round, arrays, train_config, grid),
                timeout=timeout,
            )
            train_replies = list(train_replies_iter)

            # Ignore already-removed clients
            train_replies = self._filter_removed_clients(train_replies)

            ordered, clients_state, malicious_ids = self._select_malicious_and_order(
                phase_round,
                train_replies,
            )

            attacked_replies = self._apply_attack(
                ordered,
                clients_state,
                malicious_ids,
                arrays,
            )

            # Aggregate only on currently active clients
            agg_arrays, agg_train_metrics = self.aggregate_train(phase_round, attacked_replies)

            # HVP-based distances on all active replies
            raw_scores, normalized_scores, hvp_stats = self._hvp_detect(
                current_round=phase_round,
                global_prev=arrays,
                global_curr=agg_arrays,
                train_replies=attacked_replies,
            )

            suspicious_scores = self._update_suspicious_scores(normalized_scores)

            ids_in_round = [int(m.metadata.src_node_id) for m in attacked_replies]
            new_detected, detect_stats = self._detect_malicious_from_suspicion(
                current_round=phase_round,
                active_node_ids=ids_in_round,
            )

            # Optional logging
            if suspicious_scores:
                ranked_susp = sorted(
                    suspicious_scores.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[: min(10, len(suspicious_scores))]
                log(INFO, "\t└──> Top suspicious scores: %s", ranked_susp)

            if new_detected:
                detected_ranked = sorted(
                    ((cid, suspicious_scores.get(cid, 0.0)) for cid in new_detected),
                    key=lambda x: x[1],
                    reverse=True,
                )
                log(INFO, "\t└──> Newly detected malicious clients: %s", detected_ranked)
                self._register_detected_clients(new_detected)

            # Which clients is FLDetector currently excluding?
            #
            # "Detected" = in the cumulative removed set, which is exactly what
            # _filter_removed_clients() drops from both training and evaluation.
            # The set is cumulative by design: FLDetector does not re-admit.
            #
            # READ THIS BEFORE INTERPRETING THE NUMBERS. Detection is gated by
            # `detect_from_round` (default 50) and additionally requires
            # `suspicion_window` (default 10) rounds of history. On a 50-round
            # run the gate opens on the final round at the earliest, so the
            # removed set stays EMPTY and this logs recall 0.0 / FDR 0.0 for
            # every round. That is a faithful record of a detector that never
            # got to run, not evidence of a clean false-positive rate.
            # NB: this loop's round counter is `phase_round`, not `current_round`
            # -- FLDetector can restart training, so rounds are counted per phase.
            removed = set(getattr(self.hvp_state, "removed_client_ids", set()) or set())
            self._record_detection(
                phase_round,
                clients_state,
                {cid: (cid in removed) for cid in clients_state},
            )

            # Update arrays for current phase
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays

            # Merge train metrics
            merged: Dict[str, float] = {}
            if agg_train_metrics is not None:
                merged.update(dict(agg_train_metrics))
            merged.update(hvp_stats)
            merged.update(detect_stats)
            merged["detect/removed_total"] = float(len(self.hvp_state.removed_client_ids))
            merged["detect/restart_count"] = float(restart_count)

            if suspicious_scores:
                vals = np.array(list(suspicious_scores.values()), dtype=np.float64)
                merged["detect/suspicious_score_mean_all"] = float(np.mean(vals))
                merged["detect/suspicious_score_max_all"] = float(np.max(vals))

            if merged:
                mr = MetricRecord(merged)
                log(INFO, "\t└──> Aggregated MetricRecord: %s", mr)
                result.train_metrics_clientapp[global_step] = mr

            # ------------------- EVAL (clients) -------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(phase_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )
            evaluate_replies = self._filter_removed_clients(list(evaluate_replies))
            agg_evaluate_metrics = self.aggregate_evaluate(phase_round, evaluate_replies)
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[global_step] = agg_evaluate_metrics

            # ------------------- EVAL (server) -------------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(phase_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[global_step] = res

            self.save_metrics_as_json(current_round=global_step, result=result)

            # ------------------- Optional restart -------------------
            if (
                self.restart_on_detection
                and len(new_detected) > 0
                and restart_count < self.max_restarts
            ):
                restart_count += 1
                log(
                    INFO,
                    "\t└──> Restarting training from initial model after detection "
                    "(restart %d/%d). Removed clients stay excluded: %s",
                    restart_count,
                    self.max_restarts,
                    sorted(self.hvp_state.removed_client_ids),
                )

                arrays = initial_arrays
                self._reset_detector_state(keep_removed_clients=True)

                # keep fixed malicious ids as-is for attack reproducibility
                phase_round = 1
                continue

            phase_round += 1

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        return result







class CustomFedAvg_attack(_DetectionLogMixin, FedAvg):

    def __init__(
        self,
        *args,
        num_malicious: int ,
        warmup_rounds: int ,
        attack_mode: str = "no_attack",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds
        self.attack_mode = attack_mode

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

            # Choose attack based on self.attack_mode
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
            elif self.attack_mode == "fang":
                w_re = {k: v.cpu().numpy() for k, v in arrays.to_torch_state_dict().items()}
                attacked_replies = fang_attack_messages(
                    ordered_train_replies, clients_state, w_re=w_re,
                )
            else:
                raise ValueError(f"Unknown attack_mode: {self.attack_mode}")

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
            #     omniscent=True,         
            # )

            # --- Min-Max attack ---
            # attacked_replies = minmax_attack_messages(
            #     ordered_train_replies,
            #     clients_state,
            #     omniscent=True,
            #     threshold=1e-5,
            #     lambda_init=5.0,
            #     malicious_num=len(malicious_cids),
            # )

            # Undefended FedAvg aggregates every client, so nothing is ever excluded.
            # Recording that explicitly -- rather than leaving the confusion matrix at its
            # initial zeros and reporting NaN -- makes the no-defense row measurable on the
            # same footing as every other method: detection rate 0, false-detection rate 0,
            # and an attack success rate equal to the malicious fraction of client-rounds.
            self._record_detection(
                current_round,
                clients_state,
                {cid: False for cid in clients_state},
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



class CustomFedAvg_adaptive(FedAvg):

    def __init__(
        self,
        *args,
        num_malicious: int,
        warmup_rounds: int,
        attack_mode: str = "no_attack",
        anom_decay: float = 0.6,
        tau_s_quantile: float = 0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds
        self.attack_mode = attack_mode

        # Existing fields
        self.clients_state = {}
        self.fixed_malicious_ids: list[int] | None = None
        self.all_client_ids: set[int] = set()

        self.round_log_initialized: bool = False
        self.confusion_matrix: Dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.best_acc_so_far = 0.0
        self.best_arrays: Optional[ArrayRecord] = None

        # ==== ConvLSTM / anomaly state ====
        self.feature_H = 16
        self.feature_W = 16
        self.feature_C = 6
        self.window_len = 5
        self.bootstrap_B = 20

        self.client_feature_history: Dict[int, List[torch.Tensor]] = {}
        self.history_max = 10

        self.detector = ConvLSTMForecaster(
            in_channels=self.feature_C,
            hidden_dim=16,
            kernel_size=3,
        )
        self.detector.eval()

        # ==== SVD latent projection state ====
        self.svd_target_len = self.feature_H * self.feature_W  # usually 256
        self.svd_mean: Optional[torch.Tensor] = None           # [D_full]
        self.svd_basis: Optional[torch.Tensor] = None          # [K, D_full]
        self.svd_ready: bool = False

        # warmup bootstrap buffer for fitting fixed basis
        self.svd_bootstrap_buffer: List[torch.Tensor] = []
        self.svd_bootstrap_min_samples: int = 64

        # ==== Reduced-frequency anomaly scoring (EWMA + triggers) ====
        self.anom_decay: float = anom_decay
        
        self.tau_s_quantile: float = float(np.clip(tau_s_quantile, 0.0, 1.0))

        # cold-start: score all clients for first n rounds
        self.score_all_first_n_rounds: int = 8

        self.score_refresh_max: int = 10

        # Adaptive cap: EWMA spread drives how many clients are scored per round
        self.adaptive_cap_min: int = 10   # floor — always score at least this many
        self.adaptive_cap_max: int = 80   # ceiling — never score more than this many
        self.adaptive_cap_smooth: float = 0.7  # EMA smoothing across rounds
        self._adaptive_cap: int = 50      # running smoothed cap, initialised to midpoint
        self.max_scores_per_round: int = 50   # kept for CSV logging compatibility

        self.ewma_score: Dict[int, float] = {}
        self.ewma_prev: Dict[int, float] = {}
        self.ewma_m2: Dict[int, float] = {}
        self.last_scored_round: Dict[int, int] = {}

        self.anomaly_csv_initialized = False
        self.round_global_arrays: Optional[ArrayRecord] = None
        self.last_anomaly_scores: Dict[int, float] = {}

        # ===== Semi-online selective update =====
        self.semi_online = True
        self.prev_server_acc: Optional[float] = None
        self.prev_server_loss: Optional[float] = None
        self.good_streak = 0
        self.K_good = 2

        self.acc_improve_min = 1e-4
        self.loss_improve_min = 1e-4

        self.good_client_quantile = 0.5

        self.update_every = 2
        self.buffer_min_windows = 64
        self.max_steps_per_update = 50
        self.detector_lr = 1e-4
        self.batch_size = 8

        self.clean_window_buffer: List[torch.Tensor] = []
        self.max_buffer = 2000
        self.last_detector_update_round = 0
        self.detector_optimizer = None

        self.round_sequences: Dict[int, torch.Tensor] = {}
        self.tb_writer: Optional[SummaryWriter] = None

        # ==== cost / compute logging ====
        self.cost_stats_initialized = False
        self.round_n_clients: int = 0
        self.round_n_scored: int = 0
        self.round_n_refresh: int = 0
        self.round_n_trigger_risk: int = 0
        self.round_n_trigger_drift: int = 0
        self.round_n_trigger_high_ewma: int = 0
        self.round_t_convlstm: float = 0.0

        self.total_convlstm_forwards: int = 0
        self.total_t_convlstm: float = 0.0


        # ==== Adaptive quantile filtering state ====
        self.filter_quantile_min: float = 0.35
        self.filter_quantile_max: float = 0.90
        self.filter_quantile_default: float = 0.55
        self.current_filter_quantile: float = self.filter_quantile_default

        # target drop-rate controller
        self.target_drop_rate: float = 0.20
        self.last_drop_rate: float = 0.0

        # how strongly each signal affects q
        self.q_weight_spread: float = 0.35
        self.q_weight_anomaly: float = 0.30
        self.q_weight_server: float = 0.20
        self.q_weight_drop: float = 0.15

        # optional smoothing so q does not jump too hard round-to-round
        self.q_smooth: float = 0.7

    # ------------------------------------------------------------------
    # Fixed-SVD helpers
    # ------------------------------------------------------------------

    def _layer_filter(self, name: str) -> bool:
        return ("layer4" in name) or ("fc" in name)
    
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
            thr = float(np.quantile(scores_arr, float(np.clip(quantile, 0.0, 1.0))))
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
    
    # def _round_is_good(self, acc_val: float, loss_val: float) -> bool:
    #     """Return True if we have K_good consecutive 'good' rounds based on server eval."""
    #     if self.prev_server_acc is None or self.prev_server_loss is None:
    #         self.prev_server_acc = acc_val
    #         self.prev_server_loss = loss_val
    #         self.good_streak = 0
    #         return False

    def _round_is_good(self, acc_val: float, loss_val: float) -> bool:
        """
        Return True once we have K_good consecutive 'good' rounds.

        A round is considered good if:
        - accuracy improved by at least acc_improve_min, and/or
        - loss decreased by at least loss_improve_min

        Both thresholds are interpreted as minimum required improvement.
        """
        # First observed round: initialize baseline, but don't count it as "good"
        if self.prev_server_acc is None or self.prev_server_loss is None:
            self.prev_server_acc = acc_val
            self.prev_server_loss = loss_val
            self.good_streak = 0
            return False

        acc_improved = (acc_val - self.prev_server_acc) >= self.acc_improve_min
        loss_improved = (self.prev_server_loss - loss_val) >= self.loss_improve_min

        # Treat round as good if either metric improved enough
        is_good_now = acc_improved or loss_improved

        if is_good_now:
            self.good_streak += 1
        else:
            self.good_streak = 0

        # Update baseline for next round
        self.prev_server_acc = acc_val
        self.prev_server_loss = loss_val

        return self.good_streak >= self.K_good
        
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


    def _clip01(self, x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))
    def _get_adaptive_filter_quantile(
        self,
        server_round: int,
        cids: List[int],
    ) -> float:
        """
        Option 4: combine multiple signals to choose filtering quantile.

        Lower q  -> stricter filtering
        Higher q -> looser filtering

        Signals:
        1) anomaly-score spread      (large spread => stricter)
        2) anomaly-score severity    (large mean anomaly => stricter)
        3) server trend              (good streak => looser)
        4) recent drop rate feedback (too many drops => looser)
        """
        q_min = float(np.clip(self.filter_quantile_min, 0.0, 1.0))
        q_max = float(np.clip(self.filter_quantile_max, 0.0, 1.0))
        if q_min > q_max:
            q_min, q_max = q_max, q_min

        scores = np.array(
            [self.last_anomaly_scores[cid] for cid in cids if cid in self.last_anomaly_scores],
            dtype=np.float32,
        )

        # If too few scores exist, fall back to persistent/default q
        if scores.size < 4:
            q = float(np.clip(self.current_filter_quantile, q_min, q_max))
            log(
                INFO,
                "[Round %s] Adaptive q fallback=%.3f (not enough scores: n=%d)",
                server_round,
                q,
                scores.size,
            )
            return q

        # ---------------------------------------------------------
        # 1) Spread signal: large IQR/median => more separability/risk => stricter
        # ---------------------------------------------------------
        q25, q50, q75 = np.quantile(scores, [0.25, 0.50, 0.75])
        iqr = float(q75 - q25)
        spread_ratio = iqr / (abs(float(q50)) + 1e-6)

        # map spread_ratio to [0,1], larger => stricter
        spread_signal = self._clip01(spread_ratio / (spread_ratio + 1.0))

        # ---------------------------------------------------------
        # 2) Severity signal: larger score mean relative to std => stricter
        # ---------------------------------------------------------
        score_mean = float(scores.mean())
        score_std = float(scores.std() + 1e-6)
        severity_ratio = score_mean / score_std
        severity_signal = self._clip01(severity_ratio / (severity_ratio + 1.0))

        # ---------------------------------------------------------
        # 3) Server signal: good streak => looser, no progress => stricter
        # convert to a "strictness" signal
        # ---------------------------------------------------------
        if self.good_streak >= self.K_good:
            server_signal = 0.0   # healthy => less strict
        elif self.good_streak == 0:
            server_signal = 1.0   # not improving => more strict
        else:
            # intermediate
            server_signal = 0.5

        # ---------------------------------------------------------
        # 4) Drop-rate feedback: if we already dropped too much, loosen
        # strictness decreases when last_drop_rate > target_drop_rate
        # ---------------------------------------------------------
        drop_excess = self.last_drop_rate - self.target_drop_rate
        # positive excess means "too aggressive recently", so reduce strictness
        drop_relief = self._clip01((drop_excess + 0.20) / 0.40)
        # convert relief to strictness contribution
        drop_signal = 1.0 - drop_relief

        # ---------------------------------------------------------
        # Combine into strictness in [0,1]
        # strictness high => lower quantile
        # ---------------------------------------------------------
        strictness = (
            self.q_weight_spread * spread_signal
            + self.q_weight_anomaly * severity_signal
            + self.q_weight_server * server_signal
            + self.q_weight_drop * drop_signal
        )

        strictness = self._clip01(strictness)

        # Map strictness -> quantile
        # strictness=0 => q_max
        # strictness=1 => q_min
        q_raw = q_max - strictness * (q_max - q_min)

        # Smooth with previous q to avoid oscillation
        q_prev = float(self.current_filter_quantile)
        q = self.q_smooth * q_prev + (1.0 - self.q_smooth) * q_raw
        q = float(np.clip(q, q_min, q_max))

        self.current_filter_quantile = q

        log(
            INFO,
            (
                "[Round %s] Adaptive q=%.3f | strictness=%.3f | "
                "spread_signal=%.3f severity_signal=%.3f "
                "server_signal=%.3f drop_signal=%.3f | "
                "iqr=%.6f median=%.6f mean=%.6f std=%.6f last_drop=%.3f"
            ),
            server_round,
            q,
            strictness,
            spread_signal,
            severity_signal,
            server_signal,
            drop_signal,
            iqr,
            q50,
            score_mean,
            score_std,
            self.last_drop_rate,
        )
        return q

    def _append_cost_csv(self, server_round: int) -> None:
        csv_path = f"{self.save_path}/convlstm_cost.csv"
        mode = "a" if self.cost_stats_initialized else "w"

        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self.cost_stats_initialized:
                writer.writerow([
                    "round",
                    "n_clients",
                    "n_scored",
                    "ratio_scored",
                    "t_convlstm",
                    "trigger_refresh",
                    "trigger_risk",
                    "trigger_drift",
                    "trigger_high_ewma",
                    "filter_quantile",
                ])
                self.cost_stats_initialized = True

            writer.writerow([
                server_round,
                self.round_n_clients,
                self.round_n_scored,
                self.round_n_scored / max(self.round_n_clients, 1),
                self.round_t_convlstm,
                self.round_n_refresh,
                self.round_n_trigger_risk,
                self.round_n_trigger_drift,
                self.round_n_trigger_high_ewma,
                self.current_filter_quantile,
            ])

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
            thr = float(np.quantile(scores_arr, float(np.clip(quantile, 0.0, 1.0))))
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

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

    def _get_selected_layer_names(self, global_state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """
        Use a fixed layer ordering so basis fitting and runtime projection
        always flatten parameters in the exact same order.
        """
        return [
            name
            for name in global_state_dict.keys()
            if self._layer_filter(name) and global_state_dict[name].is_floating_point()
        ]

    def _collect_raw_update_vector(
        self,
        client_state_dict: Dict[str, torch.Tensor],
        global_state_dict: Dict[str, torch.Tensor],
        selected_names: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Build the FULL raw delta vector [D_full] for one client using a fixed
        parameter ordering.
        """
        vecs: List[torch.Tensor] = []

        for name in selected_names:
            if name not in client_state_dict:
                continue
            p = client_state_dict[name]
            g = global_state_dict[name]

            if not p.is_floating_point():
                continue

            delta = (p - g).detach().cpu().flatten().float()
            vecs.append(delta)

        if not vecs:
            return None

        return torch.cat(vecs, dim=0)

    def _fit_fixed_svd_basis_from_vectors(self, vectors: List[torch.Tensor]) -> bool:
        """
        Fit a fixed PCA/SVD-style basis from a list of raw client update vectors.
        Stores:
          self.svd_mean  [D]
          self.svd_basis [K, D]
        """
        if len(vectors) < 2:
            return False

        valid = [v.detach().cpu().float() for v in vectors if v is not None]
        if len(valid) < 2:
            return False

        dim = max(v.numel() for v in valid)
        padded = []
        for v in valid:
            if v.numel() < dim:
                pad = torch.zeros(dim - v.numel(), dtype=v.dtype)
                v = torch.cat([v, pad], dim=0)
            elif v.numel() > dim:
                v = v[:dim]
            padded.append(v)

        X = torch.stack(padded, dim=0)  # [N, D]
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean

        # Degenerate case
        if torch.allclose(Xc, torch.zeros_like(Xc)):
            return False

        # SVD: Xc = U S Vh
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        k = min(self.svd_target_len, Vh.shape[0])
        basis = Vh[:k].detach().cpu().float()  # [K, D]

        self.svd_mean = mean.squeeze(0).detach().cpu().float()
        self.svd_basis = basis
        self.svd_ready = True

        log(
            INFO,
            "Fixed SVD basis fitted: samples=%d dim=%d k=%d",
            X.shape[0],
            X.shape[1],
            k,
        )
        return True

    def _maybe_fit_svd_from_warmup_replies(
        self,
        train_replies: List[Message],
        global_arrays: ArrayRecord,
    ) -> None:
        """
        During warmup rounds, collect raw update vectors and fit a fixed basis once
        enough benign samples are available.
        """
        if self.svd_ready:
            return

        gsd = global_arrays.to_torch_state_dict()
        selected_names = self._get_selected_layer_names(gsd)

        for msg in train_replies:
            csd = msg.content["arrays"].to_torch_state_dict()
            raw_v = self._collect_raw_update_vector(csd, gsd, selected_names)
            if raw_v is not None:
                self.svd_bootstrap_buffer.append(raw_v)

        # cap buffer size
        if len(self.svd_bootstrap_buffer) > 4096:
            self.svd_bootstrap_buffer = self.svd_bootstrap_buffer[-4096:]

        if len(self.svd_bootstrap_buffer) >= self.svd_bootstrap_min_samples:
            fitted = self._fit_fixed_svd_basis_from_vectors(self.svd_bootstrap_buffer)
            if fitted:
                log(INFO, "SVD basis is now ready for ConvLSTM analysis")

    def _project_raw_vector_fixed_svd(self, raw_v: torch.Tensor) -> torch.Tensor:
        """
        Project a raw delta vector into the fixed latent space.

        Returns:
            z: [self.svd_target_len]
        """
        if not self.svd_ready or self.svd_mean is None or self.svd_basis is None:
            raise RuntimeError("SVD basis is not ready")

        mean = self.svd_mean.detach().cpu().float()
        basis = self.svd_basis.detach().cpu().float()  # [K, D]
        basis_k, basis_dim = basis.shape

        x = raw_v.detach().cpu().float()

        if x.numel() < basis_dim:
            pad = torch.zeros(basis_dim - x.numel(), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        elif x.numel() > basis_dim:
            x = x[:basis_dim]

        x_centered = x - mean
        z = torch.matmul(basis, x_centered)  # [K]

        if z.numel() < self.svd_target_len:
            pad = torch.zeros(self.svd_target_len - z.numel(), dtype=z.dtype)
            z = torch.cat([z, pad], dim=0)
        elif z.numel() > self.svd_target_len:
            z = z[:self.svd_target_len]

        return z

    # ------------------------------------------------------------------
    # Existing helpers you already had
    # ------------------------------------------------------------------

    def _extract_sliding_windows(self, hist: List[torch.Tensor]) -> List[torch.Tensor]:
        L = len(hist)
        T = self.window_len
        if L < T:
            return []
        windows: List[torch.Tensor] = []
        for start in range(0, L - T + 1):
            seq = torch.stack(hist[start:start + T], dim=0)  # [T,C,H,W]
            windows.append(seq)
        return windows

    def _build_frame_for_client(
        self,
        cid: int,
        v: torch.Tensor,          # [D] projected latent vector, not raw truncated delta
        idx: int,
        stats: dict,
        val_delta_i: float,
    ) -> torch.Tensor:
        H = self.feature_H
        W = self.feature_W

        mu_v = stats["mu_v"]
        sigma_v = stats["sigma_v"]
        mu_m = stats["mu_m"]
        sigma_m = stats["sigma_m"]

        norms = stats["norms"]
        mu_N = stats["mu_N"]
        sigma_N = stats["sigma_N"]

        cos = stats["cos"]
        mu_C = stats["mu_C"]
        sigma_C = stats["sigma_C"]

        mu_L = stats["mu_L"]
        sigma_L = stats["sigma_L"]

        x1 = (v - mu_v) / sigma_v
        x1 = x1.view(1, H, W)

        x2 = torch.sign(v).view(1, H, W)

        m = v.abs()
        x3 = (m - mu_m) / sigma_m
        x3 = x3.view(1, H, W)

        N_i = norms[idx]
        N_hat = (N_i - mu_N) / sigma_N
        x4 = torch.full((1, H, W), float(N_hat))

        C_i = cos[idx]
        C_hat = (C_i - mu_C) / sigma_C
        x5 = torch.full((1, H, W), float(C_hat))

        L_i = torch.tensor(float(val_delta_i), dtype=torch.float32)
        L_hat = (L_i - mu_L) / sigma_L
        x6 = torch.full((1, H, W), float(L_hat))

        F = torch.cat([x1, x2, x3, x4, x5, x6], dim=0)  # [C,H,W]
        return F

    def log_convlstm_total_cost(self) -> None:
        """
        Print total number of ConvLSTM forwards after training.
        """
        log(
            INFO,
            "[ConvLSTM TOTAL COST] total_convlstm_forwards = %d",
            self.total_convlstm_forwards,
        )

    # ------------------------------------------------------------------
    # Rewritten: SVD-based ConvLSTM analysis
    # ------------------------------------------------------------------

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
        thr = float(np.quantile(scores, float(np.clip(self.good_client_quantile, 0.0, 1.0))))

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


    def _run_convlstm_analysis(
        self,
        server_round: int,
        attacked_replies: List[Message],
        clients_state: Dict[int, bool],
        global_arrays: ArrayRecord,
        client_val_deltas: Dict[int, float],
    ) -> None:
        """
        Run anomaly analysis using a FIXED SVD basis.

        Instead of truncating/padding raw model-update vectors to 256,
        this method projects full raw update vectors into a fixed latent
        space of size self.svd_target_len.
        """
        target_len = self.feature_H * self.feature_W
        eps = 1e-6

        if not self.svd_ready:
            log(
                INFO,
                "[Round %s] Fixed SVD basis not ready -> skipping ConvLSTM analysis",
                server_round,
            )
            return

        gsd = global_arrays.to_torch_state_dict()
        selected_names = self._get_selected_layer_names(gsd)

        # 1) Build projected client vectors
        cids: List[int] = []
        projected_vecs: List[torch.Tensor] = []

        for msg in attacked_replies:
            cid = int(msg.metadata.src_node_id)
            csd = msg.content["arrays"].to_torch_state_dict()

            raw_v = self._collect_raw_update_vector(csd, gsd, selected_names)
            if raw_v is None:
                continue

            z = self._project_raw_vector_fixed_svd(raw_v)  # [target_len]
            cids.append(cid)
            projected_vecs.append(z)

        if not projected_vecs:
            return

        all_vecs = torch.stack(projected_vecs, dim=0)  # [N, target_len]
        N, D = all_vecs.shape

        # reset per-round counters
        self.round_n_clients = len(cids)
        self.round_n_scored = 0
        self.round_n_refresh = 0
        self.round_n_trigger_high_ewma = 0
        self.round_n_trigger_risk = 0   # kept as zero for compatibility
        self.round_n_trigger_drift = 0  # kept as zero for compatibility
        self.round_t_convlstm = 0.0

        # 2) Per-round stats in latent space
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






                # 4) EWMA-only adaptive ConvLSTM scoring
        lam = float(self.anom_decay)

        # =========================================================
        # EWMA-only scoring policy
        # - Cold-start: score all clients in first N rounds
        # - After cold-start:
        #     1) periodic refresh
        #     2) high-EWMA trigger
        #     3) optional hard cap
        # =========================================================
        score_set: set[int] = set()

        # ---- Phase 1: cold-start ----
        if server_round <= int(self.score_all_first_n_rounds):
            score_set = set(cids)

        # ---- Phase 2: EWMA-only selection after cold-start ----
        else:
            # 1) periodic refresh
            for cid in cids:
                last_r = int(self.last_scored_round.get(cid, 0))
                if (server_round - last_r) >= int(self.score_refresh_max):
                    score_set.add(cid)
                    self.round_n_refresh += 1

            # 2) high-EWMA trigger
            ewma_vals = np.array(
                [self.ewma_score[c] for c in cids if c in self.ewma_score],
                dtype=np.float32,
            )

            if ewma_vals.size > 0:
                tau_s = float(np.quantile(ewma_vals, float(np.clip(self.tau_s_quantile, 0.0, 1.0))))
            else:
                tau_s = float("inf")

            for cid in cids:
                prev_ewma = self.ewma_score.get(cid, None)
                if prev_ewma is not None and prev_ewma > tau_s:
                    if cid not in score_set:
                        self.round_n_trigger_high_ewma += 1
                    score_set.add(cid)

            # 3) adaptive cap driven by EWMA spread
            if ewma_vals.size >= 4:
                q25, q50, q75 = np.quantile(ewma_vals, [0.25, 0.50, 0.75])
                iqr = float(q75 - q25)
                risk = float(np.clip(iqr / (abs(float(q50)) + 1e-6), 0.0, 1.0))
                raw_cap = int(
                    self.adaptive_cap_min
                    + (self.adaptive_cap_max - self.adaptive_cap_min) * risk
                )
            else:
                raw_cap = self._adaptive_cap  # not enough data yet — keep previous

            self._adaptive_cap = int(
                self.adaptive_cap_smooth * self._adaptive_cap
                + (1.0 - self.adaptive_cap_smooth) * raw_cap
            )
            self.max_scores_per_round = self._adaptive_cap  # keep CSV logging in sync

            if len(score_set) > self._adaptive_cap:
                ranked = sorted(
                    list(score_set),
                    key=lambda c: self.ewma_score.get(c, 0.0),
                    reverse=True,
                )
                score_set = set(ranked[: self._adaptive_cap])

        # score selected clients
        for cid in cids:
            hist = self.client_feature_history.get(cid, [])
            if len(hist) < self.window_len:
                continue

            seq = torch.stack(hist[-self.window_len :], dim=0)  # [T,C,H,W]

            prev = self.ewma_score.get(cid, None)
            prev_prev = self.ewma_prev.get(cid, prev)

            raw_score: Optional[float] = None
            if cid in score_set:
                t0 = time.perf_counter()
                raw_score = float(self.detector.anomaly_score(seq))
                dt = time.perf_counter() - t0

                self.round_n_scored += 1
                self.round_t_convlstm += dt
                self.total_convlstm_forwards += 1
                self.total_t_convlstm += dt
                self.last_scored_round[cid] = int(server_round)

            if prev is None:
                if raw_score is None:
                    continue
                new_ewma = raw_score
                new_m2 = raw_score * raw_score
                self.ewma_prev[cid] = raw_score
            else:
                if prev_prev is None:
                    prev_prev = prev
                self.ewma_prev[cid] = prev

                if raw_score is None:
                    new_ewma = lam * prev
                    prev_m2 = self.ewma_m2.get(cid, prev * prev)
                    new_m2 = lam * prev_m2
                else:
                    new_ewma = lam * prev + (1.0 - lam) * raw_score
                    prev_m2 = self.ewma_m2.get(cid, prev * prev)
                    new_m2 = lam * prev_m2 + (1.0 - lam) * (raw_score * raw_score)

            self.ewma_score[cid] = float(new_ewma)
            self.ewma_m2[cid] = float(new_m2)

            mean_score = float(new_ewma)
            var_score = float(max(new_m2 - mean_score * mean_score, 0.0))

            is_malicious = bool(clients_state.get(cid, False))
            self._append_anomaly_csv(
                server_round=server_round,
                client_id=cid,
                mean_score=mean_score,
                var_score=var_score,
                is_malicious=is_malicious,
            )

            self.last_anomaly_scores[cid] = mean_score

        if self.tb_writer is not None:
            r = server_round
            self.tb_writer.add_scalar("cost/n_clients", self.round_n_clients, r)
            self.tb_writer.add_scalar("cost/n_scored", self.round_n_scored, r)
            self.tb_writer.add_scalar(
                "cost/ratio_scored",
                self.round_n_scored / max(self.round_n_clients, 1),
                r,
            )
            self.tb_writer.add_scalar("cost/t_convlstm", self.round_t_convlstm, r)
            self.tb_writer.add_scalar("cost/trigger_refresh", self.round_n_refresh, r)
            self.tb_writer.add_scalar("cost/trigger_risk", self.round_n_trigger_risk, r)
            self.tb_writer.add_scalar("cost/trigger_drift", self.round_n_trigger_drift, r)
            self.tb_writer.add_scalar("cost/trigger_high_ewma", self.round_n_trigger_high_ewma, r)

        self._append_cost_csv(server_round)

    # ------------------------------------------------------------------
    # start(): only the important SVD-related change is shown here
    # Keep the rest of your method the same.
    # ------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 20,
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

        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            self.current_round_for_tb = current_round
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

            # ---- NEW: bootstrap fixed SVD basis during warmup / early clean rounds ----
            if not self.svd_ready:
                self._maybe_fit_svd_from_warmup_replies(
                    train_replies=ordered_train_replies,
                    global_arrays=self.round_global_arrays,
                )

            # Choose attack
            if self.attack_mode == "no_attack":
                attacked_replies = no_attack_messages(ordered_train_replies, clients_state)
            elif self.attack_mode == "gaussian":
                attacked_replies = gaussian_attack_messages(
                    ordered_train_replies, clients_state, magnitude=3,
                )
            elif self.attack_mode == "lie":
                attacked_replies = lie_attack_messages(
                    ordered_train_replies, clients_state, omniscent=True,
                    decorrelate_cos=getattr(self, "decorrelate_cos", None),
                )
            elif self.attack_mode == "min_max":
                attacked_replies = minmax_attack_messages(
                    ordered_train_replies, clients_state, omniscent=True,
                    threshold=1e-5, lambda_init=5.0, malicious_num=len(malicious_cids),
                    decorrelate_cos=getattr(self, "decorrelate_cos", None),
                )
            elif self.attack_mode == "fang":
                w_re = {k: v.cpu().numpy() for k, v in arrays.to_torch_state_dict().items()}
                attacked_replies = fang_attack_messages(
                    ordered_train_replies, clients_state, w_re=w_re,
                )
            else:
                raise ValueError(f"Unknown attack_mode: {self.attack_mode}")

            client_val_deltas: Dict[int, float] = {}
            for msg in attacked_replies:
                cid = int(msg.metadata.src_node_id)
                m = msg.content.get("metrics", {}) if isinstance(msg.content, dict) else {}
                dL = float(m.get("val_loss_delta", 0.0))
                client_val_deltas[cid] = dL

            self._run_convlstm_analysis(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                global_arrays=self.round_global_arrays,
                client_val_deltas=client_val_deltas,
            )

            train_cids = [int(msg.metadata.src_node_id) for msg in attacked_replies]

            adaptive_q_train = self._get_adaptive_filter_quantile(
                server_round=current_round,
                cids=train_cids,
            )

            filtered_replies, detected_state = self._filter_attacked_replies_by_anomaly(
                server_round=current_round,
                attacked_replies=attacked_replies,
                clients_state=clients_state,
                quantile=adaptive_q_train,
                min_clients=2,
                abs_threshold=None,
            )

            drop_rate = 1.0
            if len(attacked_replies) > 0:
                drop_rate = 1.0 - (len(filtered_replies) / len(attacked_replies))
            self.last_drop_rate = float(drop_rate)


            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
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

            eval_cids = [int(msg.metadata.src_node_id) for msg in evaluate_replies]

            adaptive_q_eval = self._get_adaptive_filter_quantile(
                server_round=current_round,
                cids=eval_cids,
            )

            filtered_eval_replies = self._filter_evaluate_replies_by_anomaly(
                server_round=current_round,
                evaluate_replies=evaluate_replies,
                quantile=adaptive_q_eval,
                abs_threshold=None,
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

                    round_is_good = self._round_is_good(acc_val, loss_val)
                    skip_update = (drop_rate > 0.4)

                    if round_is_good:
                        self._collect_clean_windows_for_training(
                            filtered_replies,
                            max_windows_per_client=3,
                        )

                    self._maybe_update_detector(
                        current_round=current_round,
                        round_is_good=round_is_good,
                        skip_update=skip_update,
                    )

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

        self.log_convlstm_total_cost()
        return result