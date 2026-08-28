"""server_app_sweep.py  –  dataset × tau_s × attack sweep for CustomFedAvg_adaptive
==========================================================================
Runs ALL (dataset, attack_mode, tau_s_quantile) combinations inside a single
`flwr run .` execution.  The Flower Grid stays open throughout; a fresh
strategy + model is created for every combination.

HOW TO USE
──────────
1.  Rename / copy this file over server_app.py:
        cp server_sweep.py server_app.py

2.  Edit the SWEEP GRID section below to choose datasets, attacks, tau
    values, and any other hyperparameters.

3.  Run normally:
        flwr run . local-sim-gpu

Results are written to:
    outputs/<date>/<time>_<dataset>_<attack>_tau<q>/   per-run outputs (CSV, JSON …)
    outputs/sweep_summary_<timestamp>.csv              one row per (dataset, attack, tau) run,
                                                        appended incrementally as each run finishes
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader

from rl_flwr.task import (
    DATASET_REGISTRY,
    get_net,
    build_transforms,
    make_eval_transform_fn,
    create_run_dir,
    test,
)
from rl_flwr.CustomFedAvg_adaptive_new import CustomFedAvg_adaptive_new

app = ServerApp()

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP GRID  –  edit here
# ══════════════════════════════════════════════════════════════════════════════
DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]
TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]
ATTACK_MODES  = ["gaussian", "lie", "min_max"]

# Fixed hyperparameters shared across every run
ANOM_DECAY      = 0.6
NUM_MALICIOUS   = 40
WARMUP_ROUNDS   = 5
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_FIELDNAMES = [
    "dataset", "attack_mode", "tau_s_quantile", "final_accuracy",
    "total_convlstm_forwards", "mean_ratio_scored", "total_convlstm_time_ms",
    "save_path", "error",
]


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the full (dataset × attack × tau) sweep within one flwr session."""

    num_rounds     = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval  = context.run_config["fraction-evaluate"]
    device         = context.run_config["server-device"]

    total_runs = len(DATASET_MODES) * len(ATTACK_MODES) * len(TAU_VALUES)
    run_idx    = 0

    summary_path = _make_summary_path()

    for dataset_name in DATASET_MODES:
        for attack_mode in ATTACK_MODES:
            for tau in TAU_VALUES:
                run_idx += 1
                tag = f"{dataset_name}_{attack_mode}_tau{tau:.2f}"
                print(
                    f"\n{'='*65}\n"
                    f"[SWEEP {run_idx}/{total_runs}]  dataset={dataset_name}  "
                    f"attack={attack_mode}  tau_s={tau:.2f}\n"
                    f"{'='*65}\n"
                )

                try:
                    row = _run_one_combination(
                        grid=grid,
                        context=context,
                        dataset_name=dataset_name,
                        attack_mode=attack_mode,
                        tau=tau,
                        tag=tag,
                        num_rounds=num_rounds,
                        fraction_train=fraction_train,
                        fraction_eval=fraction_eval,
                        device=device,
                    )
                except Exception as exc:
                    print(f"[SWEEP {run_idx}/{total_runs}]  FAILED  {exc}")
                    row = _error_row(dataset_name, attack_mode, tau, exc)

                _append_summary_csv(row, summary_path)

                acc = row["final_accuracy"]
                acc_str = f"{acc:.4f}" if isinstance(acc, float) and not math.isnan(acc) else "NaN"
                print(
                    f"[SWEEP {run_idx}/{total_runs}]  DONE  "
                    f"acc={acc_str}  "
                    f"forwards={row['total_convlstm_forwards']}\n"
                )

    print(f"\n[SWEEP] Summary written → {summary_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
def _run_one_combination(
    grid: Grid,
    context: Context,
    dataset_name: str,
    attack_mode: str,
    tau: float,
    tag: str,
    num_rounds: int,
    fraction_train: float,
    fraction_eval: float,
    device: str,
) -> dict:
    """Run a single (dataset, attack, tau) combination and return its summary row."""

    # ── Fresh global model for every run ──────────────────────────
    global_model = get_net(dataset_name)
    arrays = ArrayRecord(global_model.state_dict())

    # ── Fresh strategy for every run ──────────────────────────────
    strategy = CustomFedAvg_adaptive_new(
        num_malicious=NUM_MALICIOUS,
        warmup_rounds=WARMUP_ROUNDS,
        attack_mode=attack_mode,
        anom_decay=ANOM_DECAY,
        tau_s_quantile=tau,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
    )

    # ── Per-run output directory ──────────────────────────────────
    save_path, run_dir = create_run_dir(config=context.run_config)
    named_path = save_path.parent / f"{save_path.name}_{tag}"
    save_path.rename(named_path)
    save_path = named_path
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # ── Run FL training ───────────────────────────────────────────
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.001, "dataset": dataset_name}),
        evaluate_config=ConfigRecord({"dataset": dataset_name}),
        num_rounds=num_rounds,
        evaluate_fn=_make_evaluate_fn(device, dataset_name),
    )

    # ── Collect metrics for summary ───────────────────────────────
    return _collect_metrics(
        save_path=save_path,
        attack_mode=attack_mode,
        tau=tau,
        strategy=strategy,
        dataset_name=dataset_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
def _collect_metrics(save_path: Path, attack_mode: str,
                     tau: float, strategy, dataset_name: str) -> dict:
    """Read per-run CSVs / JSON and return a summary dict."""

    # Final accuracy from results.json
    final_accuracy = float("nan")
    results_file = save_path / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        # results.json stores nested server/client metrics per round.
        # Older runs may still have top-level accuracy fields.
        if isinstance(data, list) and data:
            final_accuracy = _extract_final_accuracy(data[-1])
        elif isinstance(data, dict) and data:
            last_key = str(max(int(k) for k in data.keys()))
            final_accuracy = _extract_final_accuracy(data[last_key])

    # ConvLSTM cost metrics from convlstm_cost.csv
    total_scored  = int(strategy.total_convlstm_forwards)
    mean_ratio    = float("nan")
    total_time_ms = float("nan")

    cost_csv = save_path / "convlstm_cost.csv"
    if cost_csv.exists():
        import pandas as pd
        df = pd.read_csv(cost_csv)
        mean_ratio    = float(df["ratio_scored"].mean())
        total_time_ms = float(df["t_convlstm"].sum() * 1000)

    return {
        "dataset":                 dataset_name,
        "attack_mode":             attack_mode,
        "tau_s_quantile":          tau,
        "final_accuracy":          round(final_accuracy, 4),
        "total_convlstm_forwards": total_scored,
        "mean_ratio_scored":       round(mean_ratio, 4) if not _isnan(mean_ratio) else mean_ratio,
        "total_convlstm_time_ms":  round(total_time_ms, 2) if not _isnan(total_time_ms) else total_time_ms,
        "save_path":               str(save_path),
        "error":                   "",
    }


def _extract_final_accuracy(entry: dict) -> float:
    """Extract final accuracy from current and legacy results.json shapes."""
    if not isinstance(entry, dict):
        return float("nan")

    server_metrics = entry.get("evaluate_metrics_serverapp") or {}
    if "accuracy" in server_metrics:
        return float(server_metrics["accuracy"])

    client_metrics = entry.get("evaluate_metrics_clientapp") or {}
    if "eval_acc" in client_metrics:
        return float(client_metrics["eval_acc"])

    return float(entry.get("accuracy", entry.get("eval_accuracy", float("nan"))))


def _error_row(dataset_name: str, attack_mode: str, tau: float, exc: Exception) -> dict:
    """Build a summary row for a combination that raised an exception."""
    return {
        "dataset":                 dataset_name,
        "attack_mode":             attack_mode,
        "tau_s_quantile":          tau,
        "final_accuracy":          float("nan"),
        "total_convlstm_forwards": float("nan"),
        "mean_ratio_scored":       float("nan"),
        "total_convlstm_time_ms":  float("nan"),
        "save_path":               "",
        "error":                   str(exc)[:200],
    }


def _isnan(x):
    try:
        import math
        return math.isnan(x)
    except Exception:
        return False


def _make_summary_path() -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"sweep_summary_{ts}.csv"


def _append_summary_csv(row: dict, summary_path: Path):
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
def _make_evaluate_fn(device: str, dataset_name: str):
    """Server-side global evaluation function for a specific dataset."""

    cfg = DATASET_REGISTRY[dataset_name]
    _, eval_transform = build_transforms(dataset_name)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        global_test_set = load_dataset(cfg["hf_path"])["test"]
        if cfg["image_key"] != "image":
            global_test_set = global_test_set.rename_column(cfg["image_key"], "image")
        testloader = DataLoader(
            global_test_set.with_transform(make_eval_transform_fn(eval_transform)),
            batch_size=32,
        )
        net = get_net(dataset_name)
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
