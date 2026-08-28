# Dataset × Attack × Tau Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `rl-flwr`'s existing attack×tau sweep (`server_sweep.py`) with a third dimension — dataset (`mnist`, `fashion_mnist`, `cifar10`) — so one `flwr run .` session produces 99 `(dataset, attack_mode, tau_s_quantile)` results in a single combined CSV, with correct per-dataset model/transform handling and resilience to per-run failures.

**Architecture:** A new `DATASET_REGISTRY` in `task.py` maps each dataset name to its HF path, batch image key, channel count, image size, and normalization stats; `get_net()`/`build_transforms()` select the right model/transforms per dataset; `load_data()` normalizes CIFAR-10's `"img"` column to `"image"` immediately after loading so downstream code never needs to branch on dataset. `server_sweep.py` gains a third outer loop and threads `dataset_name` to clients via both `train_config` and `evaluate_config`. `plot_sweep_results.py` facets its figures by dataset.

**Tech Stack:** Python 3.10, PyTorch 2.7, Flower (flwr) ≥1.23 simulation engine, flwr-datasets, pandas, matplotlib. No test framework exists in this codebase today (no `tests/`, no pytest config) — see Global Constraints.

## Global Constraints

- This directory is **not a git repository** — there are no commit steps in this plan; each task ends at its verification step instead.
- **No pytest / test framework exists in this codebase** and none is being introduced. Verification is via manual smoke-test runs against the real Flower simulation and real HF datasets — this is deliberate (see design doc's "Testing / validation" section): the bugs that matter here (channel-count mismatches, batch-key mismatches, stale dataset caching) only show up at the integration level, not in isolated unit tests.
- Source design doc: `docs/superpowers/specs/2026-06-25-dataset-sweep-design.md` — read it for the full rationale; this plan implements it exactly.
- Keep all other sweep hyperparameters (`anom_decay=0.6`, `num_malicious=40`, `warmup_rounds=5`) constant across datasets — dataset is an independent axis, all else equal.
- Do not modify `server_app.py` (the non-sweep entry point), `server_backup.py`, `FLTrust_pytorch/`, or `build_fltrust_rootset_loader()` in `task.py` — out of scope per the design doc's non-goals. `client_app.py` IS shared by both `server_app.py` and the sweep, so its changes must default `dataset` to `"fashion_mnist"` when no config is sent (preserving `server_app.py`'s current behavior unchanged).

---

### Task 1: Dataset/model registry in `task.py`

**Files:**
- Modify: `rl_flwr/task.py:33-55` (commented-out CIFAR block → active `NetCIFAR`)
- Modify: `rl_flwr/task.py:57-67` (add `DATASET_REGISTRY`, `get_net`, `build_transforms` after the existing `FM_NORMALIZATION`/`EVAL_TRANSFORMS`/`TRAIN_TRANSFORMS` block — those stay, untouched, for `build_fltrust_rootset_loader`)
- Modify: `rl_flwr/task.py:128-171` (add transform-closure factories; rewrite `load_data`)

**Interfaces:**
- Produces: `get_net(dataset_name: str) -> nn.Module`, `build_transforms(dataset_name: str) -> tuple[Compose, Compose]` (train, eval), `make_train_transform_fn(transform) -> Callable[[dict], dict]`, `make_eval_transform_fn(transform) -> Callable[[dict], dict]`, `DATASET_REGISTRY: dict`, `NetCIFAR` (class), `load_data(partition_id: int, num_partitions: int, dataset_name: str = "fashion_mnist") -> tuple[DataLoader, DataLoader]`.
- Consumes: nothing from other tasks (this is the foundation task).

- [ ] **Step 1: Replace the commented-out CIFAR block with an active `NetCIFAR` class**

Find this block at `rl_flwr/task.py:33-55`:

```python
# CIFAR-10 configuration
# TRAIN_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# EVAL_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# #MNIST and Fashion-MNIST configuration
```

Replace it with:

```python
class NetCIFAR(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz') for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# MNIST and Fashion-MNIST configuration
```

Note: the flatten dimension `16 * 5 * 5` is correct for 32×32 CIFAR-10 input: `conv1` (kernel 5, no padding) takes 32→28, `pool` takes 28→14, `conv2` takes 14→10, `pool` takes 10→5, so `16 channels × 5 × 5 = 400` matches `fc1 = nn.Linear(16 * 5 * 5, 120)`. This is verified in Step 6's smoke test (a shape mismatch here is a hard crash, not a silent bug).

- [ ] **Step 2: Add `DATASET_REGISTRY`, `get_net`, `build_transforms` after the existing transform constants**

The existing block at `rl_flwr/task.py:57-67` (`FM_NORMALIZATION`, `EVAL_TRANSFORMS`, `TRAIN_TRANSFORMS`) stays exactly as-is — it's still used by `build_fltrust_rootset_loader` (out of scope). Immediately after that block (i.e. after line 67, before `class Net(nn.Module):`), insert:

```python
DATASET_REGISTRY = {
    "mnist": {
        "hf_path": "ylecun/mnist",
        "image_key": "image",
        "channels": 1,
        "image_size": 28,
        "normalization": ((0.1307,), (0.3081,)),
    },
    "fashion_mnist": {
        "hf_path": "zalando-datasets/fashion_mnist",
        "image_key": "image",
        "channels": 1,
        "image_size": 28,
        "normalization": ((0.1307,), (0.3081,)),
    },
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
        "image_key": "img",
        "channels": 3,
        "image_size": 32,
        "normalization": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
}


def get_net(dataset_name: str) -> nn.Module:
    """Return a fresh model instance matching the dataset's input shape."""
    if dataset_name == "cifar10":
        return NetCIFAR()
    return Net()


def build_transforms(dataset_name: str):
    """Build (train_transform, eval_transform) for the given dataset."""
    cfg = DATASET_REGISTRY[dataset_name]
    normalization = cfg["normalization"]
    image_size = cfg["image_size"]

    eval_transform = Compose([ToTensor(), Normalize(*normalization)])
    train_transform = Compose(
        [
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(*normalization),
        ]
    )
    return train_transform, eval_transform
```

`get_net` references `Net`, which is defined right after this block — that's fine, Python resolves the name at call time, not at module-load time, so the forward reference is safe.

- [ ] **Step 3: Run a quick shape sanity check (catches a `NetCIFAR` flatten-dim mistake in seconds instead of mid-sweep)**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "
import torch
from rl_flwr.task import get_net, DATASET_REGISTRY

for name, cfg in DATASET_REGISTRY.items():
    net = get_net(name)
    x = torch.randn(2, cfg['channels'], cfg['image_size'], cfg['image_size'])
    out = net(x)
    print(name, '->', tuple(out.shape))
"
```
Expected output (exact shapes, order may vary by dict iteration but Python 3.7+ dicts preserve insertion order so this should match):
```
mnist -> (2, 10)
fashion_mnist -> (2, 10)
cifar10 -> (2, 10)
```
If `cifar10` raises a `RuntimeError` about a shape mismatch in `fc1`, the flatten-dim note in Step 1 was violated — re-check the `Conv2d`/`MaxPool2d` parameters were copied exactly.

- [ ] **Step 4: Add transform-closure factories and rewrite `load_data`**

Find the existing `apply_train_transforms`/`apply_eval_transforms` functions at `rl_flwr/task.py:128-139` — **leave them exactly as they are** (still used by `build_fltrust_rootset_loader`, out of scope). Immediately after them (after line 139, before `fds: FederatedDataset | None = None`), add:

```python
def make_train_transform_fn(transform):
    """Build a `.with_transform()`-compatible fn for a given train transform."""
    def apply(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    return apply


def make_eval_transform_fn(transform):
    """Build a `.with_transform()`-compatible fn for a given eval transform."""
    def apply(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    return apply
```

Then find the existing `fds`/`load_data` block at `rl_flwr/task.py:142-171`:

```python
fds: FederatedDataset | None = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            #dataset="uoft-cs/cifar10",
            # dataset="ylecun/mnist",
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader
```

Replace it with (note: `fds` global stays, untouched, for `build_fltrust_rootset_loader`; this adds a **new**, separate `_fds_cache` dict for the sweep path):

```python
fds: FederatedDataset | None = None
_fds_cache: Dict[str, FederatedDataset] = {}

def load_data(partition_id: int, num_partitions: int, dataset_name: str = "fashion_mnist"):
    """Load a client partition for the given dataset."""
    cfg = DATASET_REGISTRY[dataset_name]

    if dataset_name not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        _fds_cache[dataset_name] = FederatedDataset(
            dataset=cfg["hf_path"],
            partitioners={"train": partitioner},
        )
    fds_for_dataset = _fds_cache[dataset_name]

    partition = fds_for_dataset.load_partition(partition_id)
    if cfg["image_key"] != "image":
        partition = partition.rename_column(cfg["image_key"], "image")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_transform, eval_transform = build_transforms(dataset_name)
    train_partition = partition_train_test["train"].with_transform(
        make_train_transform_fn(train_transform)
    )
    test_partition = partition_train_test["test"].with_transform(
        make_eval_transform_fn(eval_transform)
    )
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader
```

`Dict` is already imported at the top of `task.py` (`from typing import Deque, Dict, List, Tuple, Optional, Iterable`) — no new import needed.

- [ ] **Step 5: Verify `task.py` still imports cleanly**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.task; print('ok')"
```
Expected output: `ok` (no `SyntaxError`/`ImportError`/`NameError`).

---

### Task 2: Wire dataset selection into `client_app.py`

**Files:**
- Modify: `rl_flwr/client_app.py:1-9` (imports)
- Modify: `rl_flwr/client_app.py:15-63` (`train()` handler)
- Modify: `rl_flwr/client_app.py:66-100` (`evaluate()` handler)

**Interfaces:**
- Consumes: `get_net(dataset_name: str) -> nn.Module` and `load_data(partition_id, num_partitions, dataset_name=...)` from Task 1.
- Produces: nothing new for downstream tasks — this task only makes the existing client handlers dataset-aware.

- [ ] **Step 1: Update the import**

Find:
```python
from rl_flwr.task import Net, load_data
from rl_flwr.task import test as test_fn
from rl_flwr.task import train as train_fn
```
Replace with:
```python
from rl_flwr.task import get_net, load_data
from rl_flwr.task import test as test_fn
from rl_flwr.task import train as train_fn
```

- [ ] **Step 2: Make `train()` dataset-aware**

Find (`rl_flwr/client_app.py:15-28`):
```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
```
Replace with:
```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)
```

- [ ] **Step 3: Make `evaluate()` dataset-aware**

Find (`rl_flwr/client_app.py:66-79`):
```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)
```
Replace with:
```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)
```

`.get("dataset", "fashion_mnist")` is required (not `["dataset"]`) because `server_app.py` (out of scope, untouched) never sends a `dataset` config key — this default keeps its existing single-run behavior working unchanged.

- [ ] **Step 4: Verify `client_app.py` still imports cleanly**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.client_app; print('ok')"
```
Expected output: `ok`.

Full end-to-end verification of this task happens in Task 4's smoke test (it needs a running Flower simulation, which this isolated import check can't exercise).

---

### Task 3: Three-dimensional sweep with resilience in `server_sweep.py`

**Files:**
- Modify: `rl_flwr/server_sweep.py` (entire sweep grid, loop body, metrics collection, and summary-writing — see exact replacements below)

**Interfaces:**
- Consumes: `DATASET_REGISTRY`, `get_net`, `build_transforms`, `make_eval_transform_fn`, `create_run_dir`, `test` from `rl_flwr.task` (Task 1); `CustomFedAvg_adaptive` from `rl_flwr.strategy` (unchanged).
- Produces: `outputs/sweep_summary_<timestamp>.csv` with columns `dataset, attack_mode, tau_s_quantile, final_accuracy, total_convlstm_forwards, mean_ratio_scored, total_convlstm_time_ms, save_path, error` — Task 6 (plotting) consumes this schema.

**Important Flower detail verified against `rl_flwr/strategy.py` (`CustomFedAvg_adaptive.start()`, ~line 3519):** `train_config` and `evaluate_config` are two **separate** parameters — `train_config` only reaches `configure_train` (TRAIN-round messages), `evaluate_config` only reaches `configure_evaluate` (EVALUATE-round messages). `dataset_name` must be passed in **both**, or the client's `evaluate()` handler (Task 2) won't see it.

- [ ] **Step 1: Replace the whole file**

Replace the full contents of `rl_flwr/server_sweep.py` with:

```python
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
        flwr run .

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
from rl_flwr.strategy import CustomFedAvg_adaptive

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
    strategy = CustomFedAvg_adaptive(
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
        # results.json stores a list of {"round": n, "accuracy": x, ...}
        # or a dict keyed by round number – handle both
        if isinstance(data, list) and data:
            last = data[-1]
            final_accuracy = float(
                last.get("accuracy", last.get("eval_accuracy", float("nan")))
            )
        elif isinstance(data, dict):
            last_key = str(max(int(k) for k in data.keys()))
            final_accuracy = float(
                data[last_key].get("accuracy", float("nan"))
            )

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
```

- [ ] **Step 2: Verify `server_sweep.py` still imports cleanly**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.server_sweep; print('ok')"
```
Expected output: `ok`.

Full end-to-end verification (does the loop actually run, does the CSV get the right rows, does resilience work) happens in Tasks 4 and 5 — this needs a real Flower simulation, which an import check can't exercise.

---

### Task 4: Smoke test — all 3 datasets, 1 attack, 1 tau, 2 rounds

This is the highest-value verification step in this plan: it exercises the exact integration points most likely to break (CIFAR's 3-channel/32×32 shape, the `"img"`→`"image"` rename, the per-dataset `FederatedDataset` cache) against real data, in minutes instead of after hours of the full 99-run sweep. Do this **before** Task 6 (plotting) — there's no point polishing figures for data you haven't confirmed is correct.

**Files (temporary edits, reverted at the end of this task):**
- Modify temporarily: `pyproject.toml` (`num-server-rounds`)
- Modify temporarily: `rl_flwr/server_sweep.py` (`ATTACK_MODES`, `TAU_VALUES`)
- Copy temporarily: `rl_flwr/server_app.py` → overwritten by the sweep, restored after

- [ ] **Step 1: Back up the current `server_app.py`**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py rl_flwr/server_app.py.smoketest-bak
```
This file is the normal non-sweep single-run entry point — back it up before overwriting it with the sweep version, since there's no git here to recover it otherwise.

- [ ] **Step 2: Shrink the grids for a fast smoke test**

In `pyproject.toml`, change:
```toml
num-server-rounds = 50
```
to:
```toml
num-server-rounds = 2
```

In `rl_flwr/server_sweep.py`, change:
```python
DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]
TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]
ATTACK_MODES  = ["gaussian", "lie", "min_max"]
```
to:
```python
DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]
TAU_VALUES    = [0.5]
ATTACK_MODES  = ["gaussian"]
```
(Keep `DATASET_MODES` at all three — that's the whole point of this smoke test. Shrinking `TAU_VALUES`/`ATTACK_MODES` to one value each keeps it to 3 short runs instead of 99 long ones.)

- [ ] **Step 3: Activate the sweep and run it**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_sweep.py rl_flwr/server_app.py && flwr run .
```
Expected: the console prints `[SWEEP 1/3] dataset=mnist ...`, `[SWEEP 2/3] dataset=fashion_mnist ...`, `[SWEEP 3/3] dataset=cifar10 ...` in sequence, each followed by 2 rounds of training/eval logs, then a `[SWEEP n/3] DONE acc=<number> forwards=<number>` line for each, with no unhandled exception / traceback. The run ends with `[SWEEP] Summary written → .../outputs/sweep_summary_<timestamp>.csv`.

If `cifar10` crashes with a `KeyError` mentioning `"img"`, the column-rename step in Task 1 Step 4 wasn't applied correctly. If it crashes with a shape mismatch in `fc1`, re-check Task 1 Step 1's `NetCIFAR` definition.

- [ ] **Step 4: Inspect the resulting CSV**

Run (substitute the actual timestamp printed in Step 3's last line):
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cat outputs/sweep_summary_<timestamp>.csv
```
Expected: a header row `dataset,attack_mode,tau_s_quantile,final_accuracy,total_convlstm_forwards,mean_ratio_scored,total_convlstm_time_ms,save_path,error` followed by exactly 3 data rows — one each for `mnist`, `fashion_mnist`, `cifar10` — with `attack_mode=gaussian`, `tau_s_quantile=0.5`, a `final_accuracy` between 0 and 1 (not NaN, not blank), and an empty `error` field.

- [ ] **Step 5: Restore the backed-up `server_app.py` and revert the grid shrink**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py.smoketest-bak rl_flwr/server_app.py && rm rl_flwr/server_app.py.smoketest-bak
```
In `pyproject.toml`, revert:
```toml
num-server-rounds = 2
```
back to:
```toml
num-server-rounds = 50
```
In `rl_flwr/server_sweep.py`, revert:
```python
TAU_VALUES    = [0.5]
ATTACK_MODES  = ["gaussian"]
```
back to:
```python
TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]
ATTACK_MODES  = ["gaussian", "lie", "min_max"]
```

Keep the `outputs/sweep_summary_<timestamp>.csv` and its per-run output directories from this smoke test — Task 6 reuses this exact CSV to verify the plotting changes.

---

### Task 5: Resilience check — confirm one bad run doesn't sink the sweep

**Files:**
- Modify temporarily: `rl_flwr/task.py` (`DATASET_REGISTRY["cifar10"]["hf_path"]`)

- [ ] **Step 1: Deliberately break the CIFAR-10 dataset path**

In `rl_flwr/task.py`, in the `DATASET_REGISTRY` added in Task 1 Step 2, change:
```python
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
```
to:
```python
    "cifar10": {
        "hf_path": "uoft-cs/cifar10-does-not-exist",
```

- [ ] **Step 2: Repeat the smoke-test run from Task 4 (Steps 1-3) with this break in place**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py rl_flwr/server_app.py.smoketest-bak && cp rl_flwr/server_sweep.py rl_flwr/server_app.py && flwr run .
```
(Grids should still be shrunk from Task 4 Step 2 — if you already reverted them, redo that shrink first.)

Expected: `[SWEEP 1/3] dataset=mnist ...DONE` and `[SWEEP 2/3] dataset=fashion_mnist ...DONE` complete normally; `[SWEEP 3/3] dataset=cifar10 ...` prints a `FAILED` line with an exception message instead of crashing the whole process, and the sweep still reaches `[SWEEP] Summary written → ...` at the end (not an unhandled traceback that kills the run).

- [ ] **Step 3: Inspect the CSV for the NaN/error row**

Run (substitute the new timestamp):
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cat outputs/sweep_summary_<new-timestamp>.csv
```
Expected: 3 rows total. The `mnist` and `fashion_mnist` rows have real `final_accuracy` values and empty `error`. The `cifar10` row has `final_accuracy=nan`, `total_convlstm_forwards=nan`, `mean_ratio_scored=nan`, `total_convlstm_time_ms=nan`, empty `save_path`, and a non-empty `error` field containing a message about the dataset not being found.

- [ ] **Step 4: Revert the deliberate break and restore `server_app.py`**

In `rl_flwr/task.py`, change back:
```python
    "cifar10": {
        "hf_path": "uoft-cs/cifar10-does-not-exist",
```
to:
```python
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
```

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py.smoketest-bak rl_flwr/server_app.py && rm rl_flwr/server_app.py.smoketest-bak
```

If grids are still shrunk from Task 4, revert them now using Task 4 Step 5's instructions (skip if already reverted).

---

### Task 6: Per-dataset figure faceting in `plot_sweep_results.py`

**Files:**
- Modify: `plot_sweep_results.py:24-28` (path constants)
- Modify: `plot_sweep_results.py:61-68` (`load_data`)
- Modify: `plot_sweep_results.py:312-326` (`__main__` block)

**Interfaces:**
- Consumes: the `outputs/sweep_summary_<timestamp>.csv` schema produced by Task 3 (`dataset` column added). Specifically reuses the CSV already produced by Task 4's smoke test (3 rows: mnist, fashion_mnist, cifar10) to verify this task without re-running a sweep.

- [ ] **Step 1: Auto-resolve the summary CSV instead of a hardcoded timestamp**

Find (`plot_sweep_results.py:24-28`):
```python
# ── Paths ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV  = Path("/home/tuan/get-local-model-bt-id/rl-flwr/outputs/sweep_summary_20260507_083113.csv")
OUT_DIR      = Path("/home/tuan/get-local-model-bt-id/rl-flwr/") / "sweep_figures"
SUMMARY_JSON = OUT_DIR / "sweep_summary.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```
Replace with:
```python
# ── Paths ─────────────────────────────────────────────────────────────────────
def _find_latest_summary_csv() -> Path:
    candidates = sorted(
        Path("/home/tuan/get-local-model-bt-id/rl-flwr/outputs").glob("sweep_summary_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No outputs/sweep_summary_*.csv found. Run server_sweep.py first."
        )
    return candidates[-1]


SUMMARY_CSV  = _find_latest_summary_csv()
BASE_OUT_DIR = Path("/home/tuan/get-local-model-bt-id/rl-flwr/") / "sweep_figures"
OUT_DIR      = BASE_OUT_DIR        # reassigned per-dataset in __main__ below
SUMMARY_JSON = OUT_DIR / "sweep_summary.json"   # reassigned per-dataset in __main__ below
```

- [ ] **Step 2: Sort by dataset too, and fix the stale error-message reference**

Find (`plot_sweep_results.py:61-68`):
```python
def load_data() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run sweep_tau_attack.py first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(["attack_mode", "tau_s_quantile"]).reset_index(drop=True)
    return df
```
Replace with:
```python
def load_data() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run server_sweep.py first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(["dataset", "attack_mode", "tau_s_quantile"]).reset_index(drop=True)
    return df
```

- [ ] **Step 3: Facet the `__main__` block by dataset**

Find (`plot_sweep_results.py:312-326`):
```python
if __name__ == "__main__":
    df = load_data()
    print_summary_table(df)

    print("\nExporting JSON …")
    export_summary_to_json(df)

    print("\nGenerating figures …")
    plot_accuracy_vs_tau(df)
    plot_forwards_vs_tau(df)
    plot_combined_per_attack(df)
    plot_ratio_scored_vs_tau(df)
    plot_q_vs_acc_and_forwards()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
```
Replace with:
```python
if __name__ == "__main__":
    df_all = load_data()

    for dataset_name, df in df_all.groupby("dataset"):
        print(f"\n{'='*70}\n  Dataset: {dataset_name}\n{'='*70}")

        OUT_DIR = BASE_OUT_DIR / str(dataset_name)
        SUMMARY_JSON = OUT_DIR / "sweep_summary.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        print_summary_table(df)

        print("\nExporting JSON …")
        export_summary_to_json(df)

        print("\nGenerating figures …")
        plot_accuracy_vs_tau(df)
        plot_forwards_vs_tau(df)
        plot_combined_per_attack(df)
        plot_ratio_scored_vs_tau(df)
        plot_q_vs_acc_and_forwards()

        print(f"\nAll figures for {dataset_name} saved to: {OUT_DIR.resolve()}")
```
This works without touching `_save()`, `export_summary_to_json()`, or `plot_q_vs_acc_and_forwards()` — they all read `OUT_DIR`/`SUMMARY_JSON` as module-level globals at call time, and this block reassigns those globals before calling them on each dataset's filtered slice (this reassignment happens at module scope inside `if __name__ == "__main__":`, not inside a function body, so no `global` keyword is needed).

- [ ] **Step 4: Run it against the smoke-test CSV from Task 4 and verify per-dataset output**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python plot_sweep_results.py
```
Expected: prints a `Dataset: cifar10` / `Dataset: fashion_mnist` / `Dataset: mnist` section header (alphabetical groupby order) for each, each with its own summary table and "Saved →" lines, ending in 3× `All figures for <dataset> saved to: .../sweep_figures/<dataset>`.

Then run:
```bash
ls sweep_figures/mnist/ sweep_figures/fashion_mnist/ sweep_figures/cifar10/
```
Expected: each of the 3 directories contains `sweep_accuracy_vs_tau.pdf`, `sweep_accuracy_vs_tau.png`, `sweep_forwards_vs_tau.{pdf,png}`, `sweep_combined_gaussian.{pdf,png}` (only `gaussian` since that's all the smoke-test CSV contains), `sweep_ratio_scored_vs_tau.{pdf,png}`, `sweep_q_vs_acc_and_forwards.{pdf,png}`, and `sweep_summary.json`.

---

## After This Plan

The codebase is now verified to handle all 3 datasets correctly on a small grid. The full 99-combination sweep (`DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]` × full `ATTACK_MODES` × full `TAU_VALUES`, `num-server-rounds = 50`) is a long-running, resource-intensive operation — running it is a deliberate decision for you to make when ready, not an automated step of this plan. To run it: `cp rl_flwr/server_sweep.py rl_flwr/server_app.py && flwr run .`, then `python plot_sweep_results.py` once it finishes.
