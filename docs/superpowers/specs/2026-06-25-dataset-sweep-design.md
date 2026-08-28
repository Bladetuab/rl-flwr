# Dataset × Attack × Tau Sweep — Design

## Context

`rl-flwr` currently has `server_sweep.py`, which sweeps `(attack_mode, tau_s_quantile)` — 3 attacks × 11 tau values = 33 full federated-training runs in one `flwr run .` session, on Fashion-MNIST only (hardcoded in `task.py`). The goal of this change is to add **dataset** as a third sweep dimension, crossing `{mnist, fashion_mnist, cifar10} × {gaussian, lie, min_max} × {0.0, 0.1, ..., 1.0}` = 99 runs.

This is a real architectural change, not a config tweak: dataset selection today is hardcoded inside `task.py`'s `load_data()` (`dataset="zalando-datasets/fashion_mnist"`), the `Net` model is built for 1-channel 28×28 images only, the `FederatedDataset` is cached in a module-global that only initializes once, and `client_app.py` has no concept of "which dataset" at all. CIFAR-10 additionally needs a different model architecture (3×32×32 vs 1×28×28) and a different batch image key (`"img"` vs `"image"`).

## Goals

- Sweep all 99 `(dataset, attack_mode, tau_s_quantile)` combinations in a single `flwr run .` session, producing one combined summary CSV.
- Support MNIST, Fashion-MNIST, and CIFAR-10 with correct per-dataset model architecture, transforms, and batch-key handling.
- Keep all other sweep hyperparameters (`anom_decay=0.6`, `num_malicious=40`, `warmup_rounds=5`, `num_rounds`) constant across datasets — dataset is an independent axis, all else equal.
- Update `plot_sweep_results.py` to facet figures by dataset so the new CSV schema produces correct (not garbled) plots.
- Add resilience: a crash on one combination shouldn't lose progress on the other 98.

## Non-goals

- Per-dataset hyperparameter tuning (e.g. different `num_malicious` or `warmup_rounds` for CIFAR-10) — out of scope unless results show it's needed later.
- Changing the attack implementations themselves (`attack.py`'s `dataset_name` kwarg already exists but is currently a no-op branch; left as-is).
- Migrating `server_app.py` / `server_backup.py` (non-sweep entry points) to the dataset registry — only the sweep path is in scope.

## Architecture

### 1. `task.py` — dataset/model registry

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
```

- `Net` (existing class, 1×28×28) is shared by `mnist`/`fashion_mnist`.
- `NetCIFAR` — re-enable the already-present-but-commented CNN near the top of `task.py` (currently dead code, "Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"), for `cifar10`.
- `get_net(dataset_name) -> nn.Module` factory selects between them.
- `build_transforms(dataset_name) -> (train_transform, eval_transform)` builds a `Compose` using the registry's `image_size`/`normalization` (replacing the current module-level `TRAIN_TRANSFORMS`/`EVAL_TRANSFORMS` constants, which are Fashion-MNIST/MNIST-specific).
- `apply_train_transforms`/`apply_eval_transforms` become closures over a given transform (so they can be parametrized per dataset) rather than referencing the module-level constants directly.

**Batch key normalization:** rather than threading `"image"` vs `"img"` through `train()`/`test()`, `load_data()` renames the CIFAR-10 column immediately after loading the partition (`partition.rename_column("img", "image")` when `image_key != "image"`), so `train()`, `test()`, and the transform functions never need to know which dataset is active — they always see `batch["image"]`.

**Cache:** the current `fds: FederatedDataset | None = None` global (which only initializes once via `if fds is None`) becomes a `dict[str, FederatedDataset]` keyed by `dataset_name`, so switching datasets across sweep iterations works correctly without any external reset call.

`load_data(partition_id, num_partitions, dataset_name="fashion_mnist")` gains the new parameter (defaulted for backward compatibility with non-sweep entry points).

### 2. `client_app.py` — dataset-aware handlers

`train()` and `evaluate()` read `dataset_name = msg.content["config"].get("dataset", "fashion_mnist")`, then:
- `model = get_net(dataset_name)` instead of `model = Net()`
- `load_data(partition_id, num_partitions, dataset_name=dataset_name)`

**Important:** `CustomFedAvg_adaptive.start()` (verified in `strategy.py`, e.g. line 3519 region) takes `train_config` and `evaluate_config` as two **separate** `ConfigRecord` parameters — `train_config` is only forwarded to `configure_train` (TRAIN-round messages) and `evaluate_config` only to `configure_evaluate` (EVALUATE-round messages). The `dataset` key must be passed in **both**, or the `evaluate()` client handler won't see it (today's `evaluate()` handler reads no config at all, since `evaluate_config` is never passed by the existing sweep code).

### 3. `server_sweep.py` — three nested loops + wiring

```python
DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]
TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]
ATTACK_MODES  = ["gaussian", "lie", "min_max"]

for dataset_name in DATASET_MODES:
    for attack_mode in ATTACK_MODES:
        for tau in TAU_VALUES:
            ...
            global_model = get_net(dataset_name)
            arrays = ArrayRecord(global_model.state_dict())
            strategy = CustomFedAvg_adaptive(..., tau_s_quantile=tau, ...)
            tag = f"{dataset_name}_{attack_mode}_tau{tau:.2f}"
            ...
            try:
                result = strategy.start(
                    grid=grid,
                    initial_arrays=arrays,
                    train_config=ConfigRecord({"lr": 0.001, "dataset": dataset_name}),
                    evaluate_config=ConfigRecord({"dataset": dataset_name}),
                    num_rounds=num_rounds,
                    evaluate_fn=_make_evaluate_fn(device, dataset_name),
                )
                row = _collect_metrics(save_path, attack_mode, tau, strategy, dataset_name)
            except Exception as exc:
                row = _error_row(dataset_name, attack_mode, tau, exc)
            _append_summary_csv(row, summary_path)   # incremental write
```

- `dataset_name` is passed to clients every round via the existing `train_config` `ConfigRecord` channel (the same mechanism already used for `lr`).
- `_make_evaluate_fn(device, dataset_name)` loads the matching test set (renaming the image column same as `load_data`), builds the matching eval transform, and instantiates `get_net(dataset_name)` for server-side global evaluation.
- Output directory tag gains the dataset segment: `outputs/<date>/<time>_{dataset}_{attack}_tau{q}/`.

### 4. Resilience

- `summary_path` (the final CSV filename) is generated **once** at the start of `main()`, not per-write.
- Each iteration body is wrapped in `try/except`; on failure, log the `(dataset, attack, tau)` tag and the exception, build an error row (`final_accuracy=NaN`, numeric fields `NaN`, plus an `error` field set to `str(exc)` truncated to ~200 chars so the CSV records *why* it failed), and continue to the next combination — one bad run doesn't abort the other 98.
- `_append_summary_csv()` appends one row at a time (writing the header only if the file doesn't exist yet) instead of collecting all rows and writing once at the end — so a crash at run 80/99 still leaves 79 rows of usable results on disk.

### 5. `_collect_metrics()` / CSV schema

Gains a `dataset` field and an `error` field (empty string on success, truncated exception message on failure). Final CSV `outputs/sweep_summary_<timestamp>.csv` columns: `dataset, attack_mode, tau_s_quantile, final_accuracy, total_convlstm_forwards, mean_ratio_scored, total_convlstm_time_ms, save_path, error` — up to 99 rows.

### 6. `plot_sweep_results.py` — per-dataset faceting

- `SUMMARY_CSV` stops being a hardcoded timestamped path; it auto-resolves to the most-recently-modified `sweep_summary_*.csv` in `outputs/` (fixes a pre-existing stale-path footgun in the same function being touched here).
- `load_data()` sorts by `["dataset", "attack_mode", "tau_s_quantile"]`.
- For each `dataset` value present in the CSV: filter to that subset, set `OUT_DIR` to `sweep_figures/<dataset>/`, and call the existing 5 plot functions unchanged on the filtered frame. No change to the plotting logic itself — only the filtering/output-dir wrapper around it.

## Data flow (per sweep iteration)

1. `server_sweep.py` picks `(dataset_name, attack_mode, tau)`, builds a fresh `get_net(dataset_name)` + fresh `CustomFedAvg_adaptive(tau_s_quantile=tau, ...)`.
2. `strategy.start(...)` runs `num_rounds` rounds against the persistent `grid`. Each round's `train_config` carries `dataset_name` to clients.
3. Each client's `train()`/`evaluate()` handler reads `dataset_name` from the message, calls `load_data(..., dataset_name=dataset_name)` (cache-hit after the first round of this iteration), trains/evaluates with `get_net(dataset_name)`.
4. Server-side `attack_mode` corruption of received update messages is unaffected by this change (already dataset-agnostic — operates on flattened parameter vectors).
5. After `num_rounds`, `_make_evaluate_fn(device, dataset_name)` runs final global eval; `_collect_metrics()` builds the summary row; `_append_summary_csv()` writes it immediately.
6. Loop continues to the next `(dataset, attack, tau)`.

## Error handling

- Per-iteration `try/except` in the sweep loop (see Resilience above) — failures are logged and recorded as NaN rows, not fatal.
- No new validation is added for invalid `dataset_name` values from messages — `DATASET_REGISTRY[dataset_name]` raising `KeyError` on a typo is acceptable (sweep script is the only caller and only ever passes values from its own `DATASET_MODES` list).

## Testing / validation

This is empirical ML code, not unit-tested in the traditional sense. Validation plan:
1. Smoke-test first: temporarily shrink `DATASET_MODES`/`ATTACK_MODES`/`TAU_VALUES` to one value each and `num-server-rounds` to 2, run `flwr run .`, confirm all three datasets load, train, and produce a row in the summary CSV with no shape/key errors (this is where a CIFAR-10 channel-count or batch-key mismatch would surface immediately).
2. Confirm the incremental CSV append and per-iteration try/except by deliberately breaking one combination (e.g. temporarily renaming the CIFAR-10 HF path) and verifying the sweep continues past it and records a NaN row.
3. Run `plot_sweep_results.py` against the smoke-test CSV and confirm three per-dataset figure sets are produced under `sweep_figures/<dataset>/`.
4. Once smoke-tested, restore the full grids and run the complete 99-combination sweep.

## Open items resolved during design

- **Dataset scope:** MNIST + Fashion-MNIST + CIFAR-10 (not just the two compatible ones).
- **Sweep scale:** keep the full grid (11 taus × 3 attacks × 3 datasets = 99 runs, 50 rounds each) rather than trimming for speed.
- **Wiring approach:** single `flwr run .` session with dataset as a third nested loop (not a wrapper script with one `flwr run` per dataset).
- **Plotting:** in scope — facet by dataset.
- **Resilience:** in scope — incremental CSV + per-run error handling.
