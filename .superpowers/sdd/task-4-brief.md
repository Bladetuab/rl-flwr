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

