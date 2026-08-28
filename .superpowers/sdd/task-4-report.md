# Task 4 Report: Smoke test — all 3 datasets, 1 attack, 1 tau, 2 rounds

**Status: BLOCKED** — not by a bug in Tasks 1-3's code, but by an environment-level
crash in Ray's GCS (Global Control Store) bootstrap under WSL2, which fires before
any client/dataset/model code executes. Confirmed reproducible across three
independent attempts. Step 5 (revert) was completed and verified.

---

## Step 1: Back up `server_app.py`

```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py rl_flwr/server_app.py.smoketest-bak
```

Done. Verified the backup was byte-identical to the original (same size, `diff` clean)
before proceeding.

## Step 2: Shrink the grids

`pyproject.toml`: `num-server-rounds = 50` → `num-server-rounds = 2`

`rl_flwr/server_sweep.py`:
```python
DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]   # unchanged, all 3
TAU_VALUES    = [0.5]                                    # was 11 values
ATTACK_MODES  = ["gaussian"]                             # was 3 values
```

Both edits verified in place before Step 3.

## Step 3: Activate the sweep and run it

```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_sweep.py rl_flwr/server_app.py && flwr run .
```

### What happened

`flwr run .` triggered a one-time, automatic Flower CLI migration of
`pyproject.toml`'s legacy `[tool.flwr.federations]` blocks into the new SuperLink
connection-config format (this is normal Flower 1.30.0 behavior, not something I
did — `pyproject.toml` now has those sections commented out under a
"CONFIGURATION MIGRATION NOTICE" banner; left untouched per instructions).

The run started a local SuperLink, and the **first combination (`mnist`,
`gaussian`, `tau=0.5`) made real progress**: it loaded MNIST via HuggingFace,
built the model, ran the initial server-side global evaluation successfully
(`Initial global evaluation results: {'accuracy': 0.0944, 'loss': 2.3145...}`),
and entered `[ROUND 1/2]`. Then, at the point where the strategy calls
`configure_train: Sampled 50 nodes (out of 50)` — i.e. exactly when Flower's
`RayBackend` initializes Ray to spin up the 50 simulated clients — the
**simulation host process aborted with SIGABRT**, killing the whole
`flwr-simulation` process (Ray's GCS server crashes, which then causes Ray's own
node-init code to fail with a secondary `FileNotFoundError` because the crashed
GCS server never got to write its own error log file).

### Exact traceback (representative; identical across all 3 attempts, only the
### `/tmp/ray/session_.../` path and PID differ)

```
[SWEEP 1/3]  dataset=mnist  attack=gaussian  tau_s=0.50

INFO:      Starting CustomFedAvg_adaptive strategy: ...
INFO:      Initial global evaluation results: {'accuracy': 0.0944, 'loss': 2.3145185720425445}
INFO:      [ROUND 1/2]
INFO:      configure_train: Sampled 50 nodes (out of 50)
ERROR:     An exception occured!! [Errno 2] No such file or directory: '/tmp/ray/session_2026-06-25_16-34-35_560440_4135109/logs/gcs_server.err'
ERROR:     Traceback (most recent call last):
  File ".../flwr/server/superlink/fleet/vce/vce_api.py", line 205, in run_api
    backend = backend_fn()
  File ".../flwr/server/superlink/fleet/vce/vce_api.py", line 342, in backend_fn
    return RayBackend(backend_config)
  File ".../flwr/server/superlink/fleet/vce/backend/raybackend.py", line 53, in __init__
    self.init_ray(backend_config)
  File ".../flwr/server/superlink/fleet/vce/backend/raybackend.py", line 123, in init_ray
    ray.init(...)
  File ".../ray/_private/worker.py", line 1910, in init
    _global_node = ray._private.node.Node(...)
  File ".../ray/_private/node.py", line 335, in __init__
    self.start_head_processes()
  File ".../ray/_private/node.py", line 1355, in start_head_processes
    assert self.get_gcs_client() is not None
  File ".../ray/_private/node.py", line 659, in get_gcs_client
    self._init_gcs_client()
  File ".../ray/_private/node.py", line 698, in _init_gcs_client
    with open(os.path.join(self._logs_dir, "gcs_server.err")) as err:
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/ray/session_.../logs/gcs_server.err'

WARNING:   Stopping Simulation Engine.
ERROR:     An exception occurred !! Simulation Engine crashed.
... (same traceback repeated, then:)
RuntimeError: Simulation Engine crashed.
... (then:)
RuntimeError: An error was encountered. Ending simulation.
```

Kernel-level confirmation via `dmesg` — the **actual root cause**, one entry per
attempt, each immediately preceding the FileNotFoundError above:

```
python3.10: event_engine: potentially unexpected fatal signal 6.
CPU: 6 UID: 1000 PID: <varies> Comm: event_engine Tainted: G  W  6.18.33.1-microsoft-standard-WSL2
WSL (... - CaptureCrash): Capturing crash for pid: <varies>, executable: !home!tuan!.pyenv!versions!3.10.12!bin!python3.10, signal: 6, port: 50005
```

This is a SIGABRT inside gRPC's `event_engine` thread, inside the Ray GCS server
subprocess, under the WSL2 kernel — i.e. the GCS server process itself crashes
before it can even create its own log file, which is why Ray's subsequent
"open the log file to report the error" code then raises a *second*,
misleading `FileNotFoundError`.

### Diagnosis: this is an environment regression, not a Tasks 1-3 bug

Three independent pieces of evidence rule out `task.py` / `client_app.py` /
`server_sweep.py`:

1. **The crash happens before any of Tasks 1-3's code runs for round 1's client
   side.** Dataset loading (`task.py`'s `load_data`), the `"img"`→`"image"`
   rename, `NetCIFAR`, and `client_app.py`'s dataset wiring are all client-side
   code that only executes once Ray has spun up the simulated clients — which
   never happened. The crash is purely in Flower's `RayBackend.init_ray()` →
   Ray's own cluster-bootstrap code (`ray/_private/node.py`).
2. **It is reproducible and deterministic**, not a one-off fluke:
   - Attempt 1 (fresh SuperLink, fresh `/tmp/ray` session): same crash.
   - Attempt 2 (after fully cleaning `/tmp/ray` and confirming no stray Ray
     processes): same crash, fresh session dir, identical signature.
   - Attempt 3 (diagnostic: killed the persistent SuperLink and restarted it
     fresh with `GRPC_EXPERIMENTAL_EVENT_ENGINE=poll` set — a known community
     workaround for gRPC's newer event-engine poller misbehaving on WSL2 —
     confirmed via `/proc/<pid>/environ` that the new SuperLink process actually
     inherited the env var): **still crashed, identically.** This rules out the
     specific "new gRPC event_engine poller" theory as the *fixable* cause (or
     at least shows the simple env-var fix doesn't apply here) but the SIGABRT
     signature is otherwise the same.
   - A bare standalone `ray.init(num_cpus=4)` (no Flower involved) **succeeds**
     fine in this same Python environment — so Ray itself is not categorically
     broken here; it's specifically Flower's `RayBackend` configuration (50
     simulated nodes, the specific runtime-env/log/namespace settings
     `init_ray()` passes) that triggers the crash.
3. **This exact codebase, exact Ray simulation backend, ran successfully on
   this exact machine as recently as 2026-05-07.** `outputs/` contains complete,
   successful `sweep_summary_*.csv` files from 2026-04-26, 2026-05-06, and
   2026-05-07, each with real per-row data from full Flower+Ray simulations.
   The WSL2 kernel in `uname -a` is `6.18.33.1-microsoft-standard-WSL2 #1 ...
   Fri Jun 5 01:12:21 UTC 2026` — i.e. built **after** the last known-good run.
   **This points to a regression introduced between 2026-05-07 and today
   (2026-06-25)** — most likely the WSL2 kernel update on 2026-06-05, and/or a
   Ray/gRPC version drift in the `flower-3.10.12` pyenv environment in that
   window — not a permanent "Ray can never work on WSL2" situation, and
   absolutely not something introduced by Tasks 1-3.

I did **not** attempt to fix this (e.g., pinning Ray to an older version,
downgrading gRPC, or patching Flower's `RayBackend`) — that is explicitly out of
scope for this task and would itself need its own review.

### What this means for Steps 4 and beyond

Because the SIGABRT tears down the entire `flwr-simulation` host process
abruptly, `server_sweep.py`'s own per-combination `try/except` safety net
(`main()`'s `try: row = _run_one_combination(...) except Exception as exc: ...`)
**never gets a chance to run** — the process dies from the OS signal before
Python's exception handling in `main()` can catch anything. So this is not a
case where Task 3's resilience code failed; the host process never returned
control to that code at all.

## Step 4: Inspect the resulting CSV

**No `sweep_summary_*.csv` was created.** `_make_summary_path()` only builds a
`Path` object; the physical file is created lazily by `_append_summary_csv()`'s
first `open(..., "a")` call, which happens only after `_run_one_combination`
returns (success) or raises back into `main()`'s `except` (error row). Since the
crash killed the process before either path completed, that first append never
happened.

Verified directly — listing all `sweep_summary_*.csv` files by mtime shows only
the three pre-existing ones from April/May, none from today:

```
$ ls -la --time-style=full-iso outputs/sweep_summary_*.csv
-rw-r--r-- 1 tuan tuan 4088 2026-04-26 17:28:07 outputs/sweep_summary_20260426_083459.csv
-rw-r--r-- 1 tuan tuan 1171 2026-05-06 20:53:22 outputs/sweep_summary_20260506_205322.csv
-rw-r--r-- 1 tuan tuan 4138 2026-05-07 08:31:13 outputs/sweep_summary_20260507_083113.csv
```

What *was* created (and left in place, since it's harmless evidence of the
attempt, not garbage): four partial per-run output directories under
`outputs/2026-06-25/`, one per attempt, each containing only `run_config.json`
(written by `create_run_dir()` before training starts) and nothing else — no
`results.json`, no `convlstm_cost.csv`, confirming training never proceeded past
node sampling in any attempt:

```
outputs/2026-06-25/16-34-35_mnist_gaussian_tau0.50/run_config.json
outputs/2026-06-25/16-36-53_mnist_gaussian_tau0.50/run_config.json
outputs/2026-06-25/16-38-11_mnist_gaussian_tau0.50/run_config.json
outputs/2026-06-25/16-38-56_mnist_gaussian_tau0.50/run_config.json
```

Each `run_config.json` correctly shows `"num-server-rounds": 2`, confirming
Step 2's grid-shrink was correctly picked up by the run before it crashed.

**Expected vs. actual:**
- Expected: header row + 3 data rows (mnist, fashion_mnist, cifar10), each with
  `attack_mode=gaussian`, `tau_s_quantile=0.5`, real `final_accuracy` in (0,1),
  empty `error`.
- Actual: **no file exists at all.** Step 4 is unsatisfiable in this run.

## Step 5: Restore `server_app.py` and revert the grid shrink

```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && cp rl_flwr/server_app.py.smoketest-bak rl_flwr/server_app.py && rm rl_flwr/server_app.py.smoketest-bak
```

Plus manual edits reverting `pyproject.toml` and `rl_flwr/server_sweep.py`.

**Completed and verified.** Proof, captured after the revert:

`pyproject.toml`:
```
$ grep -n "num-server-rounds" pyproject.toml
35:num-server-rounds = 50
```

`rl_flwr/server_sweep.py`:
```
$ grep -n "DATASET_MODES\|TAU_VALUES\|ATTACK_MODES" rl_flwr/server_sweep.py | head -3
51:DATASET_MODES = ["mnist", "fashion_mnist", "cifar10"]
52:TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]
53:ATTACK_MODES  = ["gaussian", "lie", "min_max"]
```

`rl_flwr/server_app.py`:
```
$ diff rl_flwr/server_app.py rl_flwr/server_sweep.py
(files differ — expected: server_app.py is back to its original, pre-sweep content)
$ head -3 rl_flwr/server_app.py
"""server_app_sweep.py  –  tau_s × attack sweep for CustomFedAvg_adaptive
==========================================================================
Runs ALL (attack_mode, tau_s_quantile) combinations inside a single
$ md5sum rl_flwr/server_app.py
9d93c53c4acf5a2c1d009f99b2809b24  rl_flwr/server_app.py
```

(`.smoketest-bak` confirmed absent via `ls`.)

**Note on `pyproject.toml`'s `[tool.flwr.federations]` section:** this section
was automatically rewritten by the Flower CLI itself during Step 3 (commented
out, replaced with a migration notice pointing to the new SuperLink
connection-config format) — this is Flower 1.30.0's own one-time legacy-config
migration behavior, triggered the first time `flwr run` saw the old-style
`[tool.flwr.federations]` block, and is unrelated to anything I edited for this
task. Per explicit instruction received mid-task, this was **left as-is** and
NOT reverted — only `num-server-rounds` (the value this task's Step 2/5 actually
touched) was reverted.

### Process/environment cleanup also performed

The crash left orphaned `flower-superlink` / `flower-superexec` /
`flwr-simulation` (defunct) processes and stale `/tmp/ray/session_*` directories
after each attempt. All were identified and killed/removed before finishing this
task, so no Flower/Ray processes from this investigation are still running.

## Issues / concerns

1. **Primary blocker (environment, not code):** Ray's GCS server SIGABRTs
   (`event_engine: potentially unexpected fatal signal 6`, confirmed via
   `dmesg`) inside Flower's `RayBackend.init_ray()`, every time, before any
   client-side training code runs. Reproduced 3x, including one isolated
   attempt with a known gRPC/WSL2 workaround env var
   (`GRPC_EXPERIMENTAL_EVENT_ENGINE=poll`) that did not help.
2. **This is a regression, not a permanent incompatibility.** `outputs/`
   contains real, successful sweep CSVs from this same machine/codebase/Ray
   backend dated through 2026-05-07; the WSL2 kernel was rebuilt 2026-06-05
   (after that last good run). The most likely culprits, in order of
   suspicion: (a) the WSL2 kernel update, (b) a Ray or gRPC version
   drift/auto-update inside the `flower-3.10.12` pyenv env since May. Next
   step for whoever picks this up: check `pip show ray grpcio` against
   whatever was installed in early May, and/or try pinning Ray to a known-good
   version, and/or test on a non-WSL2 Linux host to confirm the kernel-version
   theory.
3. **Tasks 1-3's actual code (the things this smoke test was supposed to
   exercise — `NetCIFAR`'s shape, the `"img"`→`"image"` rename, the
   per-dataset `FederatedDataset` cache, `client_app.py`'s dataset wiring) was
   never reached and remains unverified.** The only thing this smoke test
   confirmed about Tasks 1-3 is that `mnist`'s dataset load + initial
   server-side global evaluation work correctly (real accuracy/loss numbers
   were produced) — everything client-side, and all of `fashion_mnist` and
   `cifar10`, is still completely untested.
4. **Downstream impact:** Task 5 (deliberate-break check) and Task 6
   (per-dataset plot faceting, which explicitly says it "reuses this exact
   CSV") are both blocked on this — Task 6 in particular has no CSV to work
   from now, since none was produced.
5. The repo is otherwise back to its pre-task state for everything this task
   was responsible for (verified above); the only difference from before this
   task started is the new, harmless `outputs/2026-06-25/.../run_config.json`
   partial-run directories and the unrelated, instructed-to-keep
   `pyproject.toml` federations-migration comment block.
