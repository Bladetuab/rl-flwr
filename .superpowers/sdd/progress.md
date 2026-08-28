# Dataset Sweep Implementation — Progress Ledger (no-git mode)
Plan: docs/superpowers/plans/2026-06-25-dataset-sweep.md
Started: 2026-06-25


Task 1: complete (task.py registry/NetCIFAR/get_net/build_transforms/load_data, review clean — Approved, no issues)
Task 2: complete (client_app.py train/evaluate dataset wiring, review clean — Approved, no issues)
Task 3: complete (server_sweep.py full rewrite — 3D loop, train_config+evaluate_config dual wiring, resilience, review clean — Approved, no issues; train_config/evaluate_config split independently verified against strategy.py source)

Task 4: BLOCKED, not complete — `flwr run .` smoke test cannot run on this machine right now.
  Root cause: Ray GCS server SIGABRTs under WSL2 (event_engine fatal signal 6, confirmed
  via dmesg) inside Flower's RayBackend.init_ray(), every attempt (3x reproduced), before
  ANY client-side code executes. Confirmed environment regression (outputs/ has successful
  sweep CSVs through 2026-05-07; WSL2 kernel rebuilt 2026-06-05), not a Tasks 1-3 bug, and
  not fixable by retrying or by a different model. All temp edits fully reverted (verified
  pyproject.toml num-server-rounds=50, server_sweep.py grids, server_app.py md5sum match).
  Compensating check done instead (Ray-free, direct, real HF data, no Flower/Ray involved):
  for each of mnist/fashion_mnist/cifar10, called load_data() -> pulled a real batch ->
  verified batch["image"] shape (proves cifar10's "img"->"image" rename fires on real data)
  -> ran it through get_net() + CrossEntropyLoss forward+backward. All 3 PASS. This retires
  the highest-risk untested logic (CIFAR column rename + NetCIFAR shape vs real data) but
  does NOT verify the Flower integration: client_app.py reading `dataset` from per-round
  config, the 3-loop orchestration, the resilience try/except actually catching a failure,
  or the incremental CSV write. Those remain untested until the env is fixed.
  Also verified: Flower CLI's auto-migration of pyproject.toml's [tool.flwr.federations]
  into ~/.flwr/config.toml is complete and correct (local-sim/local-sim-gpu both present,
  same num-supernodes/client-resources as before) — future `flwr run .` unaffected.
  BLOCKS Task 5 (needs a working sim to trigger a deliberate break) and Task 6 (needs the
  CSV that was never produced). Escalated to human for direction — see chat for decision.

Follow-up diagnostic (same day, still Task 4's blocker): found that flwr/ray/grpcio/protobuf
  were all freshly (re)installed in a single ~20min window the same session this blocker
  was first hit — flwr 1.30.0 (released 2026-05-20) and grpcio 1.81.1 (released 2026-06-11)
  both postdate the 2026-05-07 last-good run, so a package-bump regression (not just the
  WSL2 kernel rebuild) looked equally plausible. Ran the discriminating test: pinned
  grpcio==1.80.0 (resolvable against both flwr==1.30.0's >=1.70.0,<2.0.0 and
  ray==2.51.1's >=1.42.0 constraints, confirmed via PyPI metadata before installing) as a
  throwaway (not added to pyproject.toml) and re-ran the same shrunk smoke test.
  RESULT: identical SIGABRT, identical dmesg signature (event_engine fatal signal 6).
  This rules OUT the grpcio/flwr package bump as sole cause — points back at the WSL2
  kernel (rebuilt 2026-06-05) or something else not yet isolated. grpcio is currently
  left at 1.80.0 in the live env (harmless either way, pip check clean); not reverted to
  1.81.1 since the user hasn't asked and it doesn't matter for resolution. All temp
  edits (pyproject.toml num-server-rounds, server_sweep.py grids, server_app.py) reverted
  and verified again (md5sum 9d93c53c4acf5a2c1d009f99b2809b24, same as Task 4's original
  revert). Orphaned flower-superlink/superexec processes from the crash killed, stale
  /tmp/ray session removed. Task 4 remains BLOCKED — still escalated to human.
