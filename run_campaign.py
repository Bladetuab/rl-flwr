"""run_campaign.py — drive the sweep through Flower's in-process simulation API.

Why this exists
---------------
`flwr run . <federation>` starts a SuperLink daemon which spawns the simulation
in an isolated subprocess (`--isolation subprocess`). On this machine that
subprocess cannot start Ray's GCS server:

    FileNotFoundError: '/tmp/ray/session_*/logs/gcs_server.err'

Ray allocates its ports and then never launches GCS, so no log file is written
and Ray's own error reporter crashes trying to read it. The failure is NOT in
this project's code: the strategy constructs and the run reaches [ROUND 1/N]
before Ray dies, and `ray.init(...)` succeeds from this same interpreter, with
Flower's exact `runtime_env` arguments, detached, and with stdin closed.

`flwr.simulation.run_simulation` runs the ServerApp and ClientApp in THIS
process with the same Ray backend, skipping the SuperLink and the isolation
subprocess entirely. That path works here.

One wrinkle: `run_simulation` hard-codes `run_config=UserConfig()` (empty), so
`context.run_config["num-server-rounds"]` would raise. The ServerApp's
registered entry point is wrapped below to inject the config instead of editing
`server_app.py`, which stays the entry point for `flwr run`.

Usage
-----
    python run_campaign.py --rounds 30 --clients 50 --malicious 20 \\
        --datasets mnist,fashion_mnist \\
        --attacks gaussian,lie,min_max,fang,no_attack \\
        --defenses ustad_cal --local-epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--clients", type=int, default=50)
    p.add_argument("--malicious", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--datasets", default="mnist,fashion_mnist")
    p.add_argument("--defenses", default="ustad_cal")
    p.add_argument("--attacks", default="gaussian,lie,min_max,fang,no_attack")
    p.add_argument("--taus", default="")
    p.add_argument("--num-gpus", type=float, default=0.125,
                   help="GPU fraction per ClientApp; 0.125 → 8 concurrent")
    p.add_argument("--num-cpus", type=int, default=1)
    p.add_argument("--server-device", default="cuda")
    args = p.parse_args()

    # server_app reads its sweep grid from the environment AT IMPORT TIME, so
    # every override has to be in place before the import below.
    os.environ["USTAD_DATASETS"] = args.datasets
    os.environ["USTAD_DEFENSES"] = args.defenses
    os.environ["USTAD_ATTACKS"] = args.attacks
    os.environ["USTAD_NMAL"] = str(args.malicious)
    # multikrum sizes its retained set as N - M and cannot see N otherwise
    # (--clients goes straight to run_simulation below).
    os.environ["USTAD_NCLIENTS"] = str(args.clients)
    if args.taus:
        os.environ["USTAD_TAUS"] = args.taus

    from flwr.simulation import run_simulation

    from rl_flwr.client_app import app as client_app
    from rl_flwr.server_app import app as server_app

    run_config = {
        "num-server-rounds": args.rounds,
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "local-epochs": args.local_epochs,
        "server-device": args.server_device,
        "anom-decay": 0.6,
        "tau-s-quantile": 0.8,
        "attack-mode": "gaussian",   # inert: the sweep sets attack per run
    }

    # Inject run_config without touching server_app.py / client_app.py.
    _orig_main = server_app._main

    def _main_with_config(grid, context):
        context.run_config.update(run_config)
        return _orig_main(grid, context)

    server_app._main = _main_with_config

    # Each ClientApp actor builds its own Context and client_app.py reads
    # run_config["local-epochs"]. The engine REJECTS in-place mutation of a
    # client Context's run_config ("cannot be modified"), so pass the client a
    # shim Context carrying the config instead. `state` is shared by reference,
    # so anything the client stores there still reaches the engine, and the
    # engine's own Context object is left untouched. node_config (partition-id,
    # num-partitions) comes from the simulation engine and is passed through.
    from flwr.common import Context

    def _with_config(fn):
        def inner(message, context):
            shim = Context(
                run_id=context.run_id,
                node_id=context.node_id,
                node_config=context.node_config,
                state=context.state,
                run_config={**context.run_config, **run_config},
            )
            return fn(message, shim)
        return inner

    for _name, _fn in list(client_app._registered_funcs.items()):
        client_app._registered_funcs[_name] = _with_config(_fn)

    # Count runs the way server_app actually expands the grid, so the q_s sweep
    # is reflected: the USTAD family sweeps tau-s-quantile, everything else runs
    # once at BASELINE_TAU, and no_attack collapses to a single tau.
    from rl_flwr.server_app import (
        ATTACK_MODES, DATASET_MODES, DEFENSE_MODES, TAU_SWEEP_DEFENSES, _taus_for,
    )

    n_runs = sum(
        len(_taus_for(d, a))
        for _ in DATASET_MODES for d in DEFENSE_MODES for a in ATTACK_MODES
    )
    swept = sorted({d for d in DEFENSE_MODES if d in TAU_SWEEP_DEFENSES})

    print("=" * 70)
    print(f"  {args.clients} clients | {args.malicious} malicious "
          f"({100*args.malicious/args.clients:.0f}%) | {args.rounds} rounds "
          f"| {args.local_epochs} local epochs")
    print(f"  datasets : {args.datasets}")
    print(f"  defenses : {args.defenses}")
    print(f"  attacks  : {args.attacks}")
    if swept:
        print(f"  q_s sweep: {_taus_for(swept[0], 'gaussian')}  (on {', '.join(swept)})")
        print("             no_attack collapses to a single q_s")
    else:
        print("  q_s      : not swept for these defenses (single run at BASELINE_TAU)")
    print(f"  TOTAL RUNS: {n_runs}")
    print("=" * 70, flush=True)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.clients,
        backend_config={
            "client_resources": {
                "num_cpus": args.num_cpus,
                "num_gpus": args.num_gpus,
            },
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
