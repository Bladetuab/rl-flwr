"""server_app_assumption_free.py  –  dataset × defense × attack × tau_s sweep for
CustomFedAvg_adaptive_assumption_free
==========================================================================
Same structure as server_app.py, running the assumption-free strategy instead of
CustomFedAvg_adaptive_no_valdelta.  Deliberately self-contained: it imports
nothing from server_app.py, so editing one sweep never changes the other and the
before/after comparison stays controlled.

WHAT IT RUNS
────────────
CustomFedAvg_adaptive_assumption_free — USTAD-FL with three adversary-knowledge
assumptions removed: the quantile drop floor (A), the tuned absolute threshold
(B), and the clean-warm-up SVD fit (C).  Four keys let each change be attributed:

    ustad_af            gap_abs — (A)+(B)+(C).  The default.
    ustad_af_absonly    threshold_mode="absolute" — (A) only, parent's 0.40
    ustad_af_zscore     threshold_mode="zscore"   — (A)+modified-z, ablation only
    ustad_af_meansvd    robust_svd=False          — (A)+(B), parent's SVD fit

The published baselines ("ustad", "signguard", "dnc") are also registered here,
so a head-to-head sweep can be run from this file in one session.

HOW TO USE
──────────
1.  Point Flower at this app — one line in pyproject.toml:

        [tool.flwr.app.components]
        serverapp = "rl_flwr.server_app_assumption_free:app"

    or, following the same convention as server_app.py's own header, copy it
    over the active app:

        cp rl_flwr/server_app_assumption_free.py rl_flwr/server_app.py

2.  Edit the SWEEP GRID section below, or use the USTAD_* environment overrides.

3.  Run normally:
        flwr run . local-sim-gpu

Results are written to:
    outputs/<date>/<time>_<dataset>_<defense>_<attack>_tau<q>/  per-run outputs
    outputs/sweep_summary_<timestamp>.csv                       one row per run,
                                                                appended as each finishes

The run-DIR name is the provenance record; run_config.json is NOT (it reports one
combination for every run in the sweep).

NOTE: nothing in the assumption-free strategy is validated on real runs yet. The
key control is attack_mode="no_attack", which tests whether removing the quantile
drop floor takes the ~0.44 attack-free false-detection rate down to ~0.
"""

import csv
import json
import math
from datetime import datetime
from functools import partial
from pathlib import Path

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
from rl_flwr.CustomFedAvg_adaptive_assumption_free import (
    CustomFedAvg_adaptive_assumption_free,
)
from rl_flwr.baseline_defenses import DEFENSE_REGISTRY as _PUBLISHED_DEFENSES

app = ServerApp()

# ══════════════════════════════════════════════════════════════════════════════
#  DEFENSE REGISTRY  –  assumption-free variants + the published baselines
# ══════════════════════════════════════════════════════════════════════════════
# Built as a NEW dict rather than mutating the imported one, so importing this
# module never changes what server_app.py sees.
AF_DEFENSES = {
    "ustad_af":         CustomFedAvg_adaptive_assumption_free,
    "ustad_af_absonly": partial(CustomFedAvg_adaptive_assumption_free,
                                threshold_mode="absolute"),
    "ustad_af_zscore":  partial(CustomFedAvg_adaptive_assumption_free,
                                threshold_mode="zscore"),
    "ustad_af_meansvd": partial(CustomFedAvg_adaptive_assumption_free,
                                robust_svd=False),
}
DEFENSE_REGISTRY = {**_PUBLISHED_DEFENSES, **AF_DEFENSES}

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP GRID  –  edit here
# ══════════════════════════════════════════════════════════════════════════════
DATASET_MODES = ["mnist", "fashion_mnist"]
# DATASET_MODES = ["fashion_mnist"]        # single dataset to save time
# DATASET_MODES = ["cifar10"]

# The ablation ladder. absonly -> ustad_af isolates the gap branch;
# meansvd -> ustad_af isolates the robust SVD fit.
DEFENSE_MODES = ["ustad_af"]
# DEFENSE_MODES = ["ustad_af_absonly", "ustad_af", "ustad_af_meansvd"]
# DEFENSE_MODES = ["ustad", "ustad_af", "signguard", "dnc"]   # head-to-head

# Attack de-correlation: target max pairwise cosine among malicious payloads.
# None reproduces the published LIE/Min-Max behaviour (identical vectors,
# cosine 1.0). 0.90 sits just below USTAD-FL's clone threshold tau_c = 0.95.
DECORRELATE_COS = None

# q_s only gates ConvLSTM re-scoring frequency; it is NOT the filter threshold,
# and in this strategy the quantile branch is inert, so a single operating point
# is enough for a before/after comparison against the tau=0.6 FDR table.
TAU_VALUES    = [0.6]
# TAU_VALUES  = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]

# "no_attack" is the control for the drop-floor claim (A) — keep it in.
ATTACK_MODES  = ["gaussian", "lie", "min_max", "no_attack"]
# Per-attack tau list: "no_attack" ignores tau (irrelevant without an attack) → one run;
# any attack not listed here falls back to the full TAU_VALUES sweep.
TAU_BY_ATTACK = {"no_attack": [TAU_VALUES[0]]}

# q_s (tau_s_quantile) gates how often the ConvLSTM re-scores clients. Only the
# USTAD-FL family has a ConvLSTM, so the knob is INERT for every other defense --
# sweeping it there produces byte-identical runs. Restrict the sweep to the
# defenses it actually affects; everyone else runs once at BASELINE_TAU.
TAU_SWEEP_DEFENSES = {"ustad"} | set(AF_DEFENSES)
BASELINE_TAU       = 0.6   # recorded value only; inert for non-USTAD defenses

# Which model to use for cifar10: "netcifar" (small CNN) or
# "resnet18" (CIFAR-adapted ResNet-18 + GroupNorm). Threaded to clients via run config.
CIFAR_MODEL   = "resnet18"

# ── Environment overrides (used for smoke tests; empty = use the grid above) ──
# e.g.  USTAD_DATASETS=mnist USTAD_DEFENSES=ustad_af_absonly,ustad_af \
#       USTAD_ATTACKS=no_attack USTAD_TAUS=0.6 flwr run . local-sim-gpu
import os as _os

def _env_list(key, default):
    v = _os.environ.get(key, "").strip()
    return [x.strip() for x in v.split(",") if x.strip()] if v else default

DATASET_MODES   = _env_list("USTAD_DATASETS", DATASET_MODES)
DEFENSE_MODES   = _env_list("USTAD_DEFENSES", DEFENSE_MODES)
ATTACK_MODES    = _env_list("USTAD_ATTACKS",  ATTACK_MODES)
_taus           = _env_list("USTAD_TAUS", [])
if _taus:
    TAU_VALUES  = [float(t) for t in _taus]
    TAU_BY_ATTACK = {"no_attack": [TAU_VALUES[0]]}
_dc = _os.environ.get("USTAD_DECORR", "").strip()
if _dc:
    DECORRELATE_COS = None if _dc.lower() in ("none", "off") else float(_dc)

# Fixed hyperparameters shared across every run
ANOM_DECAY      = 0.6
# NOTE: the federation size comes from ~/.flwr/config.toml (local-sim-gpu = 100
# supernodes). The paper's threat model is M/N = 40%, i.e. NUM_MALICIOUS = 40
# at N = 100. Override with USTAD_NMAL to change the adversarial fraction.
NUM_MALICIOUS   = int(_os.environ.get("USTAD_NMAL", "40"))
WARMUP_ROUNDS   = 5
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_FIELDNAMES = [
    "dataset", "defense", "attack_mode", "decorrelate_cos", "assumed_malicious",
    "tau_s_quantile", "final_accuracy", "asr", "fdr", "recall",
    "total_convlstm_forwards", "mean_ratio_scored", "total_convlstm_time_ms",
    "save_path", "error",
]


def _taus_for(defense: str, attack_mode: str) -> list:
    """Tau values to run for a (defense, attack) pair.

    The USTAD-FL family sweeps q_s because it changes ConvLSTM re-scoring
    frequency. Every other defense ignores q_s entirely, so it runs once.
    """
    if defense in TAU_SWEEP_DEFENSES:
        return TAU_BY_ATTACK.get(attack_mode, TAU_VALUES)
    return [BASELINE_TAU]


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the full (dataset × defense × attack × tau) sweep in one flwr session."""

    num_rounds     = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval  = context.run_config["fraction-evaluate"]
    device         = context.run_config["server-device"]

    unknown = [d for d in DEFENSE_MODES if d not in DEFENSE_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown defense(s) {unknown}; registered: {sorted(DEFENSE_REGISTRY)}"
        )

    total_runs = sum(
        len(_taus_for(d, a))
        for _ in DATASET_MODES
        for d in DEFENSE_MODES
        for a in ATTACK_MODES
    )
    run_idx    = 0

    summary_path = _make_summary_path()

    for dataset_name in DATASET_MODES:
      for defense in DEFENSE_MODES:
        for attack_mode in ATTACK_MODES:
            for tau in _taus_for(defense, attack_mode):
                run_idx += 1
                tag = f"{dataset_name}_{defense}_{attack_mode}_tau{tau:.2f}"
                print(
                    f"\n{'='*65}\n"
                    f"[SWEEP {run_idx}/{total_runs}]  dataset={dataset_name}  "
                    f"defense={defense}  attack={attack_mode}  tau_s={tau:.2f}\n"
                    f"{'='*65}\n"
                )

                try:
                    row = _run_one_combination(
                        grid=grid,
                        context=context,
                        dataset_name=dataset_name,
                        defense=defense,
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
                    row = _error_row(dataset_name, defense, attack_mode, tau, exc)

                _append_summary_csv(row, summary_path)

                acc = row["final_accuracy"]
                acc_str = f"{acc:.4f}" if isinstance(acc, float) and not math.isnan(acc) else "NaN"
                print(
                    f"[SWEEP {run_idx}/{total_runs}]  DONE  "
                    f"acc={acc_str}  "
                    f"fdr={row['fdr']}  recall={row['recall']}\n"
                )

    print(f"\n[SWEEP] Summary written → {summary_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
def _run_one_combination(
    grid: Grid,
    context: Context,
    dataset_name: str,
    defense: str,
    attack_mode: str,
    tau: float,
    tag: str,
    num_rounds: int,
    fraction_train: float,
    fraction_eval: float,
    device: str,
) -> dict:
    """Run a single (dataset, defense, attack, tau) combination and return its row."""

    # ── Fresh global model for every run ──────────────────────────
    global_model = get_net(dataset_name, cifar_model=CIFAR_MODEL)
    arrays = ArrayRecord(global_model.state_dict())

    # ── Fresh strategy for every run ──────────────────────────────
    strategy_cls = DEFENSE_REGISTRY[defense]
    strategy = strategy_cls(
        num_malicious=NUM_MALICIOUS,
        warmup_rounds=WARMUP_ROUNDS,
        attack_mode=attack_mode,
        anom_decay=ANOM_DECAY,
        tau_s_quantile=tau,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
    )
    # Attack de-correlation is a property of the ADVERSARY, so it is set
    # identically for every defense in the comparison.
    strategy.decorrelate_cos = DECORRELATE_COS

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
        train_config=ConfigRecord({"lr": 0.001, "dataset": dataset_name, "cifar_model": CIFAR_MODEL}),
        evaluate_config=ConfigRecord({"dataset": dataset_name, "cifar_model": CIFAR_MODEL}),
        num_rounds=num_rounds,
        evaluate_fn=_make_evaluate_fn(device, dataset_name, CIFAR_MODEL),
    )

    # ── Collect metrics for summary ───────────────────────────────
    return _collect_metrics(
        save_path=save_path,
        attack_mode=attack_mode,
        tau=tau,
        strategy=strategy,
        dataset_name=dataset_name,
        defense=defense,
    )


# ─────────────────────────────────────────────────────────────────────────────
def _collect_metrics(save_path: Path, attack_mode: str,
                     tau: float, strategy, dataset_name: str,
                     defense: str = "ustad_af") -> dict:
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

    # Detection metrics from the confusion matrix accumulated over the run.
    # ASR (evasion) = FN / N and FDR = FP / N follow the paper's normalisation
    # (plot_sweep_asr_fdr.run_metrics), so these are directly comparable.
    # NOTE: this pools warm-up rounds, where no attacker exists and every drop is
    # a false positive by construction. Split by round for a fair FDR.
    cm = getattr(strategy, "confusion_matrix", {}) or {}
    TP, FP, FN, TN = (int(cm.get(k, 0)) for k in ("TP", "FP", "FN", "TN"))
    total = TP + FP + FN + TN
    asr    = (FN / total) if total else float("nan")
    fdr    = (FP / total) if total else float("nan")
    recall = (TP / (TP + FN)) if (TP + FN) else float("nan")

    # ConvLSTM cost metrics (USTAD-FL family only; baselines report 0/NaN)
    total_scored  = int(getattr(strategy, "total_convlstm_forwards", 0) or 0)
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
        "defense":                 defense,
        "attack_mode":             attack_mode,
        "decorrelate_cos":         DECORRELATE_COS,
        "assumed_malicious":       int(getattr(strategy, "assumed_malicious", 0)),
        "tau_s_quantile":          tau,
        "final_accuracy":          round(final_accuracy, 4),
        "asr":                     round(asr, 4) if not _isnan(asr) else asr,
        "fdr":                     round(fdr, 4) if not _isnan(fdr) else fdr,
        "recall":                  round(recall, 4) if not _isnan(recall) else recall,
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


def _error_row(dataset_name: str, defense: str, attack_mode: str,
               tau: float, exc: Exception) -> dict:
    """Build a summary row for a combination that raised an exception."""
    return {
        "dataset":                 dataset_name,
        "defense":                 defense,
        "attack_mode":             attack_mode,
        "decorrelate_cos":         DECORRELATE_COS,
        "assumed_malicious":       float("nan"),
        "tau_s_quantile":          tau,
        "final_accuracy":          float("nan"),
        "asr":                     float("nan"),
        "fdr":                     float("nan"),
        "recall":                  float("nan"),
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
def _make_evaluate_fn(device: str, dataset_name: str, cifar_model: str = "netcifar"):
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
        net = get_net(dataset_name, cifar_model=cifar_model)
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
