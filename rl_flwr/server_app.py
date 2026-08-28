# """server_app_sweep.py  –  dataset × tau_s × attack sweep for CustomFedAvg_adaptive
# ==========================================================================
# Runs ALL (dataset, attack_mode, tau_s_quantile) combinations inside a single
# `flwr run .` execution.  The Flower Grid stays open throughout; a fresh
# strategy + model is created for every combination.

# HOW TO USE
# ──────────
# 1.  Rename / copy this file over server_app.py:
#         cp server_sweep.py server_app.py

# 2.  Edit the SWEEP GRID section below to choose datasets, attacks, tau
#     values, and any other hyperparameters.

# 3.  Run normally:
#         flwr run . local-sim-gpu

# Results are written to:
#     outputs/<date>/<time>_<dataset>_<attack>_tau<q>/   per-run outputs (CSV, JSON …)
#     outputs/sweep_summary_<timestamp>.csv              one row per (dataset, attack, tau) run,
#                                                         appended incrementally as each run finishes
# """

# import csv
# import json
# import math
# from datetime import datetime
# from pathlib import Path

# import torch
# from datasets import load_dataset
# from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
# from flwr.serverapp import Grid, ServerApp
# from torch.utils.data import DataLoader

# from rl_flwr.task import (
#     DATASET_REGISTRY,
#     get_net,
#     build_transforms,
#     make_eval_transform_fn,
#     create_run_dir,
#     test,
# )
# from rl_flwr.CustomFedAvg_adaptive_no_valdelta import CustomFedAvg_adaptive_no_valdelta
# from rl_flwr.baseline_defenses import DEFENSE_REGISTRY

# app = ServerApp()

# # ══════════════════════════════════════════════════════════════════════════════
# #  SWEEP GRID  –  edit here
# # ══════════════════════════════════════════════════════════════════════════════
# # DATASET_MODES = ["mnist", "fashion_mnist"]
# DATASET_MODES = ["fashion_mnist"]  # smoke test: only run Fashion-MNIST to save time
# # Defenses to compare head-to-head. "ustad" is USTAD-FL (this paper);
# # "signguard" and "dnc" are our implementations of the published baselines.
# # DEFENSE_MODES = ["ustad", "signguard", "dnc"]
# DEFENSE_MODES = ["signguard"]

# # Attack de-correlation: target max pairwise cosine among malicious payloads.
# # None reproduces the published LIE/Min-Max behaviour (identical vectors,
# # cosine 1.0). 0.90 sits just below USTAD-FL's clone threshold tau_c = 0.95.
# DECORRELATE_COS = None
# # DATASET_MODES = ["cifar10"]

# TAU_VALUES    = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]

# ATTACK_MODES  = ["min_max", "lie", "gaussian"]
# # ATTACK_MODES  = ["no_attack"]
# # Per-attack tau list: "no_attack" ignores tau (irrelevant without an attack) → one run;
# # any attack not listed here falls back to the full TAU_VALUES sweep.
# TAU_BY_ATTACK = {"no_attack": [0]}

# # q_s (tau_s_quantile) gates how often the ConvLSTM re-scores clients. Only
# # USTAD-FL has a ConvLSTM, so the knob is INERT for every other defense --
# # sweeping it there produces byte-identical runs. Restrict the sweep to the
# # defenses it actually affects; everyone else runs once at BASELINE_TAU.
# TAU_SWEEP_DEFENSES = {"ustad"}
# BASELINE_TAU       = 0.6   # recorded value only; inert for non-USTAD defenses

# # Which model to use for cifar10: "netcifar" (small CNN) or
# # "resnet18" (CIFAR-adapted ResNet-18 + GroupNorm). Threaded to clients via run config.
# CIFAR_MODEL   = "resnet18"

# # ── Environment overrides (used for smoke tests; empty = use the grid above) ──
# # e.g.  USTAD_DATASETS=mnist USTAD_DEFENSES=ustad,signguard,dnc \
# #       USTAD_ATTACKS=lie USTAD_TAUS=0.6 USTAD_DECORR=0.90 flwr run . local-sim-gpu
# import os as _os

# def _env_list(key, default):
#     v = _os.environ.get(key, "").strip()
#     return [x.strip() for x in v.split(",") if x.strip()] if v else default

# DATASET_MODES   = _env_list("USTAD_DATASETS", DATASET_MODES)
# DEFENSE_MODES   = _env_list("USTAD_DEFENSES", DEFENSE_MODES)
# ATTACK_MODES    = _env_list("USTAD_ATTACKS",  ATTACK_MODES)
# _taus           = _env_list("USTAD_TAUS", [])
# if _taus:
#     TAU_VALUES  = [float(t) for t in _taus]
#     TAU_BY_ATTACK = {"no_attack": [TAU_VALUES[0]]}
# _dc = _os.environ.get("USTAD_DECORR", "").strip()
# if _dc:
#     DECORRELATE_COS = None if _dc.lower() in ("none", "off") else float(_dc)

# # Fixed hyperparameters shared across every run
# ANOM_DECAY      = 0.6
# # NOTE: the federation size comes from ~/.flwr/config.toml (local-sim-gpu = 100
# # supernodes). The paper's threat model is M/N = 40%, i.e. NUM_MALICIOUS = 40
# # at N = 100. The value below is whatever the last experiment used; override
# # with USTAD_NMAL=40 to reproduce the paper's regime.
# NUM_MALICIOUS   = int(_os.environ.get("USTAD_NMAL", "40"))
# WARMUP_ROUNDS   = 5
# # ══════════════════════════════════════════════════════════════════════════════

# SUMMARY_FIELDNAMES = [
#     "dataset", "defense", "attack_mode", "decorrelate_cos", "assumed_malicious",
#     "tau_s_quantile", "final_accuracy", "asr", "fdr", "recall",
#     "total_convlstm_forwards", "mean_ratio_scored", "total_convlstm_time_ms",
#     "save_path", "error",
# ]


# def _taus_for(defense: str, attack_mode: str) -> list:
#     """Tau values to run for a (defense, attack) pair.

#     USTAD-FL sweeps q_s because it changes ConvLSTM re-scoring frequency.
#     Every other defense ignores q_s entirely, so it runs once.
#     """
#     if defense in TAU_SWEEP_DEFENSES:
#         return TAU_BY_ATTACK.get(attack_mode, TAU_VALUES)
#     return [BASELINE_TAU]


# @app.main()
# def main(grid: Grid, context: Context) -> None:
#     """Run the full (dataset × attack × tau) sweep within one flwr session."""

#     num_rounds     = context.run_config["num-server-rounds"]
#     fraction_train = context.run_config["fraction-train"]
#     fraction_eval  = context.run_config["fraction-evaluate"]
#     device         = context.run_config["server-device"]

#     total_runs = sum(
#         len(_taus_for(d, a))
#         for _ in DATASET_MODES
#         for d in DEFENSE_MODES
#         for a in ATTACK_MODES
#     )
#     run_idx    = 0

#     summary_path = _make_summary_path()

#     for dataset_name in DATASET_MODES:
#       for defense in DEFENSE_MODES:
#         for attack_mode in ATTACK_MODES:
#             for tau in _taus_for(defense, attack_mode):
#                 run_idx += 1
#                 tag = f"{dataset_name}_{defense}_{attack_mode}_tau{tau:.2f}"
#                 print(
#                     f"\n{'='*65}\n"
#                     f"[SWEEP {run_idx}/{total_runs}]  dataset={dataset_name}  "
#                     f"defense={defense}  attack={attack_mode}  tau_s={tau:.2f}\n"
#                     f"{'='*65}\n"
#                 )

#                 try:
#                     row = _run_one_combination(
#                         grid=grid,
#                         context=context,
#                         dataset_name=dataset_name,
#                         defense=defense,
#                         attack_mode=attack_mode,
#                         tau=tau,
#                         tag=tag,
#                         num_rounds=num_rounds,
#                         fraction_train=fraction_train,
#                         fraction_eval=fraction_eval,
#                         device=device,
#                     )
#                 except Exception as exc:
#                     print(f"[SWEEP {run_idx}/{total_runs}]  FAILED  {exc}")
#                     row = _error_row(dataset_name, defense, attack_mode, tau, exc)

#                 _append_summary_csv(row, summary_path)

#                 acc = row["final_accuracy"]
#                 acc_str = f"{acc:.4f}" if isinstance(acc, float) and not math.isnan(acc) else "NaN"
#                 print(
#                     f"[SWEEP {run_idx}/{total_runs}]  DONE  "
#                     f"acc={acc_str}  "
#                     f"forwards={row['total_convlstm_forwards']}\n"
#                 )

#     print(f"\n[SWEEP] Summary written → {summary_path.resolve()}")


# # ─────────────────────────────────────────────────────────────────────────────
# def _run_one_combination(
#     grid: Grid,
#     context: Context,
#     dataset_name: str,
#     defense: str,
#     attack_mode: str,
#     tau: float,
#     tag: str,
#     num_rounds: int,
#     fraction_train: float,
#     fraction_eval: float,
#     device: str,
# ) -> dict:
#     """Run a single (dataset, attack, tau) combination and return its summary row."""

#     # ── Fresh global model for every run ──────────────────────────
#     global_model = get_net(dataset_name, cifar_model=CIFAR_MODEL)
#     arrays = ArrayRecord(global_model.state_dict())

#     # ── Fresh strategy for every run ──────────────────────────────
#     strategy_cls = DEFENSE_REGISTRY[defense]
#     strategy = strategy_cls(
#         num_malicious=NUM_MALICIOUS,
#         warmup_rounds=WARMUP_ROUNDS,
#         attack_mode=attack_mode,
#         anom_decay=ANOM_DECAY,
#         tau_s_quantile=tau,
#         fraction_train=fraction_train,
#         fraction_evaluate=fraction_eval,
#     )
#     # Attack de-correlation is a property of the ADVERSARY, so it is set
#     # identically for every defense in the comparison.
#     strategy.decorrelate_cos = DECORRELATE_COS

#     # ── Per-run output directory ──────────────────────────────────
#     save_path, run_dir = create_run_dir(config=context.run_config)
#     named_path = save_path.parent / f"{save_path.name}_{tag}"
#     save_path.rename(named_path)
#     save_path = named_path
#     strategy.set_save_path_and_run_dir(save_path, run_dir)

#     # ── Run FL training ───────────────────────────────────────────
#     strategy.start(
#         grid=grid,
#         initial_arrays=arrays,
#         train_config=ConfigRecord({"lr": 0.001, "dataset": dataset_name, "cifar_model": CIFAR_MODEL}),
#         evaluate_config=ConfigRecord({"dataset": dataset_name, "cifar_model": CIFAR_MODEL}),
#         num_rounds=num_rounds,
#         evaluate_fn=_make_evaluate_fn(device, dataset_name, CIFAR_MODEL),
#     )

#     # ── Collect metrics for summary ───────────────────────────────
#     return _collect_metrics(
#         save_path=save_path,
#         attack_mode=attack_mode,
#         tau=tau,
#         strategy=strategy,
#         dataset_name=dataset_name,
#         defense=defense,
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# def _collect_metrics(save_path: Path, attack_mode: str,
#                      tau: float, strategy, dataset_name: str,
#                      defense: str = "ustad") -> dict:
#     """Read per-run CSVs / JSON and return a summary dict."""

#     # Final accuracy from results.json
#     final_accuracy = float("nan")
#     results_file = save_path / "results.json"
#     if results_file.exists():
#         with open(results_file) as f:
#             data = json.load(f)
#         # results.json stores nested server/client metrics per round.
#         # Older runs may still have top-level accuracy fields.
#         if isinstance(data, list) and data:
#             final_accuracy = _extract_final_accuracy(data[-1])
#         elif isinstance(data, dict) and data:
#             last_key = str(max(int(k) for k in data.keys()))
#             final_accuracy = _extract_final_accuracy(data[last_key])

#     # Detection metrics from the confusion matrix accumulated over the run.
#     # ASR (evasion) = FN / N and FDR = FP / N follow the paper's normalisation
#     # (plot_sweep_asr_fdr.run_metrics), so these are directly comparable.
#     cm = getattr(strategy, "confusion_matrix", {}) or {}
#     TP, FP, FN, TN = (int(cm.get(k, 0)) for k in ("TP", "FP", "FN", "TN"))
#     total = TP + FP + FN + TN
#     asr    = (FN / total) if total else float("nan")
#     fdr    = (FP / total) if total else float("nan")
#     recall = (TP / (TP + FN)) if (TP + FN) else float("nan")

#     # ConvLSTM cost metrics (USTAD-FL only; baselines report 0/NaN)
#     total_scored  = int(getattr(strategy, "total_convlstm_forwards", 0) or 0)
#     mean_ratio    = float("nan")
#     total_time_ms = float("nan")

#     cost_csv = save_path / "convlstm_cost.csv"
#     if cost_csv.exists():
#         import pandas as pd
#         df = pd.read_csv(cost_csv)
#         mean_ratio    = float(df["ratio_scored"].mean())
#         total_time_ms = float(df["t_convlstm"].sum() * 1000)

#     return {
#         "dataset":                 dataset_name,
#         "defense":                 defense,
#         "attack_mode":             attack_mode,
#         "decorrelate_cos":         DECORRELATE_COS,
#         "assumed_malicious":       int(getattr(strategy, "assumed_malicious", 0)),
#         "tau_s_quantile":          tau,
#         "final_accuracy":          round(final_accuracy, 4),
#         "asr":                     round(asr, 4) if not _isnan(asr) else asr,
#         "fdr":                     round(fdr, 4) if not _isnan(fdr) else fdr,
#         "recall":                  round(recall, 4) if not _isnan(recall) else recall,
#         "total_convlstm_forwards": total_scored,
#         "mean_ratio_scored":       round(mean_ratio, 4) if not _isnan(mean_ratio) else mean_ratio,
#         "total_convlstm_time_ms":  round(total_time_ms, 2) if not _isnan(total_time_ms) else total_time_ms,
#         "save_path":               str(save_path),
#         "error":                   "",
#     }


# def _extract_final_accuracy(entry: dict) -> float:
#     """Extract final accuracy from current and legacy results.json shapes."""
#     if not isinstance(entry, dict):
#         return float("nan")

#     server_metrics = entry.get("evaluate_metrics_serverapp") or {}
#     if "accuracy" in server_metrics:
#         return float(server_metrics["accuracy"])

#     client_metrics = entry.get("evaluate_metrics_clientapp") or {}
#     if "eval_acc" in client_metrics:
#         return float(client_metrics["eval_acc"])

#     return float(entry.get("accuracy", entry.get("eval_accuracy", float("nan"))))


# def _error_row(dataset_name: str, defense: str, attack_mode: str,
#                tau: float, exc: Exception) -> dict:
#     """Build a summary row for a combination that raised an exception."""
#     return {
#         "dataset":                 dataset_name,
#         "defense":                 defense,
#         "attack_mode":             attack_mode,
#         "decorrelate_cos":         DECORRELATE_COS,
#         "assumed_malicious":       float("nan"),
#         "tau_s_quantile":          tau,
#         "final_accuracy":          float("nan"),
#         "asr":                     float("nan"),
#         "fdr":                     float("nan"),
#         "recall":                  float("nan"),
#         "total_convlstm_forwards": float("nan"),
#         "mean_ratio_scored":       float("nan"),
#         "total_convlstm_time_ms":  float("nan"),
#         "save_path":               "",
#         "error":                   str(exc)[:200],
#     }


# def _isnan(x):
#     try:
#         import math
#         return math.isnan(x)
#     except Exception:
#         return False


# def _make_summary_path() -> Path:
#     out_dir = Path("outputs")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     return out_dir / f"sweep_summary_{ts}.csv"


# def _append_summary_csv(row: dict, summary_path: Path):
#     file_exists = summary_path.exists()
#     with open(summary_path, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)


# # ─────────────────────────────────────────────────────────────────────────────
# def _make_evaluate_fn(device: str, dataset_name: str, cifar_model: str = "netcifar"):
#     """Server-side global evaluation function for a specific dataset."""

#     cfg = DATASET_REGISTRY[dataset_name]
#     _, eval_transform = build_transforms(dataset_name)

#     def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
#         global_test_set = load_dataset(cfg["hf_path"])["test"]
#         if cfg["image_key"] != "image":
#             global_test_set = global_test_set.rename_column(cfg["image_key"], "image")
#         testloader = DataLoader(
#             global_test_set.with_transform(make_eval_transform_fn(eval_transform)),
#             batch_size=32,
#         )
#         net = get_net(dataset_name, cifar_model=cifar_model)
#         net.load_state_dict(arrays.to_torch_state_dict())
#         net.to(device)
#         loss, accuracy = test(net, testloader, device=device)
#         return MetricRecord({"accuracy": accuracy, "loss": loss})

#     return global_evaluate



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
from rl_flwr.CustomFedAvg_adaptive_calibrated import (
    CustomFedAvg_adaptive_calibrated,
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

# ── calibrated-fusion + fixed-training family (IMPROVEMENT.md sections 1-3) ──
# NOTE: these subclass CustomFedAvg_adaptive_no_valdelta, so they carry the
# PARENT's quantile+absolute filter, not the gap filter of ustad_af. They isolate
# the fusion and training fixes; combining them with the gap filter is a separate
# change. `ustad_cal_statonly` is the reference configuration — temporal arm off,
# i.e. the statistical screen alone, which is already at ceiling on Gaussian/
# LIE/Min-Max (see IMPROVEMENT.md section 1.4).
CAL_DEFENSES = {
    "ustad_cal":          CustomFedAvg_adaptive_calibrated,
    "ustad_cal_raw":      partial(CustomFedAvg_adaptive_calibrated,
                                  calibrate_temporal=False),
    "ustad_cal_wsum":     partial(CustomFedAvg_adaptive_calibrated,
                                  fusion_mode="wsum"),
    "ustad_cal_maxfuse":  partial(CustomFedAvg_adaptive_calibrated,
                                  fusion_mode="max", calibrate_temporal=False),
    "ustad_cal_statonly": partial(CustomFedAvg_adaptive_calibrated,
                                  temporal_weight=0.0),
    # Fang ablation ladder. The four median-referenced signals are blinded when a
    # clone majority captures the median, so reference-free evidence (clone,
    # uniformity) may override the weighted sum instead of being diluted by it.
    #   noreffree -> ustad_cal isolates the override
    #   nounif    -> ustad_cal isolates the new uniformity signal
    "ustad_cal_noreffree": partial(CustomFedAvg_adaptive_calibrated,
                                   ref_free_override=False),
    "ustad_cal_nounif":    partial(CustomFedAvg_adaptive_calibrated,
                                   use_uniformity=False),
    # Reproduces the 2026-08-26 defect: clone in the override. Benign pairwise
    # cosine drifts past tau_c = 0.95 as training converges (measured 0.979 by
    # round 30), so promoting clone flagged 39-72% of honest client-rounds.
    # Kept for the ablation table, NOT a usable configuration.
    "ustad_cal_cloneover": partial(CustomFedAvg_adaptive_calibrated,
                                   override_on_clone=True),
}

DEFENSE_REGISTRY = {**_PUBLISHED_DEFENSES, **AF_DEFENSES, **CAL_DEFENSES}

# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTED strategy.py CLASSES
# ══════════════════════════════════════════════════════════════════════════════
# _run_one() always constructs with the SAME seven kwargs (see the strategy_cls
# call below): num_malicious, warmup_rounds, attack_mode, anom_decay,
# tau_s_quantile, fraction_train, fraction_evaluate.  The USTAD family declares
# all of them; the older classes in strategy.py do not, and since each forwards
# **kwargs straight to FedAvg/MultiKrum, an undeclared name raises TypeError.
#
# _adapt() closes that gap WITHOUT touching strategy.py: it keeps only the names
# the target actually declares (plus the two FedAvg passes through), renames
# where the target spells a concept differently, and drops the rest.  Adding a
# class is then one registry line.
import inspect as _inspect

_FEDAVG_PASSTHROUGH = {"fraction_train", "fraction_evaluate"}


def _adapt(cls, rename=None, extra=None):
    """Wrap `cls` so it tolerates _run_one()'s fixed kwarg set.

    rename: {call_name: target_name}; map to None to drop the kwarg entirely.
    extra:  kwargs the target requires that _run_one() does not supply.
    """
    rename, extra = rename or {}, extra or {}
    declared = set(_inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}

    def factory(**kw):
        out = {}
        for key, val in kw.items():
            key = rename.get(key, key)
            if key is None:
                continue
            if key in declared or key in _FEDAVG_PASSTHROUGH:
                out[key] = val
        out.update(extra)
        return cls(**out)

    factory.__name__ = f"adapted_{cls.__name__}"
    return factory


# Registered but NOT selected: DEFENSE_MODES below still controls what runs.
# Verified to instantiate under _run_one()'s kwargs.
from rl_flwr.strategy import (          # noqa: E402
    CustomKrum_attack as _CustomKrum_attack,
    FedAvgWithLBFGSHVP as _FedAvgWithLBFGSHVP,
    CustomFedAvg_adaptive as _CustomFedAvg_adaptive,
    CustomFedAvg_attack as _CustomFedAvg_attack,
)

STRATEGY_PY_DEFENSES = {
    # Multi-Krum. Spells the count `num_malicious_nodes` and needs the size of
    # the retained set explicitly: N - M, i.e. 60 at the paper's N=100/M=40 and
    # 30 at the N=50/M=20 campaign.
    #
    # This was hardcoded to 60 and it is NOT a harmless default. select_multikrum
    # ends in argsort(scores)[:num_nodes_to_select] with no clamp, so 60 on a
    # 50-client run retains every reply and Multi-Krum becomes plain FedAvg over
    # all attackers -- while `num_malicious_nodes` tracks --malicious correctly,
    # so half the logged config looks right. N is not visible in this module
    # (run_campaign.py passes --clients straight to run_simulation), so read it
    # from the environment and FAIL CLOSED: defaulting to 100 would reinstate
    # exactly the silent mismatch being fixed. `_os` resolves at call time.
    "multikrum": lambda **kw: _multikrum_factory(**kw),
    # FLDetector. Spells the warm-up `warmup_rounds_attack`.
    "fldetector": _adapt(_FedAvgWithLBFGSHVP,
                         rename={"warmup_rounds": "warmup_rounds_attack",
                                 "anom_decay": None, "tau_s_quantile": None}),
    # The pre-assumption-free USTAD-FL (adaptive quantile). Needs no renaming;
    # listed so the quantile variant can be re-run under the current protocol.
    "ustad_quantile": _adapt(_CustomFedAvg_adaptive),
    "fedavg": _adapt(_CustomFedAvg_attack, rename={"anom_decay": None, "tau_s_quantile": None}),
}


def _multikrum_factory(**kw):
    """Build CustomKrum_attack with num_nodes_to_select = N - M for THIS run."""
    n_raw = _os.environ.get("USTAD_NCLIENTS", "").strip()
    if not n_raw:
        raise RuntimeError(
            "multikrum needs the federation size to size its retained set. "
            "Set USTAD_NCLIENTS (run_campaign.py exports it from --clients)."
        )
    n_clients = int(n_raw)
    n_mal = int(kw.get("num_malicious", 0))
    if n_clients - n_mal < 1:
        raise ValueError(
            f"USTAD_NCLIENTS={n_clients} with num_malicious={n_mal} leaves no "
            "honest clients to retain."
        )
    return _adapt(
        _CustomKrum_attack,
        rename={"num_malicious": "num_malicious_nodes",
                "anom_decay": None, "tau_s_quantile": None},
        extra={"num_nodes_to_select": n_clients - n_mal},
    )(**kw)


DEFENSE_REGISTRY.update(STRATEGY_PY_DEFENSES)

# ── FLTrust ───────────────────────────────────────────────────────────────────
# Not a signature mismatch but a dependency gap: FLTrust needs a server-side
# root dataset and training loop (model_fn, root_loader, loss_fn, optimizer_fn),
# and the root set must match the dataset of the run. _run_one() therefore has
# to hand the factory `dataset_name`; factories that want it set .wants_dataset.
#
# NOTE on the root set. task.build_fltrust_rootset_loader() is NOT used here: it
# is hardcoded to CIFAR-10 (mnist/fashion_mnist are commented out), it caches
# into the module-global `fds`, and it draws the root sample from the *test*
# split -- which is the split the global model is scored on, so FLTrust would be
# handed part of the evaluation set. We build from the *train* split instead.
# This is a deliberate difference from whatever produced the earlier FLTrust
# numbers; a root set drawn from test would flatter FLTrust, not ours.
import torch as _torch                                   # noqa: E402
from rl_flwr.strategy import FLTrustCustomFedAvg as _FLTrustCustomFedAvg  # noqa: E402

FLTRUST_N_ROOT     = 200     # server-side clean examples
FLTRUST_BATCH      = 32
FLTRUST_SEED       = 42
FLTRUST_SERVER_LR  = 1.0
FLTRUST_ROOT_LR    = 0.01


def _make_fltrust_root_loader(dataset_name: str,
                              n_root: int = FLTRUST_N_ROOT,
                              batch_size: int = FLTRUST_BATCH,
                              seed: int = FLTRUST_SEED) -> DataLoader:
    """A clean root loader for `dataset_name`, drawn from the TRAIN split.

    Mirrors the dataset handling of _make_evaluate_fn so the tensors FLTrust
    trains on are identical in shape and normalisation to the client updates.
    """
    import numpy as _np
    cfg = DATASET_REGISTRY[dataset_name]
    _, eval_transform = build_transforms(dataset_name)
    train_split = load_dataset(cfg["hf_path"])["train"]
    if cfg["image_key"] != "image":
        train_split = train_split.rename_column(cfg["image_key"], "image")
    idx = _np.random.default_rng(seed).choice(
        len(train_split), size=min(n_root, len(train_split)), replace=False
    )
    rootset = train_split.select(idx).with_transform(
        make_eval_transform_fn(eval_transform)
    )
    return DataLoader(rootset, batch_size=batch_size, shuffle=True)


def _fltrust_factory(**kw):
    """Build FLTrustCustomFedAvg for the dataset of this run."""
    dataset_name = kw.pop("dataset_name")
    kw.pop("anom_decay", None)          # not an FLTrust parameter
    kw.pop("tau_s_quantile", None)      # ditto -- q_s is inert here
    return _FLTrustCustomFedAvg(
        model_fn=lambda: get_net(dataset_name, cifar_model=CIFAR_MODEL),
        root_loader=_make_fltrust_root_loader(dataset_name),
        loss_fn=_torch.nn.CrossEntropyLoss(),
        optimizer_fn=lambda m: _torch.optim.SGD(
            m.parameters(), lr=FLTRUST_ROOT_LR, momentum=0.9
        ),
        server_lr=FLTRUST_SERVER_LR,
        layer_filter=None,
        **kw,
    )


_fltrust_factory.wants_dataset = True
DEFENSE_REGISTRY["fltrust"] = _fltrust_factory

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP GRID  –  edit here
# ══════════════════════════════════════════════════════════════════════════════
DATASET_MODES = ["mnist", "fashion_mnist"]
# DATASET_MODES = ["fashion_mnist"]        # single dataset to save time
# DATASET_MODES = ["cifar10"]

# The ablation ladder. absonly -> ustad_af isolates the gap branch;
# meansvd -> ustad_af isolates the robust SVD fit.
# DEFENSE_MODES = ["fltrust", "fldetector", "multikrum", ]
# DEFENSE_MODES = ["ustad_af_absonly", "ustad_af", "ustad_af_meansvd"]
DEFENSE_MODES = ["multikrum"]   

# Attack de-correlation: target max pairwise cosine among malicious payloads.
# None reproduces the published LIE/Min-Max behaviour (identical vectors,
# cosine 1.0). 0.90 sits just below USTAD-FL's clone threshold tau_c = 0.95.






DECORRELATE_COS = None

# q_s only gates ConvLSTM re-scoring frequency; it is NOT the filter threshold,
# and in this strategy the quantile branch is inert, so a single operating point
# is enough for a before/after comparison against the tau=0.6 FDR table.
# TAU_VALUES    = [0.6]
TAU_VALUES  = [round(t * 0.1, 1) for t in range(11)]   # [0.0, 0.1, …, 1.0]

# "no_attack" is the control for the drop-floor claim (A) — keep it in.
ATTACK_MODES  = ["fang"]

# ATTACK_MODES  = ["gaussian", "lie", "min_max", "no_attack"]
# Per-attack tau list: "no_attack" ignores tau (irrelevant without an attack) → one run;
# any attack not listed here falls back to the full TAU_VALUES sweep.
TAU_BY_ATTACK = {"no_attack": [TAU_VALUES[0]]}

# q_s (tau_s_quantile) gates how often the ConvLSTM re-scores clients. Only the
# USTAD-FL family has a ConvLSTM, so the knob is INERT for every other defense --
# sweeping it there produces byte-identical runs. Restrict the sweep to the
# defenses it actually affects; everyone else runs once at BASELINE_TAU.
#
# CAL_DEFENSES sweep q_s too. It is NOT inert for them: it selects which clients
# the ConvLSTM re-scores after the cold-start window (strategy.py:3721 takes the
# q_s-quantile of the EWMA scores as the re-scoring trigger), and ustad_cal
# consumes the temporal arm's output through the calibrated boost fusion. The
# knob arguably matters MORE here than for ustad_af, where the arm's
# contribution was being destroyed by the max-fusion defect.
TAU_SWEEP_DEFENSES = {"ustad"} | set(AF_DEFENSES) | set(CAL_DEFENSES)
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
    _strategy_kwargs = dict(
        num_malicious=NUM_MALICIOUS,
        warmup_rounds=WARMUP_ROUNDS,
        attack_mode=attack_mode,
        anom_decay=ANOM_DECAY,
        tau_s_quantile=tau,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
    )
    # Factories that must build dataset-specific state (FLTrust's root loader
    # and model_fn) opt in via .wants_dataset; everything else is unaffected.
    if getattr(strategy_cls, "wants_dataset", False):
        _strategy_kwargs["dataset_name"] = dataset_name
    strategy = strategy_cls(**_strategy_kwargs)
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
