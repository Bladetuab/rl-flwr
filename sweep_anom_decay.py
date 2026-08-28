"""
Sweep tau_s_quantile in [0, 0.1, ..., 1.0] with anom_decay fixed at 0.9.

Plots:
  - Left  y-axis : Final-round server accuracy
  - Right y-axis : Total ConvLSTM forward passes
  - q = 1.0 highlighted as the "EWMA-trigger OFF" baseline

Usage (from inside rl-flwr/):
    python sweep_anom_decay.py

Results saved to  sweep_tau_results.json  (resume-safe).
Plot   saved to   sweep_tau_quantile.png.
"""

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAU_VALUES: list[float] = [round(float(v), 1) for v in np.arange(0.0, 1.01, 0.1)]
FIXED_ANOM_DECAY = 0.8

OUTPUTS_DIR  = Path(__file__).parent / "outputs"
RESULTS_FILE = Path(__file__).parent / "sweep_tau_results.json"
PLOT_FILE    = Path(__file__).parent / "sweep_tau_quantile.png"
FLWR_RUN_TIMEOUT = 7200  # seconds per run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def existing_result_dirs() -> set[Path]:
    return {p.parent for p in OUTPUTS_DIR.glob("**/results.json")}


def wait_for_new_result_dir(before: set[Path], timeout: int = 120) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = existing_result_dirs() - before
        if new:
            return sorted(new)[-1]
        time.sleep(10)
    return None


def parse_final_accuracy(result_dir: Path) -> float:
    with open(result_dir / "results.json") as f:
        data = json.load(f)
    return float(data[-1]["evaluate_metrics_serverapp"]["accuracy"])


def parse_total_convlstm_forwards(result_dir: Path) -> int:
    cost_path = result_dir / "convlstm_cost.csv"
    if not cost_path.exists():
        return 0
    total = 0
    with open(cost_path, newline="") as f:
        for row in csv.DictReader(f):
            total += int(row["n_scored"])
    return total


def _find_flwr() -> str:
    venv_flwr = Path(__file__).parent.parent / "get" / "bin" / "flwr"
    if venv_flwr.exists():
        return str(venv_flwr)
    on_path = shutil.which("flwr")
    if on_path:
        return on_path
    raise RuntimeError("Cannot find the 'flwr' binary.")


FLWR_BIN = _find_flwr()


def run_one(tau_q: float) -> dict:
    print(f"\n{'='*60}")
    print(f"  tau_s_quantile = {tau_q:.1f}   (anom_decay fixed = {FIXED_ANOM_DECAY})")
    print(f"{'='*60}\n")

    before = existing_result_dirs()

    cmd = [
        FLWR_BIN, "run", ".", "local-sim-gpu",
        "--run-config",
        f"tau-s-quantile={tau_q} anom-decay={FIXED_ANOM_DECAY}",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).parent, timeout=FLWR_RUN_TIMEOUT)

    if proc.returncode != 0:
        print(f"[WARNING] flwr run exited with code {proc.returncode} for tau_q={tau_q}")

    result_dir = wait_for_new_result_dir(before)
    if result_dir is None:
        print(f"[WARNING] No new result dir found for tau_q={tau_q}")
        return {
            "tau_s_quantile": tau_q,
            "accuracy": float("nan"),
            "total_convlstm_forwards": 0,
        }

    accuracy  = parse_final_accuracy(result_dir)
    total_fwd = parse_total_convlstm_forwards(result_dir)

    print(f"  -> result_dir       : {result_dir}")
    print(f"  -> final accuracy   : {accuracy:.4f}")
    print(f"  -> total fwd passes : {total_fwd}")

    return {
        "tau_s_quantile": tau_q,
        "accuracy": accuracy,
        "total_convlstm_forwards": total_fwd,
        "result_dir": str(result_dir),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(records: list[dict]) -> None:
    records  = sorted(records, key=lambda r: r["tau_s_quantile"])
    x        = [r["tau_s_quantile"] for r in records]
    acc      = [r["accuracy"] for r in records]
    fwd      = [r["total_convlstm_forwards"] for r in records]

    color_acc = "#2563EB"   # blue
    color_fwd = "#DC2626"   # red
    color_bl  = "#16A34A"   # green  — baseline marker

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ---- accuracy (left axis) ----
    ax1.plot(x, acc, marker="o", linewidth=2, color=color_acc, label="Final Accuracy", zorder=3)
    ax1.set_xlabel("tau_s_quantile (q)", fontsize=13)
    ax1.set_ylabel("Final-Round Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # ---- forwards (right axis) ----
    ax2 = ax1.twinx()
    ax2.plot(x, fwd, marker="s", linewidth=2, linestyle="--",
             color=color_fwd, label="Total ConvLSTM Forwards", zorder=3)
    ax2.set_ylabel("Total ConvLSTM Forward Passes", fontsize=12, color=color_fwd)
    ax2.tick_params(axis="y", labelcolor=color_fwd)

    # ---- highlight q = 1.0 baseline ----
    baseline_idx = next((i for i, r in enumerate(records) if r["tau_s_quantile"] == 1.0), None)
    if baseline_idx is not None:
        bx  = x[baseline_idx]
        bfwd = fwd[baseline_idx]
        bacc = acc[baseline_idx]

        # vertical dashed line
        ax1.axvline(bx, color=color_bl, linestyle=":", linewidth=1.6, alpha=0.7)

        # baseline forward-pass annotation on right axis
        ax2.annotate(
            f"EWMA trigger OFF\n(q=1.0 baseline)\n{bfwd} forwards",
            xy=(bx, bfwd),
            xytext=(-90, -40),
            textcoords="offset points",
            fontsize=8,
            color=color_bl,
            arrowprops=dict(arrowstyle="->", color=color_bl, lw=1.2),
        )

        # extra marker on both lines
        ax1.scatter([bx], [bacc], color=color_bl, s=80, zorder=5)
        ax2.scatter([bx], [bfwd], color=color_bl, s=80, zorder=5)

    # ---- accuracy value labels ----
    for xi, yi in zip(x, acc):
        if not np.isnan(yi):
            ax1.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=color_acc,
            )

    # ---- EWMA marginal-cost annotation (cost above baseline) ----
    if baseline_idx is not None:
        base_fwd = fwd[baseline_idx]
        for xi, fi in zip(x, fwd):
            delta = fi - base_fwd
            if delta > 0:
                ax2.annotate(
                    f"+{delta}",
                    xy=(xi, fi),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=color_fwd,
                    alpha=0.75,
                )

    # ---- legend ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    baseline_handle = Line2D([0], [0], color=color_bl, linestyle=":", linewidth=1.6,
                             label="q=1.0 baseline (trigger OFF)")
    ax1.legend(lines1 + lines2 + [baseline_handle],
               labels1 + labels2 + ["q=1.0 baseline (trigger OFF)"],
               loc="lower left", fontsize=9)

    # plt.title(
    #     f"Impact of EWMA Trigger (tau_s_quantile) on Accuracy & Compute Cost\n"
    #     f"(anom_decay fixed = {FIXED_ANOM_DECAY})",
    #     fontsize=13, fontweight="bold",
    # )
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"\nPlot saved -> {PLOT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records: list[dict] = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            records = json.load(f)
        done = {r["tau_s_quantile"] for r in records}
        print(f"Loaded {len(records)} existing results: {sorted(done)}")
    else:
        done = set()

    for q in TAU_VALUES:
        if q in done:
            print(f"  Skipping tau_s_quantile={q:.1f} (already done)")
            continue

        rec = run_one(q)
        records.append(rec)

        with open(RESULTS_FILE, "w") as f:
            json.dump(records, f, indent=2)

    print("\n\nAll runs complete. Generating plot...")
    make_plot(records)


if __name__ == "__main__":
    main()
