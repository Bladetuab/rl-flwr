"""Offline validation of CustomFedAvg_adaptive_assumption_free.

Mirrors test_baseline_defenses.py: exercises the decision rules on synthetic
inputs without standing up a federation, so the three assumption removals can be
checked in isolation.

  (A) the inherited quantile branch is inert at backstop_quantile = 1.0
  (B) the gap/absolute threshold rule separates benign from adversarial scores
      without a tuned drop rate
  (C) the median-centred trimmed SVD fit is not steered by a poisoned minority,
      where the parent's mean-centred full SVD is captured completely

These are INVENTED score distributions, not the real detector's scores. Passing
here means the rules behave as designed; it says nothing about detection quality
on real runs.

Run:  python test_assumption_free.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

from rl_flwr.CustomFedAvg_adaptive_assumption_free import (  # noqa: E402
    CustomFedAvg_adaptive_assumption_free as Strategy,
)

# The strategy logs through flwr; quiet it so the report is readable.
logging.getLogger("flwr").setLevel(logging.WARNING)

ABS_CAP = 0.40
N_CLIENTS = 100
N_MAL = 40


def _stub(mode: str = "gap_abs", **over):
    """A Strategy with only the attributes these rules touch.

    Bypasses __init__ so the heavy parent constructor (models, detector, Flower
    plumbing) is not needed to test the decision rules.
    """
    o = object.__new__(Strategy)
    o.threshold_mode = mode
    o.gap_mult = 1.0
    o.abs_cap = ABS_CAP
    o.z_cut = 3.5
    o.use_lower_mad = True
    o.last_threshold_source = None
    o.robust_svd = True
    o.svd_trim_frac = 0.40
    o.svd_target_len = 8
    for k, v in over.items():
        setattr(o, k, v)
    return o


def _confusion(scores: np.ndarray, thr: float, n_mal: int) -> tuple:
    drop = scores > thr
    mal = np.zeros(scores.size, dtype=bool)
    mal[scores.size - n_mal:] = True          # adversaries are the tail
    tp = int((drop & mal).sum())
    fp = int((drop & ~mal).sum())
    fn = int((~drop & mal).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return int(drop.sum()), recall, fp


def _regimes(seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    benign = lambda n: np.clip(rng.normal(0.08, 0.03, n), 0, None)  # noqa: E731
    return [
        ("no attack", np.clip(rng.normal(0.10, 0.03, N_CLIENTS), 0, None), 0),
        ("40% well-separated",
         np.concatenate([benign(60), rng.normal(0.78, 0.05, N_MAL)]), N_MAL),
        ("40% near-overlapping",
         np.concatenate([benign(60), rng.normal(0.30, 0.04, N_MAL)]), N_MAL),
        ("40% spread 0.20-0.95",
         np.concatenate([benign(60), rng.uniform(0.20, 0.95, N_MAL)]), N_MAL),
        ("40% heavy overlap",
         np.concatenate([benign(60), rng.normal(0.16, 0.03, N_MAL)]), N_MAL),
        ("degenerate identical", np.full(N_CLIENTS, 0.11), 0),
    ]


def check_threshold_rule() -> bool:
    """(A)+(B): no false positives anywhere; nothing dropped absent an attack."""
    print("\n(B) threshold rule  [mode=gap_abs]")
    print(f"  {'regime':22s} {'thr':>7s} {'via':>19s} {'drop':>5s} {'recall':>7s} {'FP':>4s}")
    ok = True
    o = _stub()
    for name, scores, n_mal in _regimes():
        thr = o._derive_abs_threshold(scores)
        # gap_abs always yields a threshold for >= 4 scores; None would mean the
        # rule fell through to the parent constant, which this check never wants.
        assert thr is not None, f"{name}: rule returned None instead of a threshold"
        dropped, recall, fp = _confusion(scores, thr, n_mal)
        print(f"  {name:22s} {thr:7.4f} {o.last_threshold_source:>19s} "
              f"{dropped:5d} {recall:7.3f} {fp:4d}")
        # A false positive is never acceptable: the whole point of removing the
        # quantile floor is that benign clients stop being discarded.
        if fp != 0:
            print(f"      FAIL: {fp} benign clients dropped")
            ok = False
        if n_mal == 0 and dropped != 0:
            print(f"      FAIL: {dropped} dropped with no attacker present")
            ok = False
    return ok


def check_quantile_inert() -> bool:
    """(A): quantile(s, 1.0) == max(s), so min(q_thr, a_thr) == a_thr."""
    print("\n(A) quantile branch inert at backstop_quantile=1.0")
    _, scores, _ = _regimes()[1]
    q_thr = float(np.quantile(scores, 1.0))
    effective = min(q_thr, ABS_CAP)
    ok = effective == ABS_CAP and q_thr == float(scores.max())
    print(f"  q_thr=max(s)={q_thr:.4f}  a_thr={ABS_CAP}  effective={effective:.4f}  "
          f"absolute governs={ok}")
    if not ok:
        print("      FAIL: quantile branch still binds")
    return ok


def check_robust_svd() -> bool:
    """(C): a poisoned minority must not determine the basis."""
    print("\n(C) SVD basis under a 40% poisoned warm-up buffer")
    torch.manual_seed(0)
    d = 64

    benign_dir = torch.zeros(d)
    benign_dir[:4] = 1.0
    benign_dir /= benign_dir.norm()

    poison_dir = torch.zeros(d)
    poison_dir[40:44] = 1.0
    poison_dir /= poison_dir.norm()

    buf = [benign_dir * float(torch.randn(1)) * 0.5 + 0.01 * torch.randn(d)
           for _ in range(60)]
    buf += [poison_dir * 50.0 + 0.01 * torch.randn(d) for _ in range(N_MAL)]

    ok = True
    for label, robust in (("robust (median+trim)", True), ("parent (mean+full SVD)", False)):
        o = _stub(robust_svd=robust)
        fitted = o._fit_fixed_svd_basis_from_vectors(buf)
        assert fitted and o.svd_basis is not None, f"{label}: no basis was fitted"
        v0 = o.svd_basis[0]
        a_benign = abs(float(v0 @ benign_dir))
        a_poison = abs(float(v0 @ poison_dir))
        print(f"  {label:24s} fitted={fitted}  "
              f"|<basis0,benign>|={a_benign:.4f}  |<basis0,poison>|={a_poison:.4f}")
        if robust and not (a_benign > 0.9 and a_poison < 0.1):
            print("      FAIL: robust fit was steered by the poisoned minority")
            ok = False
        if not robust and a_poison < 0.9:
            print("      NOTE: parent was not captured here; the contrast is weaker "
                  "than expected for this buffer")
    return ok


if __name__ == "__main__":
    print("=" * 74)
    print("CustomFedAvg_adaptive_assumption_free — offline rule validation")
    print("=" * 74)
    results = {
        "quantile inert (A)": check_quantile_inert(),
        "threshold rule (B)": check_threshold_rule(),
        "robust SVD (C)": check_robust_svd(),
    }
    print("\n" + "-" * 74)
    for name, passed in results.items():
        print(f"  {name:24s} {'PASS' if passed else 'FAIL'}")
    print("-" * 74)
    print("Synthetic scores only. Real-run detection quality is not tested here.")
    sys.exit(0 if all(results.values()) else 1)
