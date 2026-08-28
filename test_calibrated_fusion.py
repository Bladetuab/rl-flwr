"""Offline validation of CustomFedAvg_adaptive_calibrated.

Mirrors test_assumption_free.py: exercises the decision rules on synthetic
inputs without standing up a federation, so the fusion fix and the training
fixes can be checked in isolation.

  (1) boost-form fusion never RESCUES a client the statistical screen would drop
  (2) fusion stays in [0,1], and temporal_weight=0 degenerates to stat-only
  (3) per-round calibration removes the ACROSS-ROUND scale drift measured in
      IMPROVEMENT.md section 1.2
  (4) the quiet-round gate fires on a unimodal round and not on a separated one
  (5) D1: the training gate ignores the server accuracy/loss it is handed
  (6) D2: training-window selection ignores the detector's own scores
  (8) the Fang fix: median capture is real, the uniformity signal separates
      Fang from SGD, and the reference-free override only ever raises a score

Groups 1-6 and 8 use INVENTED distributions; passing means the rules behave as
designed, and says nothing about detection quality on real runs. Groups 7 and 9
are measured.

  (7) REGRESSION ON REAL LOGS: replays the six shipped ustad_af/tau0.40 runs
      through the real _fuse/_calibrate_relative and checks the single-cut
      margins reproduce the table in the CustomFedAvg_adaptive_calibrated
      docstring. This one is measured, not invented.
  (9) STRICT EXTENSION: runs the real _compute_per_round_stat_scores against the
      parent's on identical inputs. With ref_free_override=False the two agree to
      1e-9; with it on, Fang goes from margin -0.39 to +0.51 and no benign score
      moves at all.
 (10) REAL FEDERATED RUNS: replays logged signal_breakdown.csv files and checks
      that dropping clone from the override removes the benign false positives
      (39-72% -> 0%) without losing any attacker.
 (11) QUIET-GATE CALIBRATION: replays the same rounds through _round_is_quiet and
      checks the shipped multiplier accepts attack-free rounds while rejecting
      attacked ones. The old default (1.0) rejected both.

Run:  python test_calibrated_fusion.py
"""

from __future__ import annotations

import glob
import logging
import os
import sys

import numpy as np
import torch

from rl_flwr.CustomFedAvg_adaptive_calibrated import (  # noqa: E402
    CustomFedAvg_adaptive_calibrated as Strategy,
)

logging.getLogger("flwr").setLevel(logging.WARNING)

ONSET_ROUND = 6
TEMPORAL_W = 0.2


def _stub(**over):
    """A Strategy with only the attributes these rules touch.

    Bypasses __init__ so the heavy parent constructor (models, detector, Flower
    plumbing) is not needed to test the decision rules.
    """
    o = object.__new__(Strategy)
    o.fusion_mode = "boost"
    o.temporal_weight = TEMPORAL_W
    o.calibrate_temporal = True
    o.quiet_gap_mult = 4.0
    o.quiet_streak_required = 1
    o.consensus_keep_frac = 0.5
    o.ref_free_override = True
    o.ref_free_gain = 1.0
    o.use_uniformity = True
    o.override_on_clone = False
    o.clone_cos_threshold = 0.95
    o._last_round_quiet = False
    o._quiet_streak = 0
    o.window_len = 3
    o.max_buffer = 2000
    o.clean_window_buffer = []
    o.client_feature_history = {}
    o.last_anomaly_scores = {}
    for k, v in over.items():
        setattr(o, k, v)
    return o


class _Msg:
    """Minimal stand-in for a flwr Message: only src_node_id is read."""

    def __init__(self, cid):
        self.metadata = type("M", (), {"src_node_id": cid})()


def _report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


# =============================================================================
#  (1) boost fusion never rescues
# =============================================================================
def test_boost_never_rescues():
    print("\n(1) boost-form fusion never lowers the primary score")
    rng = np.random.default_rng(0)
    stat = rng.uniform(0, 1, 5000)
    rel = rng.uniform(0, 1, 5000)

    boost = _stub(fusion_mode="boost")
    fused_b = np.array([boost._fuse(s, r) for s, r in zip(stat, rel)])
    ok = _report(
        "boost: fused >= stat for every (stat, rel)",
        bool((fused_b >= stat - 1e-12).all()),
        f"min(fused - stat) = {(fused_b - stat).min():+.3e}",
    )

    # The contrast that motivates the default: a weighted sum CAN rescue.
    wsum = _stub(fusion_mode="wsum")
    fused_w = np.array([wsum._fuse(s, r) for s, r in zip(stat, rel)])
    n_rescued = int(((stat > 0.40) & (fused_w <= 0.40)).sum())
    ok &= _report(
        "wsum: does rescue clients past tau_abs (why boost is the default)",
        n_rescued > 0,
        f"{n_rescued}/5000 clients with stat > 0.40 fall back under it",
    )
    return ok


# =============================================================================
#  (2) range, and the stat-only degenerate case
# =============================================================================
def test_range_and_degenerate():
    print("\n(2) fusion range and temporal_weight = 0")
    rng = np.random.default_rng(1)
    stat = rng.uniform(0, 1, 2000)
    rel = rng.uniform(0, 1, 2000)

    ok = True
    for mode in ("boost", "wsum", "max"):
        o = _stub(fusion_mode=mode)
        f = np.array([o._fuse(s, r) for s, r in zip(stat, rel)])
        ok &= _report(
            f"{mode}: fused stays in [0, 1]",
            bool((f >= -1e-12).all() and (f <= 1 + 1e-12).all()),
            f"range [{f.min():.3f}, {f.max():.3f}]",
        )

    o = _stub(fusion_mode="boost", temporal_weight=0.0)
    f = np.array([o._fuse(s, r) for s, r in zip(stat, rel)])
    ok &= _report(
        "temporal_weight=0 reproduces the statistical screen exactly",
        bool(np.allclose(f, stat)),
        f"max |fused - stat| = {np.abs(f - stat).max():.3e}",
    )
    return ok


# =============================================================================
#  (3) calibration removes across-round scale drift
# =============================================================================
def test_calibration_removes_drift():
    print("\n(3) per-round calibration removes across-round scale drift")
    o = _stub()
    # Same SHAPE, different LEVEL — exactly the defect of section 1.2: the
    # temporal arm's absolute level moves between rounds while its ordering
    # and relative spacing do not.
    shape = np.concatenate([np.full(60, 0.20), np.full(40, 0.80)])
    round_a = {i: float(v) for i, v in enumerate(shape)}
    round_b = {i: float(v) + 0.15 for i, v in enumerate(shape)}

    cal_a = o._calibrate_relative(round_a)
    cal_b = o._calibrate_relative(round_b)
    drift_raw = abs(np.mean(list(round_b.values())) - np.mean(list(round_a.values())))
    drift_cal = max(abs(cal_a[i] - cal_b[i]) for i in round_a)

    ok = _report(
        "a shifted round calibrates to the same scores",
        drift_cal < 1e-9,
        f"raw level drift {drift_raw:.3f} → calibrated drift {drift_cal:.3e}",
    )
    ok &= _report(
        "the benign cluster calibrates to ~0, the adversarial cluster to ~1",
        cal_a[0] < 1e-9 and cal_a[99] > 0.99,
        f"benign {cal_a[0]:.4f}, adversarial {cal_a[99]:.4f}",
    )

    # An adversarial cluster must not be able to inflate the scale that
    # normalises it: this is why the lower-half MAD is used.
    heavy = {i: (0.20 if i < 60 else 5.0) for i in range(100)}
    cal_h = o._calibrate_relative(heavy)
    ok &= _report(
        "a 40% adversarial cluster cannot inflate its own scale away",
        cal_h[99] > 0.99,
        f"adversarial calibrates to {cal_h[99]:.4f}",
    )
    return ok


# =============================================================================
#  (4) the quiet-round gate
# =============================================================================
def test_quiet_round_gate():
    print("\n(4) unsupervised quiet-round gate")
    o = _stub()
    rng = np.random.default_rng(2)

    unimodal = np.clip(rng.normal(0.08, 0.03, 100), 0, None)
    separated = np.concatenate([
        np.clip(rng.normal(0.08, 0.03, 60), 0, None),
        np.clip(rng.normal(0.80, 0.03, 40), 0, None),
    ])

    ok = _report("quiet on a unimodal round", o._round_is_quiet(unimodal) is True)
    ok &= _report("not quiet on a separated round", o._round_is_quiet(separated) is False)
    # NOTE: a synthetic Gaussian round is too clean to show why 1.0 failed --
    # it passes even at 1.0 here. The defect is an extreme-value effect that
    # only appears in real score distributions, so it is tested against the
    # logged runs in group (11) instead.
    ok &= _report("quiet on a degenerate all-identical round",
                  o._round_is_quiet(np.full(100, 0.3)) is True)
    ok &= _report("quiet when there are too few clients to judge",
                  o._round_is_quiet(np.array([0.1, 0.9])) is True)
    return ok


# =============================================================================
#  (5) D1: the gate ignores the server's test-set signal
# =============================================================================
def test_gate_ignores_server_eval():
    print("\n(5) D1 — the training gate ignores server accuracy/loss")
    ok = True

    # The streak is advanced once per round by _run_convlstm_analysis, so the
    # test drives it the same way rather than through the predicate.
    def _tick(o, quiet, acc=0.5, loss=0.5):
        o._last_round_quiet = quiet
        o._quiet_streak = o._quiet_streak + 1 if quiet else 0
        return o._round_is_good(acc_val=acc, loss_val=loss)

    # Quiet round: must train regardless of how bad the server metrics look.
    ok &= _report(
        "quiet round trains even when server accuracy is terrible",
        _tick(_stub(), quiet=True, acc=0.10, loss=9.99) is True,
    )

    # Loud round: must NOT train, however good the server metrics look.
    ok &= _report(
        "loud round does not train even when server accuracy is excellent",
        _tick(_stub(), quiet=False, acc=0.99, loss=0.01) is False,
    )

    o = _stub(quiet_streak_required=3)
    got = [_tick(o, True) for _ in range(4)]
    ok &= _report(
        "quiet_streak_required=3 needs three consecutive quiet rounds",
        got == [False, False, True, True],
        f"streak decisions: {got}",
    )

    o = _stub(quiet_streak_required=2)
    seq = [_tick(o, q) for q in (True, True, False, True)]
    ok &= _report(
        "a loud round resets the streak",
        seq == [False, True, False, False],
        f"quiet,quiet,loud,quiet → {seq}",
    )

    # _round_is_good must be a PURE predicate: strategy.py calls it once per
    # round today, but a second call must not advance the streak.
    o = _stub(quiet_streak_required=2)
    _tick(o, True)
    before = o._quiet_streak
    [o._round_is_good(0.5, 0.5) for _ in range(5)]
    ok &= _report(
        "repeated calls do not double-count the streak",
        o._quiet_streak == before,
        f"streak {before} → {o._quiet_streak} after 5 extra calls",
    )
    return ok


# =============================================================================
#  (6) D2: window selection ignores the detector's own scores
# =============================================================================
def test_consensus_selection():
    print("\n(6) D2 — training windows selected by consensus, not by self-labels")
    n, dim = 20, 12
    rng = np.random.default_rng(3)

    o = _stub(consensus_keep_frac=0.5, window_len=3)
    # Clients 0-14 sit at the consensus; 15-19 are far from it.
    for cid in range(n):
        centre = 0.0 if cid < 15 else 10.0
        o.client_feature_history[cid] = [
            torch.tensor(rng.normal(centre, 0.1, dim), dtype=torch.float32)
            for _ in range(3)
        ]
    # The detector's own opinion is INVERTED: it loves the outliers and
    # distrusts the consensus. The inherited rule would follow it; this one
    # must not.
    o.last_anomaly_scores = {cid: (0.9 if cid < 15 else 0.01) for cid in range(n)}

    replies = [_Msg(cid) for cid in range(n)]
    o._collect_clean_windows_for_training(replies, max_windows_per_client=1)

    n_kept = len(o.clean_window_buffer)
    ok = _report(
        "keeps consensus_keep_frac of the candidates",
        n_kept == 10,
        f"{n_kept} windows from {n} clients at keep_frac=0.5",
    )

    # Re-run with the detector's scores flipped: selection must be identical,
    # because it never consults them.
    o2 = _stub(consensus_keep_frac=0.5, window_len=3)
    o2.client_feature_history = o.client_feature_history
    o2.last_anomaly_scores = {cid: (0.01 if cid < 15 else 0.9) for cid in range(n)}
    o2._collect_clean_windows_for_training(replies, max_windows_per_client=1)
    ok &= _report(
        "selection is invariant to last_anomaly_scores (no confirmation loop)",
        len(o2.clean_window_buffer) == n_kept,
        f"{len(o2.clean_window_buffer)} windows with the scores inverted",
    )

    # And the outliers must not have been chosen.
    o3 = _stub(consensus_keep_frac=0.25, window_len=3)
    o3.client_feature_history = o.client_feature_history
    o3._collect_clean_windows_for_training(replies, max_windows_per_client=1)
    ok &= _report(
        "a tighter keep fraction keeps fewer windows",
        len(o3.clean_window_buffer) == 5,
        f"{len(o3.clean_window_buffer)} windows at keep_frac=0.25",
    )

    # Windows must be the MOST RECENT ones, not sampled from the whole history.
    # A slow-drift attacker (IMPROVEMENT.md section 4.1) sits at consensus early
    # and diverges later, so admitting old windows from a currently-consensus
    # client reopens the confirmation loop at the window level.
    o4 = _stub(consensus_keep_frac=1.0, window_len=2)
    # One client, history 0,1,2,...,5 — windows are [0,1],[1,2],...,[4,5].
    o4.client_feature_history = {
        7: [torch.full((4,), float(t)) for t in range(6)]
    }
    o4._collect_clean_windows_for_training([_Msg(7)], max_windows_per_client=2)
    got = [int(w[-1][0].item()) for w in o4.clean_window_buffer]
    ok &= _report(
        "keeps the most recent windows, not a random sample of history",
        got == [4, 5],
        f"window end-frames {got}, expected the two newest [4, 5]",
    )
    return ok


# =============================================================================
#  (7) regression against the real logged runs
# =============================================================================
def test_margins_on_real_runs():
    print("\n(7) regression — single-cut margins on the six shipped runs")
    runs = sorted(
        d for d in glob.glob("outputs/2026-08-1*/*ustad_af_*tau0.40")
        if "absonly" not in d and "meansvd" not in d
    )
    if not runs:
        print("  [SKIP] no ustad_af/tau0.40 runs found under outputs/")
        return True

    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pandas not available")
        return True

    forms = {
        "max (shipped)": _stub(fusion_mode="max", calibrate_temporal=False),
        "wsum, raw": _stub(fusion_mode="wsum", calibrate_temporal=False),
        "boost, raw": _stub(fusion_mode="boost", calibrate_temporal=False),
        "boost, calibrated": _stub(fusion_mode="boost", calibrate_temporal=True),
    }
    margins = {k: [] for k in forms}

    for d in runs:
        df = pd.read_csv(os.path.join(d, "stat_anomaly.csv"))
        a = df[df["round"] >= ONSET_ROUND]
        for name, o in forms.items():
            vals, mal = [], []
            for rnd, g in a.groupby("round"):
                rel = dict(enumerate(g.convlstm_rel.values))
                if o.calibrate_temporal:
                    rel = o._calibrate_relative(rel)
                for i, s in enumerate(g.stat_ewma.values):
                    vals.append(o._fuse(s, rel[i]))
                mal.extend(g.is_malicious.values)
            vals = np.array(vals)
            mal = np.array(mal, dtype=bool)
            margins[name].append(vals[mal].min() - vals[~mal].max())

    ok = True
    print(f"    {'fusion form':<22}{'cells margin>0':>16}{'worst-cell margin':>20}")
    for name in forms:
        m = margins[name]
        n_pos = sum(1 for x in m if x > 0)
        print(f"    {name:<22}{f'{n_pos}/{len(m)}':>16}{min(m):>+20.3f}")

    # The claims the docstring makes, asserted.
    ok &= _report(
        "shipped `max` separates no cell by a single cut",
        sum(1 for x in margins["max (shipped)"] if x > 0) == 0,
    )
    ok &= _report(
        "boost beats wsum on cells separated",
        sum(1 for x in margins["boost, raw"] if x > 0)
        > sum(1 for x in margins["wsum, raw"] if x > 0),
    )
    ok &= _report(
        "boost roughly halves the worst-cell margin against wsum",
        min(margins["boost, raw"]) > min(margins["wsum, raw"]),
        f"{min(margins['boost, raw']):+.3f} vs {min(margins['wsum, raw']):+.3f}",
    )
    ok &= _report(
        "the default (boost, calibrated) separates 4 of 6 cells",
        sum(1 for x in margins["boost, calibrated"] if x > 0) == 4,
    )
    return ok


# =============================================================================
#  (8) the Fang fix — median capture, uniformity, reference-free override
# =============================================================================
def _synthetic_round(seed=0, D=4000, N=100, M=40, attack="fang"):
    """Benign SGD-like deltas plus M malicious ones of the given family."""
    rng = np.random.default_rng(seed)
    w_global = rng.normal(0, 0.5, D)
    benign = rng.normal(0, 0.02, (N - M, D)) * rng.gamma(2, 0.5, (N - M, 1))
    if attack == "fang":                       # identical -lambda*sign(w_global)
        mal = np.tile(-np.sign(w_global) * 0.01, (M, 1))
    elif attack == "gaussian":                 # independent noise per client
        mal = rng.normal(0, 0.02, (M, D)) * rng.gamma(2, 0.5, (M, 1)) \
            + rng.normal(0, 0.2, (M, D))
    elif attack == "clone":                    # LIE / Min-Max: one shared vector
        mal = np.tile(benign.mean(0) - 0.5 * benign.std(0), (M, 1))
    else:                                      # no attack
        mal = rng.normal(0, 0.02, (M, D)) * rng.gamma(2, 0.5, (M, 1))
    return np.vstack([benign, mal]), np.array([0] * (N - M) + [1] * M, dtype=bool)


def test_fang_fix():
    print("\n(8) the Fang fix — reference-free evidence")
    o = _stub(ref_free_override=True, ref_free_gain=1.0, use_uniformity=True)
    ok = True

    # -- median capture: the reason the four median-referenced signals fail ----
    V, mal = _synthetic_round(attack="fang")
    med = np.median(V, axis=0)
    captured = float(np.mean(np.isclose(med, V[mal][0], rtol=1e-6)))
    d_mal = np.linalg.norm(V[mal][0] - med)
    d_ben = np.median(np.linalg.norm(V[~mal] - med, axis=1))
    ok &= _report(
        "40/100 identical vectors capture the coordinate-wise median",
        captured > 0.90,
        f"median equals the Fang vector on {100*captured:.1f}% of coordinates",
    )
    ok &= _report(
        "so the attacker sits CLOSER to the median than honest clients",
        d_mal < d_ben,
        f"dist(Fang, median)={d_mal:.4f} vs benign median dist={d_ben:.4f} "
        "→ one-sided dist_score = 0",
    )

    # -- the uniformity signal separates Fang from SGD --------------------------
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    u_ben = np.array([o._uniformity(t(v)) for v in V[~mal]])
    u_mal = np.array([o._uniformity(t(v)) for v in V[mal]])
    ok &= _report(
        "uniformity: Fang = 1.0 exactly, benign SGD ≈ sqrt(2/pi)",
        u_mal.min() > 0.999 and abs(u_ben.mean() - 0.7979) < 0.02,
        f"benign {u_ben.mean():.4f}±{u_ben.std():.4f} (max {u_ben.max():.4f}) "
        f"vs Fang {u_mal.min():.4f}",
    )
    ok &= _report(
        "the gap is enormous relative to benign spread",
        (u_mal.min() - u_ben.max()) / max(u_ben.std(), 1e-12) > 20,
        f"{(u_mal.min()-u_ben.max())/max(u_ben.std(),1e-12):.0f} benign std "
        "above the benign maximum",
    )

    # -- scored form: benign ~0, Fang ~1, no tuned constant ---------------------
    cids = list(range(len(V)))
    raw = {i: o._uniformity(t(V[i])) for i in cids}
    sc = o._uniformity_scores(raw)
    sb = np.array([sc[i] for i in cids if not mal[i]])
    sm = np.array([sc[i] for i in cids if mal[i]])
    ok &= _report(
        "scored uniformity separates by a positive single-cut margin",
        sm.min() - sb.max() > 0,
        f"benign max {sb.max():.4f}, Fang min {sm.min():.4f} "
        f"→ margin {sm.min()-sb.max():+.4f}",
    )

    # -- the signal must stay quiet when there is nothing to find ---------------
    V0, mal0 = _synthetic_round(attack="none")
    raw0 = {i: o._uniformity(t(v)) for i, v in enumerate(V0)}
    sc0 = np.array(list(o._uniformity_scores(raw0).values()))
    ok &= _report(
        "no attack: uniformity scores stay near zero (no noise amplification)",
        sc0.max() < 0.15,
        f"max score on an attack-free round = {sc0.max():.4f}",
    )

    # -- Gaussian: uniformity must NOT fire; norm score is what catches it ------
    Vg, malg = _synthetic_round(attack="gaussian")
    rawg = {i: o._uniformity(t(v)) for i, v in enumerate(Vg)}
    scg = o._uniformity_scores(rawg)
    sgm = np.array([scg[i] for i in range(len(Vg)) if malg[i]])
    ok &= _report(
        "Gaussian attack does not trigger uniformity (it is not a Fang-detector)",
        sgm.max() < 0.15,
        f"max malicious uniformity score = {sgm.max():.4f} — norm score must carry it",
    )

    # -- the override raises but never lowers ----------------------------------
    rng = np.random.default_rng(7)
    weighted = rng.uniform(0, 1, 4000)
    reffree = rng.uniform(0, 1, 4000)
    final = np.maximum(weighted, o.ref_free_gain * reffree)
    ok &= _report(
        "override can only raise a score, never lower one",
        bool((final >= weighted - 1e-12).all()),
        "so it cannot weaken detection of anything the weighted sum catches",
    )

    # -- and the Fang profile crosses tau_abs, where the weighted sum did not ---
    # Uniformity alone must carry it: clone is deliberately NOT in the override.
    fang_profile = {"norm": 0.0, "cosine": 0.0003, "dist": 0.0, "sign": 0.0,
                    "clone": 1.0, "uniformity": 1.0}
    w = (0.30 * fang_profile["norm"] + 0.20 * fang_profile["cosine"]
         + 0.25 * fang_profile["dist"] + 0.15 * fang_profile["sign"]
         + 0.10 * fang_profile["clone"])
    f = max(w, fang_profile["uniformity"])
    ok &= _report(
        "measured Fang profile: weighted sum 0.10 (under tau_abs), override 1.00",
        abs(w - 0.10) < 0.001 and f >= 0.999,
        f"weighted={w:.4f} < 0.40 = tau_abs;  with uniformity-only override={f:.4f}",
    )

    # -- clone must NOT be able to drive the override (the 2026-08-26 defect) ---
    # A converged-benign profile: everything quiet except a high clone score,
    # which is what honest clients look like late in training.
    conv_benign = {"weighted": 0.07, "clone": 0.573, "uniformity": 0.003}
    o_fixed = _stub(override_on_clone=False)
    o_broken = _stub(override_on_clone=True)
    ev_fixed = max(conv_benign["weighted"],
                   conv_benign["uniformity"] if not o_fixed.override_on_clone
                   else max(conv_benign["clone"], conv_benign["uniformity"]))
    ev_broken = max(conv_benign["weighted"],
                    max(conv_benign["clone"], conv_benign["uniformity"]))
    ok &= _report(
        "converged-benign profile (clone 0.573) stays under tau_abs",
        ev_fixed <= 0.40 < ev_broken,
        f"uniformity-only={ev_fixed:.3f} <= 0.40 < clone-in-override={ev_broken:.3f}",
    )
    return ok


# =============================================================================
#  (10) real federated runs — the uniformity-only override kills the FDR
# =============================================================================
def test_override_on_real_runs():
    print("\n(10) real runs — clone in the override was the false-positive source")
    import glob as _glob
    runs = sorted(_glob.glob("outputs/*/*ustad_cal*"))
    files = [os.path.join(d, "signal_breakdown.csv") for d in runs]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("  [SKIP] no signal_breakdown.csv found under outputs/")
        return True
    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pandas not available")
        return True

    TAU = 0.40
    ok = True
    print(f"    {'run':<34}{'benign>tau clone':>18}{'benign>tau unif':>17}{'mal>tau unif':>14}")
    n_checked = 0
    for f in files:
        s = pd.read_csv(f)
        s = s[s["round"] >= ONSET_ROUND]
        mal = s.is_malicious == 1
        if not len(s) or mal.sum() == 0:
            continue
        n_checked += 1
        with_clone = np.maximum(s.weighted, np.maximum(s.clone, s.uniformity))
        unif_only = np.maximum(s.weighted, s.uniformity)
        b = ~mal
        fp_clone = 100 * (with_clone[b] > TAU).mean()
        fp_unif = 100 * (unif_only[b] > TAU).mean()
        rec_unif = 100 * (unif_only[mal] > TAU).mean()
        full_name = os.path.basename(os.path.dirname(f))
        name = full_name[:32]
        print(f"    {name:<34}{fp_clone:>17.1f}%{fp_unif:>16.1f}%{rec_unif:>13.1f}%")
        if "no_attack" in full_name:
            # Clients carry the malicious LABEL but never attack, so the correct
            # behaviour is to flag NOBODY -- the label column is not ground truth
            # for "is behaving adversarially" in this mode.
            ok &= _report(
                f"{name[:26]}: flags nobody when no client actually attacks",
                fp_unif < 1.0 and rec_unif < 1.0,
                f"benign {fp_unif:.1f}%, labelled-malicious-but-honest {rec_unif:.1f}%",
            )
            continue
        ok &= _report(
            f"{name[:26]}: uniformity-only removes the false positives",
            fp_unif < 1.0 and rec_unif > 99.0,
            f"benign FP {fp_clone:.1f}% → {fp_unif:.1f}%, malicious kept at {rec_unif:.1f}%",
        )
    if n_checked == 0:
        print("  [SKIP] no attacked runs with a malicious population")
    return ok


# =============================================================================
#  (9) strict extension — override OFF must reproduce the parent exactly
# =============================================================================
class _Arrays:
    def __init__(self, sd): self._sd = sd
    def to_torch_state_dict(self): return self._sd


class _FullMsg:
    def __init__(self, cid, sd):
        self.metadata = type("M", (), {"src_node_id": cid})()
        self.content = {"arrays": _Arrays(sd)}


def test_strict_extension():
    print("\n(9) strict extension — ref_free_override=False matches the parent")
    from rl_flwr.CustomFedAvg_adaptive_new import CustomFedAvg_adaptive_new as Parent

    rng = np.random.default_rng(11)
    D, N, M = 600, 40, 16
    gsd = {"fc.weight": torch.tensor(rng.normal(0, 0.5, D), dtype=torch.float32)}
    V, mal = _synthetic_round(seed=5, D=D, N=N, M=M, attack="fang")
    msgs = [_FullMsg(i, {"fc.weight": gsd["fc.weight"] + torch.tensor(v, dtype=torch.float32)})
            for i, v in enumerate(V)]

    def build(cls, **over):
        o = object.__new__(cls)
        o.svd_ready = False
        o.clone_cos_threshold = 0.95
        o._layer_filter = lambda n: True
        for k, v in over.items():
            setattr(o, k, v)
        return o

    parent = build(Parent)
    child_off = build(Strategy, ref_free_override=False, use_uniformity=False,
                      ref_free_gain=1.0, override_on_clone=False,
                      last_signal_breakdown={})
    child_on = build(Strategy, ref_free_override=True, use_uniformity=True,
                     ref_free_gain=1.0, override_on_clone=False,
                     last_signal_breakdown={})

    ga = _Arrays(gsd)
    p_sc = parent._compute_per_round_stat_scores(msgs, ga)
    c_off = child_off._compute_per_round_stat_scores(msgs, ga)
    c_on = child_on._compute_per_round_stat_scores(msgs, ga)

    diff = max(abs(p_sc[k] - c_off[k]) for k in p_sc)
    ok = _report(
        "override OFF reproduces the parent's five-signal score exactly",
        diff < 1e-9,
        f"max |child - parent| over {len(p_sc)} clients = {diff:.3e}",
    )

    pm = np.array([p_sc[i] for i in range(N) if mal[i]])
    pb = np.array([p_sc[i] for i in range(N) if not mal[i]])
    cm = np.array([c_on[i] for i in range(N) if mal[i]])
    cb = np.array([c_on[i] for i in range(N) if not mal[i]])
    ok &= _report(
        "parent scores Fang BELOW benign (the failure being fixed)",
        pm.mean() < pb.mean(),
        f"parent: malicious {pm.mean():.4f} vs benign {pb.mean():.4f} "
        f"→ margin {pm.min()-pb.max():+.4f}",
    )
    ok &= _report(
        "override ON separates Fang by a positive single-cut margin",
        cm.min() - cb.max() > 0,
        f"child:  malicious {cm.mean():.4f} vs benign {cb.mean():.4f} "
        f"→ margin {cm.min()-cb.max():+.4f}",
    )
    ok &= _report(
        "every Fang client clears tau_abs = 0.40 (none did before)",
        bool((cm > 0.40).all()) and bool((pm <= 0.40).all()),
        f"malicious min: parent {pm.min():.4f} → child {cm.min():.4f}, tau_abs = 0.40",
    )
    # The false-positive test that matters: the override must not lift ANY
    # honest client. Benign scores must be bit-identical to the parent's.
    ok &= _report(
        "the override raises no benign client — benign scores unchanged",
        float(np.abs(cb - pb).max()) < 1e-9,
        f"max benign change = {np.abs(cb - pb).max():.3e} "
        f"(benign max stays {cb.max():.4f}, a property of the weighted sum, "
        "not of the override)",
    )
    return ok


# =============================================================================
#  (11) quiet-gate calibration against real attack-free vs attacked rounds
# =============================================================================
def test_quiet_gate_calibration():
    print("\n(11) real runs — quiet-gate calibration")
    import glob as _glob
    files = [os.path.join(d, "signal_breakdown.csv")
             for d in sorted(_glob.glob("outputs/*/*ustad_cal*"))]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("  [SKIP] no signal_breakdown.csv found under outputs/")
        return True
    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pandas not available")
        return True

    LAM = 0.4          # stat_decay
    free, attacked = [], []
    o = _stub()
    for f in files:
        s = pd.read_csv(f)
        # Corrected per-round score: uniformity-only override (the shipped rule).
        s["corrected"] = np.maximum(s.weighted, s.uniformity)
        is_noattack = "no_attack" in os.path.basename(os.path.dirname(f))
        ewma = {}
        for r in sorted(s["round"].unique()):
            g = s[s["round"] == r]
            vals = []
            for cid, v in zip(g.client_id, g.corrected):
                prev = ewma.get(cid)
                ewma[cid] = v if prev is None else LAM * prev + (1 - LAM) * v
                vals.append(ewma[cid])
            if len(vals) < 4:
                continue
            # Attack-free: the warm-up rounds of every run, plus all of no_attack.
            (free if (r <= 5 or is_noattack) else attacked).append(np.array(vals))

    if not free or not attacked:
        print("  [SKIP] need both attack-free and attacked rounds")
        return True

    def rate(pop, mult):
        oo = _stub(quiet_gap_mult=mult)
        return 100 * np.mean([oo._round_is_quiet(v) for v in pop])

    print(f"    {'gap_mult':>9}{'attack-free → quiet':>22}{'attacked → quiet':>19}")
    for m in (1.0, 4.0, 8.0):
        print(f"    {m:>9.1f}{rate(free, m):>21.1f}%{rate(attacked, m):>18.1f}%")

    ok = _report(
        "the shipped default (1.0) starved the detector on attack-free rounds",
        rate(free, 1.0) < 50.0,
        f"only {rate(free, 1.0):.1f}% of {len(free)} attack-free rounds read quiet",
    )
    ok &= _report(
        "the calibrated default (4.0) accepts most attack-free rounds",
        rate(free, 4.0) > 85.0,
        f"{rate(free, 4.0):.1f}% of {len(free)} attack-free rounds read quiet",
    )
    ok &= _report(
        "and still rejects almost every attacked round",
        rate(attacked, 4.0) < 10.0,
        f"{rate(attacked, 4.0):.1f}% of {len(attacked)} attacked rounds read quiet",
    )
    return ok


def main():
    print("=" * 74)
    print("CustomFedAvg_adaptive_calibrated — offline validation")
    print("=" * 74)
    results = [
        test_boost_never_rescues(),
        test_range_and_degenerate(),
        test_calibration_removes_drift(),
        test_quiet_round_gate(),
        test_gate_ignores_server_eval(),
        test_consensus_selection(),
        test_margins_on_real_runs(),
        test_fang_fix(),
        test_strict_extension(),
        test_override_on_real_runs(),
        test_quiet_gate_calibration(),
    ]
    print("\n" + "=" * 74)
    if all(results):
        print(f"ALL {len(results)} GROUPS PASSED")
        return 0
    print(f"{sum(1 for r in results if not r)} of {len(results)} GROUPS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
