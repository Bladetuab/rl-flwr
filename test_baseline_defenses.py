"""Offline validation of the SignGuard and DnC selection rules.

Bypasses the Flower harness: builds synthetic update matrices with a known
malicious set and checks each defense's keep-set directly. This catches logic
bugs in minutes instead of after a 50-round federated run.

Run:  python3 test_baseline_defenses.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from rl_flwr.baseline_defenses import SignGuardDefense, DnCDefense  # noqa: E402


N_BENIGN, N_MAL, D = 12, 8, 4000
RNG = np.random.default_rng(0)


def make_updates(kind: str, decorr_cos=None):
    """Return (U, cids, malicious_cids). Benign clients are heterogeneous
    (non-IID), malicious clients share a payload, optionally de-correlated."""
    benign = RNG.normal(0.0, 0.02, size=(N_BENIGN, D)) + RNG.normal(0, 0.005, size=(1, D))

    if kind == "gaussian":                 # norm-inflating
        payload = RNG.normal(0.0, 0.20, size=(1, D))
        mal = np.repeat(payload, N_MAL, axis=0) + RNG.normal(0, 0.2, (N_MAL, D))
    elif kind == "lie":                    # sub-mean, shared vector
        payload = benign.mean(0, keepdims=True) - 0.9 * benign.std(0, keepdims=True)
        mal = np.repeat(payload, N_MAL, axis=0)
    elif kind == "min_max":                # inside benign hull, shared vector
        payload = benign.mean(0, keepdims=True) - 0.5 * benign.std(0, keepdims=True)
        mal = np.repeat(payload, N_MAL, axis=0)
    else:
        raise ValueError(kind)

    if decorr_cos is not None and kind in ("lie", "min_max"):
        p_norm = np.linalg.norm(payload)
        s = p_norm * np.sqrt((1 - decorr_cos) / (decorr_cos * D))
        mal = mal + RNG.normal(0.0, s, size=mal.shape)

    U = np.vstack([benign, mal]).astype(np.float32)
    cids = list(range(N_BENIGN + N_MAL))
    return U, cids, set(cids[N_BENIGN:])


def keepset(defense_cls, U, cids, **kw):
    """Invoke a defense's selection rule on a synthetic matrix."""
    d = object.__new__(defense_cls)
    d.current_filter_quantile = 0.55
    if defense_cls is SignGuardDefense:
        d.sg_norm_lo, d.sg_norm_hi, d.sg_n_sub = 0.1, 3.0, 1000
        d._sg_rng = np.random.default_rng(42)
    else:
        d.dnc_niters, d.dnc_d_sub, d.dnc_c = 5, 2000, 1.0
        d._dnc_rng = np.random.default_rng(42)
        d.assumed_malicious = kw.get("assumed_malicious", N_MAL)

    captured = {}
    d._updates_matrix = lambda replies: (cids, U)
    d._finalize = lambda r, ar, cs, ac, kc, mc: captured.update(keep=set(kc)) or (None, None)
    d._filter_attacked_replies_by_anomaly(1, [], {}, min_clients=2)
    return captured["keep"]


def report(name, keep, mal, n_all):
    tp = len(mal - keep)                 # malicious correctly dropped
    fn = len(mal & keep)                 # malicious admitted  -> evasion
    fp = len((set(range(n_all)) - mal) & (set(range(n_all)) - keep))  # benign dropped
    recall = tp / len(mal) if mal else float("nan")
    print(f"    {name:10s} recall={recall:5.2f}  evaded={fn}/{len(mal)}  benign_dropped={fp}/{n_all-len(mal)}")
    return recall


print("=" * 74)
print("SignGuard / DnC selection rules on synthetic updates")
print(f"  {N_BENIGN} benign (non-IID) + {N_MAL} malicious, d={D}")
print("=" * 74)

fails = []
for kind in ("gaussian", "lie", "min_max"):
    for decorr in (None, 0.90):
        if decorr is not None and kind == "gaussian":
            continue
        tag = "identical payload" if decorr is None else f"de-correlated (cos={decorr})"
        print(f"\n  {kind.upper():8s} | {tag}")
        U, cids, mal = make_updates(kind, decorr)
        if kind != "gaussian":
            M = U[N_BENIGN:] / np.linalg.norm(U[N_BENIGN:], axis=1, keepdims=True)
            G = M @ M.T
            np.fill_diagonal(G, -np.inf)
            print(f"    (max pairwise cosine among malicious = {G.max():.4f})")
        for nm, cls in (("SignGuard", SignGuardDefense), ("DnC", DnCDefense)):
            r = report(nm, keepset(cls, U, cids), mal, len(cids))
            if np.isnan(r):
                fails.append(f"{nm}/{kind}/{decorr}")

print("\n" + "=" * 74)
print("Both rules executed without error on every configuration."
      if not fails else f"FAILURES: {fails}")
print("Recall here is a sanity check on the implementation, not a result:")
print("synthetic updates are not the federated runs.")
print("=" * 74)
