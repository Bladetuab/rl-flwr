# §IV (Proposed Method) — alignment to the rl-flwr implementation (2026-07-14)

Every claim below was read from source in `/home/tuan/get-local-model-bt-id/rl-flwr/`.
Line numbers are as of this date.

## Which code actually runs

`pyproject.toml:31` → `serverapp = "rl_flwr.server_app:app"`.
`server_app.py:44` imports **`CustomFedAvg_adaptive_no_valdelta`** (created 2026-07-07 19:28).

MRO: `CustomFedAvg_adaptive_no_valdelta` → `CustomFedAvg_adaptive_new` → `CustomFedAvg_adaptive` → flwr `FedAvg`.

`server_sweep.py` (which imports `CustomFedAvg_adaptive_new`) is a **stale template** — its own
docstring says "Rename / copy this file over server_app.py". It is NOT what runs.

**All 66 MNIST + F-MNIST sweep cells used in the paper are dated 07-09…07-11**, i.e. after the
5-channel switch. The detection and efficiency figures are uniformly 5-channel.

## Divergences found and FIXED in §IV

| # | Paper said | Code does | Source |
|---|---|---|---|
| 1 | **6**-channel frame; ch.6 = client-reported $\Delta\ell$ | **5 channels**; ch.6 removed as "attacker-controllable" | `no_valdelta.py:47,95-97` |
| 2 | $K = H\times W$ (=256) | $K = \min(HW, n_b) = \mathbf{100}$, zero-padded to 256 | `strategy.py:3028,3099` |
| 3 | basis fit "during warm-up" | fit in **round 1**, as soon as buffer ≥ $n_{\min}$=64 (gets 100) | `strategy.py:3053,3069` |
| 4 | "deeper conv + FC layers" | **FC head only** (`layer4`/`fc`); D = 66,954 | `strategy.py:2334` |
| 5 | $\beta$ "tuned on held-out validation, reported in §V" — **FALSE** (never reported) | hard-coded **(.30,.20,.25,.15,.10)** | `_new.py:280` |
| 6 | $\lambda_r \in (0,1)$ generic | $\lambda_r = \mathbf{0.4}$ (stat), distinct from $\lambda = 0.6$ (ConvLSTM) | `_new.py:107`, `strategy.py:2199` |
| 7 | $\tau_c \in (0,1)$ generic | $\tau_c = \mathbf{0.95}$ | `_new.py:108` |
| 8 | cap $B_t = B_{\min}+(B_{\max}-B_{\min})\cdot\nu/(\nu+1)$ | $\min(\nu,1)$, **then EWMA-smoothed** ($\alpha_B$=0.7) | `strategy.py:3415-3426` |
| 9 | (not stated) | cold-start scores ALL clients, **no cap**, rounds ≤ 8 | `strategy.py:3381` |
| 10 | $\phi_t$ weights unnamed | spread **.35**, severity **.30**, server **.20**, drop **.15** | `strategy.py:2322-25, 2719-23` |
| 11 | $\tau_{\rm abs}\in(0,1]$ generic | $\tau_{\rm abs} = \mathbf{0.40}$ | `_new.py:109` |
| 12 | AGG generic | **plain FedAvg weighted by `num-examples`**; no robust aggregator, no re-weighting | `strategy.py:3638`, `client_app.py:62` |
| 13 | ConvLSTM cost $O(B_t T C^2 HWk^2)$ | hidden width $\eta$=16 dominates $C$=5 → $O(B_t T (C{+}\eta)\eta HWk^2)$ | `convlstm_detector.py:119-131` |
| 14 | "scales linearly with $|S_t|$" | **quadratic** — the clone score is $O(|S_t|^2)$ | `_new.py:258-269` |

Also propagated: §III notation table ($K = \min(HW,n_b)$), and Table II
($C$ 6→**5**, $K$ 256→**100**, plus $n_{\min}$, $\lambda_r$, $\beta$, $\tau_c$, $\alpha_B$,
strictness weights, $\tau_{\rm abs}$).

## VERIFIED CORRECT — equations preserved unchanged

All five statistical-score formulas match the code **exactly**, checked line by line
against `_compute_per_round_stat_scores` (`_new.py:195-284`):
norm ($\log_2$-ratio vs trimmed median of the bottom 80%, `:220-221`);
cosine ($(1-\cos)/2$, `:225-230`); distance (MAD $z$ with 0.6745, $\tanh(z/2)$, `:251-253`);
sign (magnitude-weighted disagreement, `:238-242`); clone (hinge above $\tau_c$, `:258-269`).
Note the norm score uses the **raw full-$D$** delta norm while the other four use the SVD
latent — the paper already had this right.

Also correct and unchanged: the ConvLSTM cell equations; the next-frame MSE score; the
two-sided MAD $z$-score (`z_upper`/`z_lower`/`max`/`tanh`, `_new.py:324-327`); max-pooling
fusion (`_new.py:420`); the EWMA recursion **including the decay of unscored clients**
(`strategy.py:3471-3476`) — which is exactly the mechanism §V-E reports.

## The one thing NOT claimed (deliberately)

Channel 6 was **never populated in any run, ever**: `client_app.py:56` computes
`val_loss_delta` but the metrics dict (`:60-63`) transmits only `train_loss` and
`num-examples`. So the server always received 0.0.

It is therefore TRUE that "channel 6 never carried information."

It is **NOT** safe to claim the April runs (Table IV, Figs. 2-3, 6-channel code) are
numerically identical to the 5-channel method now described. The channel-6 *input* was
zero, but a 6-channel ConvLSTM still emits a nonzero *prediction* for that plane against a
zero target, contributing a per-client error term that MAD-normalization does **not**
cancel (a uniform scale factor cancels; a per-client term does not). For a trained net the
difference is small, but it is not zero. **This is an open item for the user**, not
something to dissolve with an equivalence assertion.

## Deliberately NOT imported into §IV

- "The ConvLSTM may never train" — the training gates are real, but the earlier finding of
  7.2x benign/malicious separation at $q_s{=}0$ shows it *did* train in these runs. Would be
  speculation; belongs in limitations if anywhere.
- Gaussian attack `magnitude=3` (call site) vs `magnitude=10` (stale docstring in
  `_new.py:52,219`) — an attack/experiment fact, not a method fact. Flagged, not written.
