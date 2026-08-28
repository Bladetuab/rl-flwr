"""
CustomFedAvg_adaptive_no_valdelta.py

CustomFedAvg_adaptive_new WITHOUT the val_loss_delta ConvLSTM channel (channel 6).

Motivation
----------
The ConvLSTM anomaly detector encodes each client update as a 6-channel frame
(see CustomFedAvg_adaptive._build_frame_for_client). Channels 1-5 are computed by
the SERVER from the client's submitted weight update (standardized projected
vector, sign pattern, magnitude, L2 norm, cosine-to-mean). Channel 6 is the
client-reported `val_loss_delta` metric — evaluated on the client's own data and
ingested verbatim by the server (strategy.py: `client_val_deltas.get(cid, 0.0)`).

In the Byzantine threat model a malicious client controls everything it sends, so
channel 6 is an unverifiable, attacker-controllable input to the detector:
  1. A malicious client can submit poison yet report a benign-looking delta,
     decoupling "how harmful I am" from "how benign I look" — a free evasion axis
     that channels 1-5 (derived from the submitted update itself) do not offer.
  2. mu_L / sigma_L are population statistics over ALL reported deltas, so a few
     extreme reported values can poison the standardization for benign clients too.

This strategy removes channel 6 entirely (feature_C 6 -> 5) so the ConvLSTM only
consumes signals the server derives from the submitted update. Everything else
(5-signal stat detector, two-sided relative ConvLSTM score, fused max-scoring,
dual-threshold filter) is inherited unchanged from CustomFedAvg_adaptive_new.
"""

from __future__ import annotations

import torch

from rl_flwr.convlstm_detector import ConvLSTMForecaster
from rl_flwr.CustomFedAvg_adaptive_new import CustomFedAvg_adaptive_new


class CustomFedAvg_adaptive_no_valdelta(CustomFedAvg_adaptive_new):
    """CustomFedAvg_adaptive_new with the attacker-controllable val_loss_delta
    ConvLSTM channel (channel 6) removed — the ConvLSTM sees 5 server-computed
    channels instead of 6.  Drop-in replacement: identical constructor signature.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Drop channel 6 (client-reported val_loss_delta): 6 -> 5 channels, and
        # rebuild the detector so its input channel count matches the frame.
        self.feature_C = 5
        self.detector = ConvLSTMForecaster(
            in_channels=self.feature_C,
            hidden_dim=16,
            kernel_size=3,
        )
        self.detector.eval()

    # ------------------------------------------------------------------
    # Override: 5-channel frame (drop x6 = standardized val_loss_delta)
    # ------------------------------------------------------------------
    def _build_frame_for_client(
        self,
        cid: int,
        v: torch.Tensor,          # [D] projected latent update vector (D = H*W)
        idx: int,
        stats: dict,
        val_delta_i: float,       # kept for signature compatibility; intentionally unused
    ) -> torch.Tensor:
        """Identical to CustomFedAvg_adaptive._build_frame_for_client but WITHOUT
        channel 6 (val_loss_delta).  Returns a [5, H, W] frame."""
        H = self.feature_H
        W = self.feature_W

        mu_v = stats["mu_v"]
        sigma_v = stats["sigma_v"]
        mu_m = stats["mu_m"]
        sigma_m = stats["sigma_m"]
        norms = stats["norms"]
        mu_N = stats["mu_N"]
        sigma_N = stats["sigma_N"]
        cos = stats["cos"]
        mu_C = stats["mu_C"]
        sigma_C = stats["sigma_C"]

        # 1) standardized projected update vector
        x1 = ((v - mu_v) / sigma_v).view(1, H, W)
        # 2) sign pattern
        x2 = torch.sign(v).view(1, H, W)
        # 3) standardized coordinate magnitude
        x3 = ((v.abs() - mu_m) / sigma_m).view(1, H, W)
        # 4) standardized L2 norm (scalar tiled across the plane)
        N_hat = (norms[idx] - mu_N) / sigma_N
        x4 = torch.full((1, H, W), float(N_hat))
        # 5) standardized cosine-to-mean (scalar tiled across the plane)
        C_hat = (cos[idx] - mu_C) / sigma_C
        x5 = torch.full((1, H, W), float(C_hat))

        # channel 6 (standardized val_loss_delta) intentionally omitted —
        # it is a client self-reported, attacker-controllable signal.
        return torch.cat([x1, x2, x3, x4, x5], dim=0)  # [5, H, W]
