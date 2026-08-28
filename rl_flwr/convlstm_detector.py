# # convlstm_detector.py

# import torch
# import torch.nn as nn


# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
#         super().__init__()
#         padding = kernel_size // 2
#         self.hidden_dim = hidden_dim
#         self.conv = nn.Conv2d(
#             in_channels=input_dim + hidden_dim,
#             out_channels=4 * hidden_dim,
#             kernel_size=kernel_size,
#             padding=padding,
#             bias=bias,
#         )

#     def forward(self, x: torch.Tensor, state):
#         # x: [B, C_in, H, W]
#         h_cur, c_cur = state  # [B, H_dim, H, W]
#         combined = torch.cat([x, h_cur], dim=1)
#         conv_out = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)
#         return h_next, c_next

#     def init_hidden(self, batch_size: int, spatial_size, device=None):
#         H, W = spatial_size
#         if device is None:
#             device = next(self.parameters()).device
#         h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
#         c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
#         return h, c


# class ConvLSTMForecaster(nn.Module):
#     """
#     Simple ConvLSTM forecaster:
#     Input: sequence X[0..T-2], Output: prediction of X[T-1]
#     """

#     def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
#         super().__init__()
#         self.cell = ConvLSTMCell(in_channels, hidden_dim, kernel_size)
#         self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

#     def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
#         """
#         x_seq: [B, T, C, H, W]
#         returns pred: [B, C, H, W] for frame T-1
#         """
#         B, T, C, H, W = x_seq.shape
#         h, c = self.cell.init_hidden(B, (H, W), device=x_seq.device)
#         # Use frames 0..T-2 to predict T-1
#         for t in range(T - 1):
#             h, c = self.cell(x_seq[:, t], (h, c))
#         pred = self.out_conv(h)
#         return pred
    
#     def training_loss(self, seq: torch.Tensor) -> torch.Tensor:
#         """
#         Unsupervised loss for warmup training.

#         For example: sequence forecasting
#           - Input: seq[0:T-1]
#           - Target: seq[T-1]
#           - Loss: MSE between predicted last frame and true last frame.
#         """
#         if seq.dim() == 4:
#             # [T, C, H, W] -> [1, T, C, H, W]
#             seq = seq.unsqueeze(0)

#         # example: use everything except last frame as input
#         inp = seq[:, :-1]   # [B, T-1, C, H, W]
#         target = seq[:, -1] # [B, C, H, W]

#         pred = self.forward(inp)       # [B, C, H, W]
#         loss = torch.nn.functional.mse_loss(pred, target)
#         return loss

#     @torch.no_grad()
#     def anomaly_score(self, x_seq: torch.Tensor) -> float:
#         """
#         x_seq: [T, C, H, W] (single sequence)
#         returns scalar anomaly score (MSE on last frame)
#         """
#         x_seq = x_seq.unsqueeze(0)  # [1, T, C, H, W]
#         pred = self.forward(x_seq)  # [1, C, H, W]
#         target = x_seq[:, -1]       # [1, C, H, W]
#         score = torch.mean((pred - target) ** 2).item()
#         return score



import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state):
        # x: [B, C_in, H, W] on the SAME device as self.conv.weight
        h_cur, c_cur = state  # [B, H_dim, H, W]
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial_size, device=None):
        H, W = spatial_size
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """
    Simple ConvLSTM forecaster:
    Input: sequence X[0..T-2], Output: prediction of X[T-1]
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_dim, kernel_size)
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, T, C, H, W]
        returns pred: [B, C, H, W] for frame T-1
        """
        # --- NEW: move input to same device as model parameters ---
        device = next(self.parameters()).device
        x_seq = x_seq.to(device)

        B, T, C, H, W = x_seq.shape
        h, c = self.cell.init_hidden(B, (H, W), device=device)

        # Use frames 0..T-2 to predict T-1
        for t in range(T - 1):
            # x_seq[:, t] is [B, C, H, W] on 'device'
            h, c = self.cell(x_seq[:, t], (h, c))

        pred = self.out_conv(h)
        return pred

    def training_loss(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.dim() == 4:
            seq = seq.unsqueeze(0)  # [1, T, C, H, W]

        pred = self.forward(seq)        # forward sẽ dùng 0..T-2 để dự đoán T-1
        target = seq[:, -1].to(pred.device)
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss

    @torch.no_grad()
    def anomaly_score(self, x_seq: torch.Tensor) -> float:
        """
        x_seq: [T, C, H, W] (single sequence)
        returns scalar anomaly score (MSE on last frame)
        """
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0)  # [1, T, C, H, W]

        # Move to same device as model
        device = next(self.parameters()).device
        x_seq = x_seq.to(device)

        pred = self.forward(x_seq)    # [1, C, H, W]
        target = x_seq[:, -1]         # [1, C, H, W]
        score = torch.mean((pred - target) ** 2).item()
        return score
