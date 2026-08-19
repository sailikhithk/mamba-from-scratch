"""S4 layer: structured state-space model with recurrent and convolutional views.

Implements the core S4 layer from Gu et al. (2022). The key insight is
that the same linear time-invariant state-space model

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

can be evaluated two ways:
  1. Recurrent (step-by-step) - O(L) time, good for autoregressive generation
  2. Convolutional (FFT-based) - O(L log L) time, good for parallel training

Both views produce identical outputs (up to numerical precision), which is
the theoretical guarantee we verify in the test suite.

This is the naive (non-Krylov, non-structured) implementation for pedagogical
clarity. An optimized version using truncated generating kernels is planned
(see CHANGELOG.md).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .hippo import bilinear_discretize, hippo_legs, zoh_discretize


class S4Layer(nn.Module):
    """A single S4 state-space layer.

    Args:
        d_model:  input/output (hidden) dimension H
        d_state:  state size N (HiPPO matrix is N x N)
        dropout:  dropout probability
        discretization: "zoh" or "bilinear"
        bidirectional: if True, scan both directions and sum (like a bi-LSTM)
        learnable_A: if True, A is a learnable parameter (initialized to HiPPO).
                    If False, A is a fixed buffer (original S4 default).

    Shapes:
        input  u: (B, L, H)
        output y: (B, L, H)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        discretization: str = "bilinear",
        bidirectional: bool = False,
        learnable_A: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.discretization = discretization
        self.bidirectional = bidirectional

        # A: (N, N) HiPPO-LegS, optionally learnable
        A_init = hippo_legs(d_state, dtype=torch.float32)
        if learnable_A:
            self.A = nn.Parameter(A_init.clone())
        else:
            self.register_buffer("A", A_init.clone())

        # B: (N, 1) - input projection. S4 parameterizes B as learnable.
        # We follow the common convention of a fixed B = [1, 0, ..., 0]^T
        # and learn C (the readout). A learnable B is also fine.
        B_init = torch.zeros(d_state, 1, dtype=torch.float32)
        B_init[0, 0] = 1.0
        self.register_buffer("B", B_init)

        # C: (H, N) - learnable readout, one row per hidden channel
        # Initialized with a small normal so initial outputs are near zero.
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / math.sqrt(d_state)))

        # D: (H,) - skip connection (residual). Initialized to 1 so the layer
        # behaves close to identity at start.
        self.D = nn.Parameter(torch.ones(d_model))

        # step size dt. S4 learns log(dt) for stability. Init to 0.01.
        self.log_step = nn.Parameter(torch.tensor(math.log(0.01)))

        # Per-channel learnable scaling on the state contribution (S4 "gamma")
        # Optional; helps expressiveness. We keep it simple here.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def step(self) -> Tensor:
        return torch.exp(self.log_step)

    def discretize(self) -> tuple[Tensor, Tensor]:
        """Return (Abar, Bbar) for the current parameters.

        ``step`` is kept as a tensor so gradients flow to ``log_step``.
        """
        step = self.step  # tensor, requires_grad=True
        if self.discretization == "zoh":
            return zoh_discretize(self.A, self.B, step)
        elif self.discretization == "bilinear":
            return bilinear_discretize(self.A, self.B, step)
        else:
            raise ValueError(f"Unknown discretization: {self.discretization}")

    def forward_recurrent(self, u: Tensor) -> Tensor:
        """Recurrent view. u: (B, L, H) -> y: (B, L, H).

        For each channel h in [0, H), the SSM is independent:
            x_{k+1} = Abar x_k + Bbar u_k^{(h)}
            y_k^{(h)} = C[h] @ x_k + D[h] * u_k^{(h)}

        We vectorize across batch and channels.
        """
        B, L, H = u.shape
        Abar, Bbar = self.discretize()  # (N, N), (N, 1)
        N = self.d_state

        # State: (B, H, N) - one state vector per (batch, channel)
        x = torch.zeros(B, H, N, device=u.device, dtype=u.dtype)
        outputs = []

        # Pre-extract C, D
        C = self.C  # (H, N)
        D = self.D  # (H,)

        for t in range(L):
            ut = u[:, t, :]  # (B, H)
            # x_{k+1} = Abar x_k + Bbar * u_k
            # Bbar is (N, 1); u_k is (B, H). We need (B, H, N) update.
            # x_new[b, h, n] = sum_m Abar[n, m] * x[b, h, m] + Bbar[n, 0] * u[b, h]
            x = x @ Abar.T + ut.unsqueeze(-1) * Bbar.squeeze(-1).unsqueeze(0).unsqueeze(0)
            # y[b, h] = C[h, :] @ x[b, h, :] + D[h] * u[b, h]
            y = (x * C.unsqueeze(0)).sum(dim=-1) + D.unsqueeze(0) * ut
            outputs.append(y)

        y_seq = torch.stack(outputs, dim=1)  # (B, L, H)
        return self.dropout(y_seq)

    def forward_conv(self, u: Tensor) -> Tensor:
        """Convolutional view. u: (B, L, H) -> y: (B, L, H).

        Computes the truncated convolution kernel
            K_t = C @ Abar^t @ Bbar     for t = 0, 1, ..., L-1
        then convolves with u via FFT. Mathematically identical to
        forward_recurrent (up to fp precision).
        """
        B, L, H = u.shape
        Abar, Bbar = self.discretize()  # (N, N), (N, 1)
        N = self.d_state
        C = self.C  # (H, N)
        D = self.D  # (H,)

        # Build kernel K of shape (L, H): K[t, h] = C[h] @ Abar^t @ Bbar
        # We compute powers of Abar iteratively (naive; the structured
        # algorithm in s4_conv.py is faster).
        K = torch.zeros(L, H, device=u.device, dtype=u.dtype)
        Abar_power = torch.eye(N, device=u.device, dtype=u.dtype)
        Bbar_squeezed = Bbar.squeeze(-1)  # (N,)
        for t in range(L):
            # K[t, h] = C[h, :] @ (Abar^t @ Bbar)
            state_at_t = Abar_power @ Bbar_squeezed  # (N,)
            K[t] = C @ state_at_t  # (H,)
            Abar_power = Abar_power @ Abar

        # Causal convolution via FFT: y_t = sum_{s=0}^{t} K_{t-s} * u_s
        # This is a standard (non-flipped) convolution truncated to length L.
        # Pad both to length n_fft >= 2L-1, FFT, multiply, IFFT, take first L.
        n_fft = 2 * L  # >= 2L-1, power-of-2 friendly
        U = torch.fft.rfft(u, n=n_fft, dim=1)  # (B, n_fft//2+1, H)
        Kf = torch.fft.rfft(K, n=n_fft, dim=0)  # (n_fft//2+1, H)
        y_full = torch.fft.irfft(U * Kf.unsqueeze(0), n=n_fft, dim=1)  # (B, n_fft, H)
        # The first L samples are the causal convolution output.
        y = y_full[:, :L, :]  # (B, L, H)

        # Add the D * u skip connection
        y = y + D.unsqueeze(0).unsqueeze(0) * u
        return self.dropout(y)

    def forward(self, u: Tensor, mode: str = "conv") -> Tensor:
        """Forward pass. mode: "conv" (default, fast) or "recurrent".

        If ``bidirectional`` was set, the SSM is run forward and backward
        (on the reversed sequence) and the two outputs are summed, like a
        bi-LSTM.
        """
        if mode == "conv":
            y = self.forward_conv(u)
        elif mode == "recurrent":
            y = self.forward_recurrent(u)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.bidirectional:
            u_rev = torch.flip(u, dims=[1])
            if mode == "conv":
                y_rev = self.forward_conv(u_rev)
            else:
                y_rev = self.forward_recurrent(u_rev)
            y = y + torch.flip(y_rev, dims=[1])
        return y
