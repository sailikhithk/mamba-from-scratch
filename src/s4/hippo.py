"""HiPPO-LegS matrix and discretization routines for S4.

Reference: Gu, Goel, Re (2022), "Efficiently Modeling Long Sequences
with Structured State Spaces", ICLR. https://arxiv.org/abs/2111.00396
"""
from __future__ import annotations

import torch
from torch import Tensor


def hippo_legs(N: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the HiPPO-LegS matrix A of shape (N, N).

    LegS = Legendre (scaled) basis over [0, 1]. Canonical form from the
    S4 paper (Appendix B, Eq. 8):

        A[n, k] = -sqrt(2n+1) * sqrt(2k+1)   if n > k
        A[n, n] = -(n + 1)
        A[n, k] = 0                          if n < k
    """
    A = torch.zeros(N, N, dtype=dtype)
    for n in range(N):
        for k in range(n + 1):
            if n > k:
                A[n, k] = -((2 * n + 1) ** 0.5) * ((2 * k + 1) ** 0.5)
            else:  # n == k
                A[n, k] = -(n + 1)
    return A


def zoh_discretize(A: Tensor, B: Tensor, step: Tensor | float) -> tuple[Tensor, Tensor]:
    """Zero-order-hold discretization: x_{k+1} = Abar x_k + Bbar u_k.

    Abar = expm(A * step)
    Bbar = A^{-1} (Abar - I) B

    A is strictly stable (eigenvalues with negative real part) so it is
    invertible and the closed form is well-defined. ``step`` may be a tensor
    with requires_grad=True; gradients flow through it.
    """
    device, dtype = A.device, A.dtype
    I = torch.eye(A.shape[0], device=device, dtype=dtype)
    Abar = torch.matrix_exp(A * step)
    A_inv = torch.linalg.inv(A)
    Bbar = A_inv @ (Abar - I) @ B
    return Abar, Bbar


def bilinear_discretize(A: Tensor, B: Tensor, step: Tensor | float) -> tuple[Tensor, Tensor]:
    """Bilinear (Tustin) discretization. More numerically stable than ZOH
    for stiff A and the default in the official S4 reference code.
    ``step`` may be a tensor with requires_grad=True.
    """
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    Abar = torch.linalg.solve(I - step * A / 2, I + step * A / 2)
    Bbar = torch.linalg.solve(I - step * A / 2, step * B)
    return Abar, Bbar
