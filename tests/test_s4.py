"""Tests for the S4 layer.

Verifies:
  1. HiPPO-LegS matrix matches the closed-form definition
  2. Discretization stability (Abar eigenvalues inside unit disk)
  3. Recurrent and convolutional views produce identical outputs
  4. Output shape correctness
  5. Gradient flow through learnable parameters
"""
from __future__ import annotations

import pytest
import torch

from s4 import S4Layer, bilinear_discretize, hippo_legs, zoh_discretize


# ---------------------------------------------------------------------------
# HiPPO matrix
# ---------------------------------------------------------------------------


def test_hippo_legs_shape_and_zeros():
    A = hippo_legs(8)
    assert A.shape == (8, 8)
    # Strictly lower triangular (zeros on and above diagonal except diagonal)
    assert torch.all(A[torch.triu_indices(8, 8, offset=1).unbind()] == 0)


def test_hippo_legs_diagonal():
    A = hippo_legs(10)
    # Diagonal entries: A[n, n] = -(n + 1)
    for n in range(10):
        assert torch.isclose(A[n, n], torch.tensor(-(n + 1.0)))


def test_hippo_legs_off_diagonal():
    N = 6
    A = hippo_legs(N)
    for n in range(N):
        for k in range(n):
            expected = -((2 * n + 1) ** 0.5) * ((2 * k + 1) ** 0.5)
            assert torch.isclose(A[n, k], torch.tensor(expected)), f"A[{n},{k}]"


def test_hippo_legs_is_stable():
    """All eigenvalues of HiPPO-LegS have negative real part (strictly stable)."""
    A = hippo_legs(16, dtype=torch.float64)
    eigs = torch.linalg.eigvals(A)
    assert (eigs.real < 0).all()


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------


def test_zoh_discretize_shapes():
    A = hippo_legs(8)
    B = torch.zeros(8, 1)
    B[0, 0] = 1.0
    Abar, Bbar = zoh_discretize(A, B, step=0.01)
    assert Abar.shape == (8, 8)
    assert Bbar.shape == (8, 1)


def test_bilinear_discretize_shapes():
    A = hippo_legs(8)
    B = torch.zeros(8, 1)
    B[0, 0] = 1.0
    Abar, Bbar = bilinear_discretize(A, B, step=0.01)
    assert Abar.shape == (8, 8)
    assert Bbar.shape == (8, 1)


def test_discretized_Abar_is_stable():
    """Abar eigenvalues must lie inside the unit disk (|lambda| < 1)."""
    A = hippo_legs(16, dtype=torch.float64)
    B = torch.zeros(16, 1, dtype=torch.float64)
    B[0, 0] = 1.0
    for step in [0.001, 0.01, 0.1, 1.0]:
        Abar, _ = bilinear_discretize(A, B, step=step)
        eigs = torch.linalg.eigvals(Abar)
        assert (eigs.abs() < 1.0).all(), f"Unstable at step={step}"


# ---------------------------------------------------------------------------
# S4 layer forward
# ---------------------------------------------------------------------------


@pytest.fixture
def small_s4():
    torch.manual_seed(0)
    return S4Layer(d_model=4, d_state=8, discretization="bilinear")


def test_s4_output_shape(small_s4):
    u = torch.randn(2, 16, 4)
    y = small_s4(u)
    assert y.shape == (2, 16, 4)


def test_s4_recurrent_output_shape(small_s4):
    u = torch.randn(2, 16, 4)
    y = small_s4(u, mode="recurrent")
    assert y.shape == (2, 16, 4)


def test_s4_conv_and_recurrent_match():
    """The two views of the SSM must produce identical outputs.

    This is the central theoretical guarantee of S4. We use float64 for a
    tight tolerance.
    """
    torch.manual_seed(42)
    layer = S4Layer(d_model=2, d_state=4, discretization="bilinear")
    layer = layer.double()
    u = torch.randn(1, 32, 2, dtype=torch.float64)
    y_conv = layer(u, mode="conv")
    y_rec = layer(u, mode="recurrent")
    # The D*u term is identical in both; the state contribution should match.
    # Use a modest tolerance because the FFT path uses fp accumulation.
    assert torch.allclose(y_conv, y_rec, atol=1e-6, rtol=1e-5), (
        f"Max diff: {(y_conv - y_rec).abs().max().item()}"
    )


def test_s4_gradient_flow(small_s4):
    """Gradients must flow to C, D, and log_step."""
    u = torch.randn(2, 16, 4)
    y = small_s4(u)
    loss = y.sum()
    loss.backward()
    assert small_s4.C.grad is not None
    assert small_s4.D.grad is not None
    assert small_s4.log_step.grad is not None
    assert small_s4.C.grad.shape == small_s4.C.shape


def test_s4_bidirectional():
    torch.manual_seed(0)
    layer = S4Layer(d_model=4, d_state=8, bidirectional=True)
    u = torch.randn(2, 16, 4)
    y = layer(u)
    assert y.shape == (2, 16, 4)


def test_s4_long_sequence_no_nan():
    """A long sequence (the whole point of S4) must not produce NaNs."""
    torch.manual_seed(0)
    layer = S4Layer(d_model=2, d_state=16)
    u = torch.randn(1, 1024, 2)
    y = layer(u)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
