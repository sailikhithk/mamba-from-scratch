"""Quick smoke test: build an S4 layer, run both views, print shapes and diff.

Run:  PYTHONPATH=src python scripts/smoke_test.py
"""
from __future__ import annotations

import torch

from s4 import S4Layer, hippo_legs


def main() -> None:
    torch.manual_seed(0)
    layer = S4Layer(d_model=8, d_state=16, discretization="bilinear")
    u = torch.randn(2, 64, 8)

    y_conv = layer(u, mode="conv")
    y_rec = layer(u, mode="recurrent")

    print(f"input  shape: {tuple(u.shape)}")
    print(f"conv   shape: {tuple(y_conv.shape)}")
    print(f"recur  shape: {tuple(y_rec.shape)}")
    print(f"max |conv - recur|: {(y_conv - y_rec).abs().max().item():.2e}")

    # HiPPO matrix sanity
    A = hippo_legs(8, dtype=torch.float64)
    eigs = torch.linalg.eigvals(A)
    print(f"HiPPO(8) eigenvalue real parts all < 0: {(eigs.real < 0).all().item()}")
    print("OK")


if __name__ == "__main__":
    main()
