# mamba-from-scratch

Educational, from-scratch implementations of the S4, Mamba-1 (S6), and Mamba-2 (SSD) state-space models, with clear math and passing tests.

## Goal

Demonstrate "original contributions of major significance" for the EB-1A petition by building a clean, well-tested, pedagogical implementation of structured state-space models (S4 / S6 / SSD) that bridges the gap between the math and production code.

## Project structure

```
mamba-from-scratch/
  pyproject.toml
  src/
    s4/
      hippo.py    # HiPPO-LegS matrix + ZOH / bilinear discretization
      layer.py    # S4Layer: recurrent + convolutional views
    mamba1/       # (coming) S6 selective scan
    mamba2/       # (coming) SSD parallel scan
  tests/
    test_s4.py    # 13 tests, all passing
  notebooks/      # (coming) visual walkthroughs
  docs/           # (coming) math derivations
  benchmarks/     # (coming) vs mamba-ssm
```

## S4 layer (done)

The S4 layer from [Gu, Goel, Re (2022)](https://arxiv.org/abs/2111.00396) implements the continuous-time state-space model

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

with two equivalent computational views:

1. **Recurrent** (step-by-step): O(L) time, used for autoregressive generation
2. **Convolutional** (FFT-based): O(L log L) time, used for parallel training

Both views produce identical outputs (verified in `test_s4_conv_and_recurrent_match`).

### Key components

- `hippo_legs(N)`: the HiPPO-LegS matrix A, the structured initialization that gives S4 its long-range memory
- `zoh_discretize` / `bilinear_discretize`: convert (A, B, step) to discrete (Abar, Bbar)
- `S4Layer`: a PyTorch `nn.Module` with learnable C, D, and log-step; supports both `mode="conv"` and `mode="recurrent"`

### Run the tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Tested with Python 3.12 + torch 2.2.2. All 13 tests pass.

## Roadmap

- [x] S4: HiPPO + ZOH/bilinear + recurrent/conv views + tests
- [ ] S4 optimized: truncated generating kernel (Krylov / FFT power)
- [ ] Mamba-1 (S6): input-dependent (selective) scan
- [ ] Mamba-2 (SSD): parallel structured scan via chunked attention
- [ ] Notebooks: visual walkthrough of each component
- [ ] Benchmarks: vs `mamba-ssm` on Long-Range Arena

## References

- Gu, Goel, Re. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. https://arxiv.org/abs/2111.00396
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." 2023. https://arxiv.org/abs/2312.00752
- Dao, Gu. "Transformers are SSMs: Generalized Models and Parallel Algorithms." 2024. https://arxiv.org/abs/2405.21060
