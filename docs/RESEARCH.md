# Mamba from Scratch - Research Findings

## Existing Implementations Landscape

### Official
| Repo | Stars | CUDA | Pure PyTorch | Implements |
|------|-------|------|--------------|------------|
| state-spaces/mamba | 18,682 | Optional | Yes | Mamba-1, 2, 3 |
| state-spaces/s4 | 2,917 | Optional | Yes (slow) | S4, S4D |

### Educational / From-Scratch
| Repo | Stars | CUDA | Pure PyTorch | Implements |
|------|-------|------|--------------|------------|
| johnma2006/mamba-minimal | 2,962 | No | Yes | Mamba-1 only |
| alxndrTL/mamba.py | 1,469 | No (M1), Yes (M2) | Yes | Mamba-1, 2, Jamba, Vim |
| varad-more/mamba-from-scratch | 0 | Optional Triton | Yes | Mamba-1, 2 + SSD |
| rishikksh20/mamba3-pytorch | 32 | No | Yes | Mamba-3 |
| Xinguang/MiniMamba | 10 | No | Yes | Mamba-1 (PyPI) |

### Key Gap Analysis

**johnma2006/mamba-minimal** (2.9K stars) is the most popular educational impl but:
- Only Mamba-1, no Mamba-2 SSD
- Single file, no notebooks, no visualizations
- No benchmark suite
- Last updated Dec 2023

**varad-more/mamba-from-scratch** (0 stars) is comprehensive but:
- No notebooks or visualizations
- No educational commentary
- 0 stars = no adoption

**No existing repo** covers ALL of:
1. S4 from scratch (the foundation)
2. Mamba-1 (S6 selective scan)
3. Mamba-2 (SSD algorithm)
4. Jupyter notebooks with visualizations per component
5. Benchmark suite comparing all three
6. Substack/blog series cross-referenced

### Our Differentiation

Our repo will be the FIRST to provide:
1. Layer-by-layer progression: S4 -> S6 -> SSD (Mamba-2)
2. Visual notebooks for each component (selective scan, parallel scan, ZOH discretization)
3. Side-by-side benchmark: speed + perplexity + memory
4. Cross-referenced Substack series
5. Connection to published SSM chapter (Cambridge Scholars 2025)

## Technical Foundation

### S4 (ICLR 2022)
- Continuous-time SSM: x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t)
- DPLR parameterization of A (Diagonal Plus Low-Rank)
- HiPPO matrix for long-range memory
- Convolution mode (training) + recurrent mode (inference)
- Paper: arxiv.org/abs/2111.00396

### Mamba / S6 (Dec 2023)
- Input-dependent parameters: delta_t, B_t, C_t are functions of x_t
- Selective scan: model chooses what to remember/forget
- Hardware-aware parallel scan (FlashAttention-style kernel fusion)
- No attention, no MLP blocks - just selective SSM + gating
- Paper: arxiv.org/abs/2312.00752

### Mamba-2 / SSD (ICML 2024)
- Scalar A matrix (A = a * I) instead of full diagonal
- Parallel projection of A, B, C, D, delta from input
- Multihead SSM structure (like multihead attention)
- SSD algorithm: block decomposition combining linear recurrence + quadratic attention
- 2-8x faster than Mamba-1 via tensor core utilization
- Paper: arxiv.org/abs/2405.21060

### Key Equations

**S4 (time-invariant):**
```
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C * h_t
where A_bar = exp(delta * A), B_bar = (A_bar - I) * A^-1 * B
```

**S6 (selective, input-dependent):**
```
delta_t = Linear_delta(x_t)
B_t = Linear_B(x_t)
C_t = Linear_C(x_t)
A_bar_t = exp(delta_t * A)
B_bar_t = delta_t * B_t
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t
```

**SSD (Mamba-2, scalar A):**
```
A_t = a_t * I  (scalar)
A, B, C, D, delta = Linear(x)  (parallel projection)
A_bar_t = exp(delta_t * a_t) * I
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t + D_t * x_t
```

### Known Limitations (from critique papers)
1. Associative recall: weaker than attention for KV retrieval
2. Copy operations: struggles without state growth
3. Symmetry: asymmetry bias in pattern recognition
4. Forgetting: difficulty forgetting old tokens
5. Optimization: brittle training dynamics, narrow LR window

### Production Mamba Models
- Jamba (AI21): hybrid Transformer-Mamba-MoE, 52B total, 256K context
- Falcon-Mamba 7B (TII): pure Mamba, #1 open-source SSLM
- Codestral-Mamba (Mistral): 7B code model, 75% HumanEval

## Implementation Plan

### Phase 1: S4 (Week 1-2)
- Implement HiPPO matrix
- Implement ZOH discretization
- Implement S4 layer (convolution mode)
- Notebook: visualization of state evolution
- Test: parity with state-spaces/s4 on synthetic data

### Phase 2: Mamba-1 / S6 (Week 3-4)
- Implement selective scan (naive sequential)
- Implement parallel scan (Blelloch)
- Implement Mamba block (Conv1D + SSM + gating)
- Notebook: visualization of selective forgetting
- Test: parity with johnma2006/mamba-minimal

### Phase 3: Mamba-2 / SSD (Week 5-6)
- Implement scalar A SSD layer
- Implement chunked SSD algorithm
- Implement multihead SSM
- Notebook: visualization of SSD vs S6
- Test: parity with state-spaces/mamba2

### Phase 4: Benchmarks (Week 7-8)
- Train all three on WikiText-103 (small scale)
- Compare perplexity, speed, memory
- Notebook: benchmark dashboard
- Substack series: one post per phase

## Dependencies to Build On
- PyTorch (no CUDA required for educational version)
- johnma2006/mamba-minimal: reference for Mamba-1 parity testing
- state-spaces/mamba: reference for Mamba-2 parity testing
- alxndrTL/mamba.py: parallel scan reference
