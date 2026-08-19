# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- S4 optimized kernel: truncated generating function via Krylov / FFT powers
- Mamba-1 (S6): input-dependent selective scan
- Mamba-2 (SSD): parallel structured scan via chunked attention
- Jupyter notebooks with visual walkthroughs of each component
- Benchmark suite vs `mamba-ssm` on Long-Range Arena
- GitHub Pages site with full SEO (deferred to v0.2.0)

### Added (unreleased)
- `llms.txt` at repo root: AI-crawler-readable project summary following
  the llmstxt.org convention.
- 12 GitHub topics set on the repo for search and topic-page
  discoverability.

## [0.1.0] - 2026-08-19

### Added
- HiPPO-LegS matrix (`hippo_legs(N)`) with closed-form entries from
  Gu, Goel, Re (2022), Appendix B.
- Zero-order-hold and bilinear (Tustin) discretization routines with
  gradient flow through the learnable step size.
- `S4Layer` PyTorch module implementing the structured state-space model
  in two equivalent views:
  - Recurrent view: O(L) per step, used for autoregressive generation.
  - Convolutional view: O(L log L) via FFT, used for parallel training.
- Test suite (13 tests, all passing) verifying:
  - HiPPO-LegS matrix structure, diagonal, off-diagonal, and stability
    (all eigenvalues have negative real part).
  - Discretized `Abar` eigenvalues inside the unit disk across step sizes
    0.001, 0.01, 0.1, 1.0.
  - Output shape correctness for both views.
  - Recurrent and convolutional views produce identical outputs to 1e-6
    in float64 (the central S4 theorem).
  - Gradient flow through C, D, and log_step.
  - Long sequence (L=1024) produces no NaNs or Infs.
- Smoke test script (`scripts/smoke_test.py`).
- MIT license.
- README with motivation, design principles, and verified guarantees.
- Research notes (`docs/RESEARCH.md`) surveying existing implementations.

### Fixed
- FFT convolution was using flipped-kernel cross-correlation semantics and
  taking the wrong slice of the IFFT output. Corrected to a straight
  convolution truncated to the first L samples.
- `self.step.item()` was detaching the learnable `log_step` parameter from
  the autograd graph. Replaced with a tensor-preserving `discretize()` so
  gradients flow through the step size.

[Unreleased]: https://github.com/sailikhithk/mamba-from-scratch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sailikhithk/mamba-from-scratch/releases/tag/v0.1.0
