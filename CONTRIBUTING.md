# Contributing

Thanks for your interest in `mamba-from-scratch`. This project prioritizes
clarity and verified correctness over performance. The conventions below
keep the codebase honest and the history readable.

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Given `MAJOR.MINOR.PATCH`:

- **MAJOR** - incompatible API changes (a layer's forward signature changes,
  a public function is removed or renamed).
- **MINOR** - new components or features, backwards compatible
  (a new layer, a new discretization method, a new benchmark).
- **PATCH** - backwards compatible bug fixes
  (a numerics fix, a test that was wrong, a doc correction).

While the project is `0.x`, the API may change between MINOR bumps. Once we
hit `1.0.0`, SemVer applies strictly.

### Where the version lives

The canonical version string is in [`pyproject.toml`](pyproject.toml) under
`[project] version`. It is mirrored into `src/s4/__init__.py` as
`__version__` for runtime access. The two must always agree.

### Releases

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/s4/__init__.py`.
2. Add a new section to [`CHANGELOG.md`](CHANGELOG.md) under the new
   version, dated `YYYY-MM-DD`, with Added / Changed / Fixed / Removed
   subsections as needed.
3. Move any unfinished planned items from the new section down to
   `[Unreleased]`.
4. Commit: `Release vX.Y.Z`.
5. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
6. Push the commit and the tag: `git push && git push --tags`.
7. Create a GitHub Release from the tag, pasting the CHANGELOG section as
   the release notes.

## Changelog format

[`CHANGELOG.md`](CHANGELOG.md) follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release has:

- A `## [X.Y.Z] - YYYY-MM-DD` header.
- Subsections: `### Added`, `### Changed`, `### Deprecated`,
  `### Removed`, `### Fixed`, `### Security` (only those used).
- A link line at the bottom comparing to the previous release.

## Commit messages

- Imperative mood: "Add HiPPO matrix", not "Added HiPPO matrix".
- Subject line under 72 characters.
- Body explains why, not what. The diff already shows what.
- No AI-tool attribution ("Generated with ...", "Co-Authored-By: ...").
  All work is authored by the human committer.
- No em-dashes, en-dashes, or minus signs (U+2014, U+2013, U+2212). Use a
  hyphen, colon, comma, or semicolon. A pre-commit hook enforces this.

## Tests

Every new component ships with tests that verify the theoretical
guarantees, not just shapes. For an SSM layer, that means at minimum:

- Output shape correctness.
- The two computational views agree (recurrent vs convolutional) to a
  stated tolerance in float64.
- Gradient flow through every learnable parameter.
- Long-sequence stability (no NaNs / Infs at L >= 1024).

Run the suite before pushing:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Code style

- Pure PyTorch plus NumPy. No `mamba-ssm`, no `causal-conv1d`, no Triton
  in the core library. Optional dependencies live under
  `[project.optional-dependencies] bench`.
- Type hints on every public function.
- Docstrings on every public function and class, including the math when
  the function implements an equation.
- Line length 100 (enforced by ruff config in `pyproject.toml`).
- One component per module. `hippo.py` is the math, `layer.py` is the
  module. Do not mix them.

## Pull requests

- Branch from `main`.
- One logical change per PR. A bug fix and a new feature are two PRs.
- The PR description references the CHANGELOG entry it adds.
- All tests must pass on the PR branch.
