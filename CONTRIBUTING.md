# Contributing to `fortranspire`

Thanks for considering a contribution. This project ports legacy Fortran HPC
code to GPU (OpenACC) and optionally to JAX, exposed through a Model Context
Protocol (MCP) server. Contributions of all sizes are welcome — bug reports,
documentation fixes, new Fortran transformation patterns, additional LLM
backends, tests, or example kernels.

This file is intentionally short. If you can't find what you need here,
open a [Discussion](https://github.com/maurinl26/fortranspire/discussions)
or an [Issue](https://github.com/maurinl26/fortranspire/issues).

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md)
(Contributor Covenant 2.1). Report any unacceptable behavior to
`maurin.loic.ac@gmail.com`.

## Ways to contribute

- **Report a bug** — open an issue with a minimal Fortran kernel that triggers
  the problem and the full log of the failing pipeline step.
- **Add a Fortran transformation pattern** — see
  [`README.md` §8](README.md) for the existing 9 patterns. New patterns should
  come with a test kernel under `tests/fixtures/` and an end-to-end check that
  the pipeline produces compilable Fortran + Cython output.
- **Add an LLM backend** — anything OpenAI-compatible plugs in via
  `fortranspire/llm/`. A new backend should be selectable through
  environment variables and documented in the README.
- **Improve docs** — Sphinx sources live under `docs/`; the rendered site is
  published on Read the Docs.
- **Help with deployment recipes** — devcontainer, OpenStack tenants,
  Apptainer for HPC sites. See the
  [deployment roadmap section of the README](README.md).

## Development setup

Pre-requisites: Python ≥ 3.11, [`uv`](https://github.com/astral-sh/uv),
`gfortran` for the deterministic validation step, and optionally
`nvfortran` (NVIDIA HPC SDK) for GPU compilation checks.

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env        # fill in your Mistral / vLLM endpoint + API key

# Pick an install profile (default = analyze-only, no LLM, ~50 MB):
uv sync                       # core only — agent-analyze works
uv sync --extra gpu           # + LLM + Cython (Phase 1 transformation)
uv sync --extra all           # everything (Phase 1 + Phase 2 + MCP server)
uv sync --extra docs --extra tests   # docs build + test suite
```

See [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
for the full extras matrix.

Run the test suite:

```bash
uv run pytest
```

Render the docs locally:

```bash
uv run --extra docs sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## Pull request workflow

1. Fork the repo and create a topic branch off `main`
   (`git checkout -b fix/intent-detection`).
2. Make your change. Keep commits focused and write a commit message that
   explains the *why*.
3. Add or update tests. Pipeline changes must come with at least one Fortran
   fixture that exercises the new path end-to-end.
4. Run the test suite locally and make sure `uv run pytest` is green.
5. Update documentation if you changed user-visible behavior, CLI flags, or
   the MCP tool surface.
6. Open a PR against `main`. Describe what the change does, why it is needed,
   and how you tested it. Link any related issue.

A maintainer will review within ~1 week. Smaller PRs get merged faster — if
you are tackling something large, please open an issue first to align on the
approach.

## Style

- Python: ruff-formatted, type-annotated. `uv run ruff check .` and
  `uv run ruff format .` before pushing.
- Fortran fixtures: free-form `.f90`, four-space indent, lowercase keywords.
- Commit messages: imperative mood, ≤ 72 chars in the subject line.
- Docs: Markdown (MyST) or reStructuredText, hard-wrap at 88 cols where
  practical.

## Tests

- Unit tests live next to the code they exercise (`tests/` mirrors the
  package layout).
- End-to-end pipeline tests use small Fortran kernels under
  `tests/fixtures/` and assert that (a) the generated Fortran compiles with
  `gfortran -fsyntax-only`, (b) the generated Cython wrapper builds, and
  (c) outputs match the reference solution within numerical tolerance.
- Tests that need an LLM call should be marked `@pytest.mark.llm` and
  skipped by default in CI unless a mock backend is wired up.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities.
See [SECURITY.md](SECURITY.md) for private disclosure instructions.

## Licensing

By submitting a contribution, you agree that your work will be licensed
under the [Apache License 2.0](LICENSE), the same license as the rest of the
project. No CLA is required.
