# Quickstart

This page walks you through porting a small Fortran 90 kernel to GPU and
calling it from Python in about five minutes.

## 1. Configure the LLM endpoint

Edit `.env` to point at your endpoint:

```bash
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<your-key>"
MISTRAL_MODEL="mistral-large-latest"
```

See [LLM endpoints](../concepts/llm-endpoints) for self-hosted alternatives.

## 2. Run Phase 1 — Fortran → GPU + Cython

Pick a Fortran source — for example a 2-D finite-difference stencil:

```bash
uv run agent-gpu path/to/kernel.f90
```

The pipeline writes its results under `output/`:

```text
output/
├── fortran_gpu/
│   ├── kernel_pure.f90      # PURE/ELEMENTAL annotated
│   └── kernel_gpu.f90       # OpenACC pragmas
└── cython/
    ├── kernel_wrapper.pyx   # NumPy-typed wrapper
    ├── kernel_c.h           # iso_c_binding header
    └── setup.py             # scikit-build entry point
```

A typical run consumes four LLM calls and about two minutes of wall-clock.

## 3. Run Phase 2 — Fortran → JAX (optional)

```bash
uv run agent-pipeline path/to/kernel.f90 --to jax
```

This produces a JAX translation under `output/jax/` that is JIT-compilable
and differentiable through `jax.grad` / `jax.vjp`.

## 4. Analyze-only mode (no LLM, CI hook)

If you just want to know whether a Fortran file is GPU-ready — without
spending tokens or rewriting anything — use `agent-analyze`:

```bash
uv run agent-analyze src/kernel.f90
uv run agent-analyze --format sarif --output report.sarif src/
uv run agent-analyze --fail-on warning src/   # CI-friendly exit code
```

The analyzer runs only the deterministic Loki parser stage, detects nine
recurring patterns (COMMON, SAVE, I/O in kernels, missing IMPLICIT NONE,
POINTER, derived types, suspected loop-carried deps, missing KIND,
parse errors), and emits human-readable text, JSON, or **SARIF 2.1.0**
that GitHub Code Scanning consumes for inline PR annotations.

A ready-to-use workflow lives at
[`.github/workflows/analyze.yml`](https://github.com/maurinl26/fortranspire/blob/main/.github/workflows/analyze.yml);
a lightweight HPC container (no CUDA, no NVIDIA HPC SDK) lives at
[`Apptainer.analyze`](https://github.com/maurinl26/fortranspire/blob/main/Apptainer.analyze).

## 5. Use the MCP server from an IDE

```bash
uv run run-mcp
```

The server listens on `http://localhost:8000/sse` and exposes the
`translate_kernel_gpu`, `translate_kernel`, `profile_kernels`, and
`ask_agent` tools. Connect any MCP-aware client (Claude Desktop, Cursor,
VS Code agent, Mistral Le Chat) to drive the pipeline from your editor.

Set `API_KEY` in your environment to require a bearer token on every
request.

## Next steps

- Read the [Architecture](../concepts/architecture) page to understand the
  six pipeline stages and where the LLM is — and is not — involved.
- Skim the [Fortran patterns](../concepts/fortran-patterns) page to see the
  nine recurring patterns the pipeline handles automatically.
- Browse the [API reference](../api/index) for the Python entry points.
