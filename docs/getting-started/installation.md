# Installation

## Prerequisites

- **Python ≥ 3.11** (3.12 recommended).
- [**uv**](https://github.com/astral-sh/uv) — the package manager used by
  this project.
- **gfortran** — required for the deterministic syntax-check validation
  step. `brew install gcc` on macOS, `apt install gfortran` on Debian/Ubuntu.
- **NVIDIA HPC SDK** (`nvfortran`) — *optional*, required only to validate
  the generated GPU code locally. On HPC sites a module load is usually
  enough; on a laptop, prefer running the GPU validation in a cloud VM.
- An **OpenAI-compatible LLM endpoint** — Mistral La Plateforme, or any
  self-hosted vLLM / TGI / Ollama server. See
  [LLM endpoints](../concepts/llm-endpoints).

## From source (recommended)

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env       # fill in MISTRAL_ENDPOINT / MISTRAL_API_KEY
uv sync
```

The `uv sync` step installs all runtime dependencies into a project-local
virtual environment (`.venv/`). Activate it with `source .venv/bin/activate`
or prefix commands with `uv run`.

## Optional extras

The default `uv sync` installs only the **core** dependencies (Loki +
NumPy + LangGraph + python-dotenv, ~50 MB). That is enough to run
`agent-analyze` — the analyze-only mode. Pull in extras for the other
agents:

| Extra      | What it adds                                                | When to use                                  |
| ---------- | ----------------------------------------------------------- | -------------------------------------------- |
| *(none)*   | core: Loki, NumPy, python-dotenv, LangGraph                 | `agent-analyze` (CI / pre-flight)            |
| `cpu`      | alias for "no extras" — discoverable for CI scripts         | Same as above, explicit for `uv sync --extra cpu` |
| `gpu`      | LangChain stack + Cython                                    | `agent-gpu` (Fortran → GPU + Cython, Phase 1)|
| `mcp`      | FastMCP + `[gpu]`                                           | `run-mcp` (HTTP/SSE server in IDEs / CI)     |
| `jax`      | JAX, Flax, Equinox                                          | `agent-translate` / `agent-pipeline --to jax`|
| `all`      | `[gpu]` + `[mcp]` + `[jax]`                                 | Full developer install                       |
| `docs`     | Sphinx + Furo + MyST + extensions                           | Build this documentation site                |
| `tests`    | pytest, pytest-cov                                          | Run the test suite                           |

```bash
# Analyze-only (smallest footprint — CI / pre-commit)
uv sync                       # or: uv sync --extra cpu

# Full transformation pipeline (Phase 1 + Phase 2)
uv sync --extra all

# Anything in between
uv sync --extra gpu           # Phase 1 only — no MCP, no JAX
uv sync --extra mcp           # MCP server (pulls [gpu])
uv sync --extra jax           # Phase 2 only
uv sync --extra docs --extra tests
```

## Compiler detection

`agent-analyze` probes `PATH` for a Fortran compiler at every run and
reports its OpenACC capability. No flag is needed; disable with
`--no-toolchain-check`:

```text
Toolchain:
  gfortran   13.2.0       family=gnu            openacc=experimental (-fopenacc)
  nvfortran  24.5         family=nvidia         openacc=native (-acc)
  → recommended for GPU port: nvfortran 24.5
```

Two new findings come from this check:

- **FORT010** (warning) — no Fortran compiler on PATH; generated code can't
  be validated even via `gfortran -fsyntax-only`.
- **FORT011** (warning) — source uses `!$acc` pragmas but no compiler
  understands them. Fires when `nvfortran` is missing and `gfortran` is
  too old (< 7) or absent.

## Docker

A multi-stage `Dockerfile` is shipped at the repo root and a compose file
runs the MCP server on `http://localhost:8000`:

```bash
docker compose up --build
```

For HPC sites use `Dockerfile.hpc` (CUDA + NVIDIA HPC SDK) or build the
Apptainer image from `apptainer.def`.

## Verifying the install

```bash
uv run run-mcp --help
uv run agent-gpu --help
```

If both commands print their usage, you're ready. Continue to the
[Quickstart](quickstart.md).
