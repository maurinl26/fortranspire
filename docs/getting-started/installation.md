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

## From PyPI

```bash
pip install fortranspire                                              # core
pip install "loki @ git+https://github.com/ecmwf-ifs/loki@0.3.7"     # AST (see below)
pip install "fortranspire[gpu]"                                       # Phase 1 transforms
```

### Why is Loki a separate command?

ECMWF Loki is not published on PyPI under that name (the PyPI `loki` is an
unrelated astronomy package), and [PEP 715] forbids direct URL dependencies
(`git+https://…`) inside published metadata. We therefore ship fortranspire
with **no Loki dependency declared** and ask users to install it in a
second step. The parser falls back to a regex frontend when Loki is absent,
so the package imports cleanly — but `analyze` is significantly more
accurate with Loki present.

[PEP 715]: https://peps.python.org/pep-0715/

## From source (recommended for development)

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env       # fill in MISTRAL_ENDPOINT / MISTRAL_API_KEY
uv sync --group loki       # core + Loki (resolved via [tool.uv.sources])
```

The `--group loki` flag activates the PEP 735 dependency group declared in
`pyproject.toml`; uv resolves it transparently to the pinned git tag
(`0.3.7`). Dependency groups are not part of the published PyPI metadata,
so they do not violate PEP 715.

Activate the venv with `source .venv/bin/activate` or prefix commands with
`uv run`.

## Optional extras

The default install (`uv sync --group loki`) gives you the **core** plus
the Loki AST toolkit (~50 MB). That is enough to run `fortranspire
analyze` — the analyze-only mode. Pull in extras for the other agents:

| Extra      | What it adds                                                | When to use                                  |
| ---------- | ----------------------------------------------------------- | -------------------------------------------- |
| *(none)*   | core: NumPy, python-dotenv, LangGraph                       | imports + parser regex fallback              |
| `cpu`      | alias for "no extras" — discoverable for CI scripts         | Same as above, explicit                      |
| `gpu`      | LangChain stack + Cython                                    | `fortranspire gpu` (Phase 1)                 |
| `mcp`      | FastMCP + `[gpu]`                                           | `run-mcp` (HTTP/SSE server in IDEs / CI)     |
| `jax`      | JAX, Flax, Equinox                                          | `fortranspire translate` / Phase 2           |
| `all`      | `[gpu]` + `[mcp]` + `[jax]`                                 | Full developer install                       |
| `docs`     | Sphinx + Furo + MyST + extensions                           | Build this documentation site                |
| `tests`    | pytest, pytest-cov                                          | Run the test suite                           |

```bash
# Analyze-only (CI / pre-commit)
uv sync --group loki                              # core + Loki AST

# Full transformation pipeline (Phase 1 + Phase 2)
uv sync --group loki --extra all

# Anything in between
uv sync --group loki --extra gpu                  # Phase 1 only
uv sync --group loki --extra mcp                  # MCP server (pulls [gpu])
uv sync --group loki --extra jax                  # Phase 2 only
uv sync --extra docs --extra tests                # docs + tests, no Loki needed
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
