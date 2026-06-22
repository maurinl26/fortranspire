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
pip install fortranspire                # core + Loki (via loki-ifs) — analyze
pip install "fortranspire[gpu]"         # + Phase 1 transformations
pip install "fortranspire[mcp]"         # + MCP HTTP/SSE server
pip install "fortranspire[all]"         # everything except docs/tests
```

### How Loki is shipped on PyPI

ECMWF Loki is not published on PyPI under that name (the PyPI `loki` is an
unrelated astronomy package), and [PEP 715] forbids direct URL dependencies
(`git+https://…`) inside published metadata.

We maintain an *unofficial PyPI redistribution* of upstream Loki under the
name **[`loki-ifs`](https://pypi.org/project/loki-ifs/)** at upstream tag
`0.3.7`, source at https://github.com/maurinl26/loki-ifs. `fortranspire`
depends on `loki-ifs>=0.3.7` as a regular PyPI dep, so `pip install
fortranspire` resolves Loki in one step.

The Python import name remains `loki` — code written for upstream ECMWF
Loki imports unchanged:

```python
from loki import Sourcefile, Frontend, FindNodes, BasicType
```

If you prefer to install Loki yourself (e.g. against a different upstream
tag or a private fork), you can override the PyPI dep:

```bash
pip install --no-deps fortranspire
pip install "loki @ git+https://github.com/ecmwf-ifs/loki@<tag>"   # in your own project
```

[PEP 715]: https://peps.python.org/pep-0715/

## From source (recommended for development)

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env       # fill in MISTRAL_ENDPOINT / MISTRAL_API_KEY
uv sync                    # core + Loki (via loki-ifs from PyPI)
```

Activate the venv with `source .venv/bin/activate` or prefix commands with
`uv run`.

## Optional extras

The default install (`uv sync`) gives you the **core** plus the Loki AST
toolkit (~50 MB). That is enough to run `fortranspire analyze` — the
analyze-only mode. Pull in extras for the other agents:

| Extra      | What it adds                                                | When to use                                  |
| ---------- | ----------------------------------------------------------- | -------------------------------------------- |
| *(none)*   | core: loki-ifs, NumPy, python-dotenv, LangGraph             | `fortranspire analyze` (CI / pre-commit)     |
| `cpu`      | alias for "no extras" — discoverable for CI scripts         | Same as above, explicit                      |
| `gpu`      | LangChain stack + Cython                                    | `fortranspire gpu` (Phase 1)                 |
| `mcp`      | FastMCP + `[gpu]`                                           | `run-mcp` (HTTP/SSE server in IDEs / CI)     |
| `jax`      | JAX, Flax, Equinox                                          | `fortranspire translate` / Phase 2           |
| `all`      | `[gpu]` + `[mcp]` + `[jax]`                                 | Full developer install                       |
| `docs`     | Sphinx + Furo + MyST + extensions                           | Build this documentation site                |
| `tests`    | pytest, pytest-cov                                          | Run the test suite                           |

```bash
# Analyze-only (CI / pre-commit)
uv sync                          # core + Loki

# Full transformation pipeline (Phase 1 + Phase 2)
uv sync --extra all

# Anything in between
uv sync --extra gpu              # Phase 1 only
uv sync --extra mcp              # MCP server (pulls [gpu])
uv sync --extra jax              # Phase 2 only
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
