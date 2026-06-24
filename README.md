<div align="center">

# 🚀 fortranspire

**An LLM + Model Context Protocol pipeline that ports legacy Fortran
HPC kernels to GPU (OpenACC) and differentiable JAX.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/fortranspire.svg)](https://pypi.org/project/fortranspire/)
[![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)
[![Documentation Status](https://readthedocs.org/projects/fortranspire/badge/?version=latest)](https://fortranspire.readthedocs.io/en/latest/?badge=latest)
[![JOSS draft](https://github.com/maurinl26/fortranspire/actions/workflows/draft-paper.yml/badge.svg)](https://github.com/maurinl26/fortranspire/actions/workflows/draft-paper.yml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

---

`fortranspire` automates the four transformations a senior HPC engineer
would perform by hand when porting a Fortran 90 kernel to GPU and
Python: AST parsing with [Loki](https://github.com/ecmwf-ifs/loki)
(ECMWF), kernel extraction into a `MODULE` with explicit `INTENT`,
`PURE`/`ELEMENTAL` annotation, OpenACC pragma insertion, and a Cython
wrapper validated by `gfortran` and `nvfortran`. The pipeline is
exposed both as a CLI and as an MCP server, so it is callable from
mistral-vibe, Claude Code, Claude Desktop, Cursor, or any MCP-aware
agent — and from CI.

### 📑 Table of contents

- [⚡ Quick start with mistral-vibe](#-quick-start-with-mistral-vibe)
- [🎯 What this solves](#-what-this-solves)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🛠️ CLI usage](#️-cli-usage)
- [🔌 IDE integrations](#-ide-integrations)
- [🔬 Fortran patterns handled](#-fortran-patterns-handled)
- [🗺️ Roadmap](#️-roadmap)
- [📚 Documentation, citation, contribution](#-documentation-citation-contribution)
- [📬 Contact](#-contact)
- [📜 License](#-license)

---

## ⚡ Quick start with mistral-vibe

`fortranspire` ships an MCP server you can plug straight into
[`mistral-vibe`](https://github.com/mistralai/mistral-vibe) over stdio
— **no port, no auth, the inference stays in EU.**

```bash
uv tool install fortranspire          # console script on PATH, isolated venv
brew install mistral-vibe
```

Find the absolute path of the installed binary — vibe is a brew
subprocess and may not inherit your shell `PATH`:

```bash
which fortranspire                    # e.g. /Users/you/.local/bin/fortranspire
```

Then append to `~/.vibe/config.toml`:

```toml
[[mcp_servers]]
name      = "fortranspire"
transport = "stdio"
command   = "/Users/you/.local/bin/fortranspire"   # paste output of `which`
args      = ["mcp", "--stdio"]

[mcp_servers.env]
MISTRAL_API_KEY  = "${MISTRAL_API_KEY}"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1"
MISTRAL_MODEL    = "codestral-latest"
```

Launch vibe in a Fortran project and ask, in plain French or English:

> *Donne-moi le coût estimé de portage GPU pour
> `src/common/turb/mode_compute_function_thermo.F90`.*

In ~2 seconds, vibe calls the `fortranspire_explain_port_cost` MCP
tool and prints a port-cost table — routines, tokens, USD estimate,
FORT00x risks — with **zero LLM tokens consumed by fortranspire**.

Full walkthrough (PHYEX kernel triage → call-graph → Phase-1 OpenACC
port) in [`docs/getting-started/with-mistral-vibe.md`](docs/getting-started/with-mistral-vibe.md).

> [!NOTE]
> **Why this matters.** The codebase stays on disk, Mistral codestral
> runs in EU, Loki + the deterministic harness absorb 60-70 % of the
> work — the LLM is a fungible component, swappable for Claude or
> self-hosted vLLM.

---

## 🎯 What this solves

Legacy scientific HPC codes total several million lines of Fortran 90
written between 1990 and 2010 across the major European research
institutions:

- **Numerical weather prediction** — IFS at ECMWF, ARPEGE and AROME at
  Météo-France, PHYEX at CNRM, SURFEX (the demo target used throughout
  this documentation).
- **Energy & defence physics** — `code_aster` and Code\_Saturne at EDF
  (structural mechanics, CFD), neutron-transport and plasma-physics
  codes at CEA (TRIPOLI, MCNP-style chains), reservoir simulators at
  the major energy research labs.
- **Geophysics** — seismic imaging (`seismic_CPML`, full-waveform
  inversion frontends), oceanography (NEMO and forks).

All of these run on CPU clusters (BullSequana, Cray EX) while the GPU
partitions of EuroHPC (LUMI, MareNostrum, Jean Zay) sit underused —
not for lack of hardware, but for lack of ported code.

A manual port of one production-grade kernel to GPU takes a senior HPC
engineer 2–6 weeks. The skill profile — Fortran 90 + OpenACC + Cython
+ numerics — is structurally rare in Europe, and the engineers who
wrote these codes are retiring.

`fortranspire` automates that port: deterministic AST analysis
(Loki) for 60-70% of the work, a small number of targeted LLM calls
for the semantic stages (kernel extraction, OpenACC pragmas, Cython
wrapping). The same toolchain produces JAX-translatable kernels (Phase
2) — opening legacy physics to differentiable training, RL feedback
loops, and surrogate modelling.

---

## 🏗️ Architecture

A six-stage LangGraph state machine. Two stages call the LLM; four are
deterministic.

```text
📂 kernel.f90  (monolithic Fortran)
     │
     ▼ 🔍 parser          Loki AST — detects INTENT, SAVE, COMMON, loops, I/O
     │                    Deterministic, no LLM
     │
     ▼ 🔧 extractor       LLM (1 call) — refactors PROGRAM into a MODULE of
     │                    subroutines; removes COMMON, surfaces SAVE as
     │                    INTENT(INOUT)
     │
     ▼ ✨ pure_elemental   AST rules — annotate PURE/ELEMENTAL where legal
     │                    (no I/O, no SAVE, INTENT explicit)
     │
     ▼ 🚀 openacc         LLM (1 call) — !$acc parallel loop collapse(2),
     │                    !$acc data copyin/copy around the time loop
     │
     ▼ 🐍 cython_wrapper  LLM (2 calls) — .pyx + iso_c_binding C header,
     │                    NumPy typed memoryviews
     │
     ▼ ✅ validation       gfortran × 2 flavours + nvfortran -acc
     │                    Deterministic, compilation only
     │
     📦 output/fortran_gpu/*.f90  +  output/cython/*.pyx
```

Four LLM calls maximum per kernel (~2 min wall-clock, ~$0.06 at
codestral tariffs). See
[`docs/concepts/architecture.md`](docs/concepts/architecture.md) for
the activation-order rationale and worked examples.

---

## 📦 Installation

### From PyPI (recommended)

```bash
uv tool install fortranspire                 # console script, isolated venv
# or, inside a project venv:
uv pip install fortranspire                  # core + Loki (via loki-ifs)
uv pip install "fortranspire[gpu]"           # Phase 1 (LangChain + Cython)
uv pip install "fortranspire[mcp]"           # MCP server
uv pip install "fortranspire[all]"           # Phase 1 + Phase 2 + MCP
```

> [!IMPORTANT]
> **About Loki.** ECMWF Loki is not published on PyPI under its
> original name (the slot is taken by an unrelated astronomy package).
> We maintain a redistribution under the name
> [`loki-ifs`](https://pypi.org/project/loki-ifs/), tracking
> [`ecmwf-ifs/loki@0.3.7`](https://github.com/maurinl26/loki-ifs). The
> Python import name is still `from loki import …` — `loki-ifs` is
> the PyPI distribution name only. This redistribution is maintained
> **until ECMWF publishes an official Loki distribution on PyPI**, at
> which point fortranspire will switch its dependency over and the
> `loki-ifs` package will be deprecated.

### From source

```bash
git clone https://github.com/maurinl26/fortranspire
cd fortranspire
cp .env.example .env
uv sync --extra all
```

### Connect to a Mistral endpoint

```bash
# .env
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<your-key-from-console.mistral.ai>"
MISTRAL_MODEL="codestral-latest"             # default for code-gen stages
```

For self-hosted inference (vLLM / TGI / Ollama on an OpenStack tenant
or on-prem GPU cluster), point `MISTRAL_ENDPOINT` at your
OpenAI-compatible server — see
[`docs/concepts/llm-endpoints.md`](docs/concepts/llm-endpoints.md).

---

## 🛠️ CLI usage

```bash
fortranspire analyze src/                    # Loki AST audit — no LLM, CI-friendly
fortranspire explain src/kernel.f90          # Pre-flight cost & risk estimate
fortranspire graph -o graph.md src/          # Mermaid call-graph
fortranspire doc --no-llm src/               # Inject !> docstring placeholders
fortranspire gpu src/kernel.f90              # Phase 1 — Fortran → OpenACC + Cython
fortranspire translate src/kernel.f90        # Phase 2 — Fortran → JAX (experimental)
fortranspire mcp --stdio                     # MCP server, stdio transport
fortranspire mcp                             # MCP server, SSE on $MCP_HOST:$MCP_PORT
```

> [!TIP]
> Always call `explain` before a paid `gpu`/`translate` — it tells
> you whether the file is portable and what a full port would cost in
> tokens. Zero LLM tokens are consumed by `explain`, `analyze`,
> `graph`, or `doc --no-llm`.

---

## 🔌 IDE integrations

### mistral-vibe

See the [quick start](#-quick-start-with-mistral-vibe)
above, or the
[full walkthrough](docs/integrations/mistral-vibe.md) covering stdio
config, HTTP/SSE deployment for permanent service, and the PHYEX demo.

### Claude Code

A turnkey skill is shipped at
[`skills/claude-code/fortranspire/`](skills/claude-code/fortranspire).
Copy it into your project's `.claude/skills/` and the `/fortranspire`
slash command appears in Claude Code. The skill instructs Claude to
always run `explain` first.

For MCP-based use (rather than skill-based), register the same stdio
server:

```bash
claude mcp add fortranspire --scope user \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e MISTRAL_ENDPOINT=https://api.mistral.ai/v1 \
  -e MISTRAL_MODEL=codestral-latest \
  -- $(which fortranspire) mcp --stdio
```

See [`docs/integrations/claude-code.md`](docs/integrations/claude-code.md).

### MCP tool surface

Nine tools exposed by the server:

| Tool | Purpose | LLM call? |
| ---- | ------- | --------- |
| `analyze_kernels` | Loki AST analysis, optional SARIF output | No |
| `explain_port_cost` | Pre-flight cost & risk estimate | No |
| `build_call_graph` | Mermaid call-graph | No |
| `generate_docs` | `!>` docstring injection (no-LLM or LLM-driven) | Optional |
| `translate_kernel_gpu` | Phase 1 — Fortran → OpenACC + Cython | Yes |
| `translate_kernel` | Phase 2 — Fortran → JAX ⚠️ experimental | Yes |
| `profile_kernels` | Performance benchmarking | No |
| `ask_agent` | Natural-language query against the codebase | Yes |
| `agent_status` | Dump server config | No |

---

## 🔬 Fortran patterns handled

Eleven recurring patterns from production seismic and atmospheric
codes are documented in
[`docs/concepts/fortran-patterns.md`](docs/concepts/fortran-patterns.md):

`INTENT`, `COMMON`, `SAVE`, `POINTER`, AoS → SoA + `collapse`,
stencil-vs-recurrence dependencies, `ELEMENTAL` + `!$acc routine seq`,
explicit `KIND` types, `LOGICAL PARAMETER` flags → `#ifdef`, MPI halo
exchange → GHEX (Phase 3), and Fortran I/O → xarray/zarr + DLPack
(Phase 4).

> [!TIP]
> If you hit a pattern not on that list, please open an
> [issue](https://github.com/maurinl26/fortranspire/issues) with a
> minimal reproducer — adding a new pattern is usually a one-stage
> change plus a fixture.

---

## 🗺️ Roadmap

The strategic roadmap, technical phases, and deployment surfaces are
tracked in [**ROADMAP.md**](ROADMAP.md). Quick view:

| Phase | Target | Status |
| ----- | ------ | ------ |
| 1 — OpenACC + Cython | Fortran → GPU + Python wrapper | **Shipped (0.1.0)** |
| 1.5 — OpenMP target | Multi-vendor pragmas (gfortran 13+, nvfortran, ifx) | Shipped (0.1.0) |
| 1.6 — GT4Py.next | Python cartesian + unstructured DSL (FVM / ICON) | Scoped — [#42](https://github.com/maurinl26/fortranspire/issues/42) |
| 2 — JAX | Differentiable functional kernels | Partial (0.2.x) |
| 3 — GHEX | GPU-to-GPU MPI communications | Future |
| 4 — Modern I/O | xarray / zarr cloud-native | Future |
| 5 — Neural surrogates | FNO operators trained on GPU outputs | Future |

Work in progress is tracked publicly on the
[issue tracker](https://github.com/maurinl26/fortranspire/issues).

---

## 📚 Documentation, citation, contribution

- **Documentation**: [fortranspire.readthedocs.io](https://fortranspire.readthedocs.io/) (Sphinx)
- **JOSS paper draft**: [`paper.md`](paper.md) — build via the
  `draft-paper` GitHub Action
- **Citation**: see [`CITATION.cff`](CITATION.cff) /
  [`codemeta.json`](codemeta.json) / [`.zenodo.json`](.zenodo.json).
  Once a DOI is assigned, the badge above will resolve.
- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md) — testing, PR
  conventions, pattern-fixture format.
- **Code of conduct**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) —
  Contributor Covenant 2.1.

> [!IMPORTANT]
> **Beta testers welcome.** If you maintain a Fortran legacy code
> (weather/climate, structural mechanics, CFD, plasma, seismic,
> reactor physics, oceanography) and a `.f90` file breaks the
> pipeline, please open an issue with the file attached — adding
> fixture-driven coverage is the fastest path to making the pipeline
> more robust.

---

## 📬 Contact

[Loïc Maurin](https://www.linkedin.com/in/lo%C3%AFc-maurin/) —
Independent Researcher, Toulouse, France —
ORCID [0009-0004-8117-4850](https://orcid.org/0009-0004-8117-4850).

Email: [maurin.loic.ac@gmail.com](mailto:maurin.loic.ac@gmail.com) —
for collaborations, JOSS-related questions, custom HPC porting
missions on legacy Fortran codes (NWP, structural mechanics, CFD,
plasma, reactor physics, seismic), or self-hosted deployment
assistance.

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).
