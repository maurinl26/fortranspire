---
name: fortranspire
description: Use this skill when the user asks to analyze, port to GPU, document, format, or otherwise transform legacy Fortran 90 source. Covers static analysis (Loki AST, no LLM), inline documentation generation, OpenACC/OpenMP GPU porting, Cython wrappers, Fortran→JAX translation, call-graph reports, and pipeline benchmarks. Trigger phrases: "this Fortran file", "port to GPU", "analyze .f90", "document this legacy code", "explain port cost", "convert to JAX".
---

# fortranspire skill

`fortranspire` ports legacy Fortran 90 HPC kernels through a deterministic
pipeline (Loki AST → PURE/ELEMENTAL → OpenACC/OpenMP → Cython wrapper),
with optional Phase 2 translation to differentiable JAX. It runs entirely
on the user's machine, talking to any OpenAI-compatible LLM endpoint
(Mistral La Plateforme by default — sovereign EU compute).

## When to use this skill

Invoke this skill when the user wants to operate on Fortran code in one of
the following ways:

| User intent                                              | Subcommand          |
| -------------------------------------------------------- | ------------------- |
| Static analysis / lint / CI gate                         | `analyze`           |
| Pre-flight cost & risk estimate (no LLM, no tokens)      | `explain`           |
| Generate inline `!>` docstrings + optional Sphinx site   | `doc`               |
| Format Fortran source (fprettify-based)                  | `format`            |
| Mermaid call-graph report                                | `graph`             |
| Semantic before/after diff (text or HTML)                | `diff`              |
| HTML audit dashboard for a Phase-1 output directory      | `report`            |
| Pipeline-output benchmark + regression detector          | `bench`             |
| **Port one file** to GPU (OpenACC) + Cython              | `gpu`               |
| **Port many files** in parallel                          | `port-batch`        |
| Fortran → JAX (experimental, differentiable)             | `translate`         |
| Performance benchmarking against the original kernel     | `profile`           |
| Fortran → gt4py.next field operators                     | `gt4py`             |
| Domain geometry + MPI decomposition (interactive)        | `domain`            |
| Start the MCP HTTP/SSE server                            | `mcp`               |

## Prerequisites — check before invoking

1. **Is fortranspire installed?** Probe with `fortranspire --help`. If it
   fails, propose: `pip install fortranspire` (analyze/explain/format/doc
   work with this alone) or `pip install "fortranspire[gpu]"` (Phase 1
   transformations needing a LangChain LLM stack).
2. **For LLM-driven verbs (`gpu`, `port-batch`, `translate`, `doc` with
   `--with-llm`)**: an OpenAI-compatible endpoint must be configured. Check
   `MISTRAL_ENDPOINT` and `MISTRAL_API_KEY` in `.env` (or
   `OPENAI_API_BASE`/`OPENAI_API_KEY` for a generic endpoint). If absent,
   ask the user whether to use Mistral La Plateforme or a self-hosted vLLM
   / TGI / Ollama endpoint, then point them at
   `docs/getting-started/installation.md`.
3. **For `analyze`**: no LLM needed — runs offline against Loki AST.
4. **For `gpu` final validation**: `nvfortran` (NVIDIA HPC SDK) is
   optional but recommended. Without it, syntax-only validation runs
   under `gfortran -fsyntax-only`.

## How to invoke

Run the subcommand via `Bash` and report what changed. Always pass
**absolute paths**.

### Static analysis (no LLM)

```bash
fortranspire analyze /abs/path/to/kernels/      # whole directory
fortranspire analyze /abs/path/to/kernel.f90    # single file
fortranspire analyze --sarif report.sarif /abs/path/to/kernel.f90
```

`--sarif` emits a SARIF 2.1.0 report — feed it into GitHub Code Scanning
or any SARIF-aware tool. Use `--no-toolchain-check` to skip the gfortran/
nvfortran probe.

### Pre-flight cost & risk estimate (no LLM, no tokens)

```bash
fortranspire explain /abs/path/to/kernel.f90
```

Outputs the routine count, control-flow complexity, GPU portability
score, and a token / wall-time estimate for `gpu` and `translate`.
**Read this output before recommending the user pay for an LLM-driven
port** — it's free and tells you whether the file is portable.

### Documentation generator

```bash
fortranspire doc --no-llm --dry-run /abs/path/to/kernel.f90   # see what would change
fortranspire doc --no-llm /abs/path/to/kernel.f90             # inline !> docstrings
fortranspire doc --with-llm --sphinx-out docs_site/ /abs/path/to/kernels/
```

`--no-llm` injects placeholders. `--with-llm` calls the configured LLM
to fill them with prose. `--sphinx-out` scaffolds a Sphinx site with
auto-generated docstrings + LLM-written narrative pages.

### Source formatting

```bash
fortranspire format /abs/path/to/kernel.f90              # in-place
fortranspire format --check /abs/path/to/kernel.f90      # CI mode (non-zero rc on diff)
```

### Call graph

```bash
fortranspire graph /abs/path/to/kernels/ --out call_graph.md
```

Outputs a Mermaid `flowchart LR` diagram. Renders inline in any Markdown
viewer (GitHub, mdBook, Sphinx with `sphinxcontrib-mermaid`).

### Phase 1 — Fortran → GPU (OpenACC + Cython)

```bash
fortranspire gpu /abs/path/to/kernel.f90 --out output/
# Variants:
fortranspire gpu --pragma omp /abs/path/to/kernel.f90      # OpenMP target instead of OpenACC
fortranspire gpu --equivalence /abs/path/to/kernel.f90     # add the equivalence test harness
```

Pipeline: parser (Loki) → PURE/ELEMENTAL → OpenACC kernels → Cython
wrapper → validation (`gfortran -fsyntax-only` or `nvfortran`). Final
artifacts in `output/`:
- `fortran_gpu/kernel_pure.f90` — PURE/ELEMENTAL annotated
- `fortran_gpu/kernel_gpu.f90` — OpenACC pragmas
- `cython/*.pyx` + `cython/kernel_c.h` — Python-callable wrapper

After this runs, summarize the validation log and the GPU port-cost
estimate (tokens used, walltime). If `validation_passed=False`, surface
the error and propose `fortranspire analyze` on the file to localize the
issue before retrying.

### Phase 1 — Batch port (parallel)

```bash
fortranspire port-batch /abs/path/to/kernels/ --jobs 4 --out output/
```

Parallel port across many files with per-file output isolation
(ContextVar-based). Use when the user has a directory of independent
kernels. Reports a success/fail count at the end.

### Phase 2 — Fortran → JAX (experimental)

```bash
fortranspire translate /abs/path/to/kernel.f90 --out output/jax/
```

Experimental. Lower success rate than `gpu` — recommend running
`explain` first; if its JAX-portability score is low, propose `gpu`
instead.

### MCP server

```bash
fortranspire mcp --port 8000
# or set env: MCP_HOST=0.0.0.0 MCP_PORT=8000
```

Starts the FastMCP HTTP/SSE server. Use this when the user is wiring
fortranspire into a different agent host (Claude Code MCP client,
mistral-vibe, a custom orchestrator). The server exposes the same tools
as the CLI plus structured outputs (Pydantic schemas).

## Quick decision flow

When the user mentions a Fortran file:

1. **Don't pay for tokens before knowing the file is portable** —
   start with `fortranspire explain <file>`.
2. If the user wants to gate a CI pipeline → `analyze --sarif`.
3. If the user wants to understand the code → `doc` (offline `--no-llm`
   first, then `--with-llm` if they want prose).
4. If the user wants to port → `gpu` for a single file, `port-batch`
   for many. Read the validation log; if it fails, drop back to
   `analyze` to localize.
5. If the user mentions training a surrogate / differentiability →
   `translate` (JAX).

## What this skill cannot do

- **Connect to a remote MCP server** — this skill shells out to the
  local CLI. For remote MCP, point the host's MCP client at the
  user's `fortranspire mcp` server (see
  `docs/integrations/claude-code.md` and
  `docs/integrations/mistral-vibe.md`).
- **Run on Windows natively** — Loki + Fortran toolchain expect
  POSIX. Suggest WSL2 if the user is on Windows.
- **Compile to native binaries** — fortranspire produces source. The
  user still needs `nvfortran` / `gfortran` to compile.

## Reference

- Project repo: https://github.com/maurinl26/fortranspire
- PyPI: https://pypi.org/project/fortranspire/
- Documentation: https://fortranspire.readthedocs.io
- Issue tracker: https://github.com/maurinl26/fortranspire/issues


## Interactive: domain geometry & decomposition

The `domain` verb — and the MCP tools `domain_geometries` /
`domain_decomposition` — are **interactive**, because the geometry cannot be
read from the Fortran (a kernel `t(k+1)-t(k-1)` has a halo of 1 whatever the
mesh; whether that mesh is octahedral or HEALPix is the modeller's choice).

When the user asks about domain decomposition, halos, or MPI layout for a
kernel, run this flow:

1. Read the kernel's **stencil halo** — pass the source to
   `domain_decomposition`; it reports the halo from the typed model.
2. **Ask the user** which geometry and resolution (present the catalogue
   from `domain_geometries` — O1280, nside=1024, C768, R2B9, …) and how many
   MPI ranks. Do not guess these.
3. Call `domain_decomposition(resolution, n_ranks, source)` for the proposal.

The tools nudge you: with no geometry they return the catalogue and ask;
with a geometry but no ranks they read the halo and ask for the ranks. The
grid imposes the decomposition — a requested rank count is snapped to what
the topology allows (a cubed sphere to 6·p², HEALPix nested to 12·4^d), and
the report says when it snapped.
