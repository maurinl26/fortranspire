---
title: 'fortranspire: an LLM-driven Model Context Protocol pipeline for porting legacy Fortran HPC kernels to GPU (OpenACC) and differentiable JAX'
tags:
  - Python
  - Fortran
  - HPC
  - GPU
  - OpenACC
  - JAX
  - Cython
  - large language models
  - Model Context Protocol
  - scientific computing
authors:
  - name: Loïc Maurin
    orcid: 0009-0004-8117-4850
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent Researcher, Toulouse, France
    index: 1
date: 24 June 2026
bibliography: paper.bib
---

# Summary

`fortranspire` is an open-source pipeline that incrementally transforms
legacy Fortran scientific code into GPU-accelerated, Python-callable, and
optionally differentiable form. It combines deterministic abstract-syntax-tree
(AST) analysis with targeted Large Language Model (LLM) calls, exposed
through a Model Context Protocol (MCP) [@mcp2024] server so the pipeline
can be driven from an editor, a notebook, or a continuous-integration job.

A single invocation takes a Fortran 90 source file and produces (i) an
extracted module of `PURE`/`ELEMENTAL` kernels with explicit `INTENT`,
(ii) the same kernels annotated with OpenACC [@openacc2024] parallel-loop
and data-region pragmas, (iii) a Cython [@behnel2011cython] wrapper compiled
through `scikit-build`, and (iv) on demand, a JAX [@jax2018github]
translation of the same kernel that is differentiable and JIT-compilable by
XLA. The AST stage uses Loki [@loki2024], the ECMWF Fortran transformation
toolkit; the LLM stage uses any OpenAI-compatible endpoint, including the
sovereign European Mistral API [@mistral2024] or a self-hosted
vLLM [@kwon2023efficient] / TGI / Ollama server. The pipeline ships with
container images (Docker, Apptainer) for HPC sites and reproducible local
runs.

# Statement of need

Scientific high-performance computing (HPC) codebases are dominated by
Fortran 90 written between the late 1980s and the early 2000s. Production
codes for seismic imaging, numerical weather prediction (e.g., ARPEGE,
AROME, IFS) [@ecmwf-ifs], computational fluid dynamics, and reservoir
simulation routinely run for days on CPU clusters, while modern GPU
accelerators sit underused because porting these codes by hand is slow
(2–6 weeks per kernel for a senior HPC engineer) and requires a rare
combination of skills — Fortran 90, OpenACC, MPI, Cython, and modern
Python packaging.

This porting bottleneck has become the limiting factor for two adjacent
research directions that depend on running the reference physics solver
thousands of times: (1) generating training data for neural surrogates
(Fourier neural operators [@li2020fourier], physics-informed neural
networks [@raissi2019physics]) and (2) physics-aware validation of those
surrogates during training. Both require the reference solver to be both
fast (GPU) and callable from a Python machine-learning stack, which legacy
Fortran does not natively support.

`fortranspire` addresses this gap by automating the four transformations a
human expert would perform, while keeping the engineer in the loop:

1. **Parsing.** Loki extracts an AST and detects `COMMON` blocks, `SAVE`
   variables, implicit `INTENT`, and inline loops — fully deterministically.
2. **Kernel extraction.** A single LLM call converts a monolithic
   `PROGRAM` into a `MODULE` of subroutines, replaces `COMMON` blocks with
   explicit `INTENT(INOUT)` arguments, and produces a driver that preserves
   the original time-stepping loop.
3. **Purity annotation and OpenACC port.** Rule-based AST checks annotate
   `PURE`/`ELEMENTAL` where legal; one further LLM call adds
   `!$acc parallel loop collapse(...)` and `!$acc data copyin/copy`
   directives around the time loop.
4. **Cython packaging and validation.** Two LLM calls generate a `.pyx`
   wrapper with NumPy typed memoryviews and a C header using
   `iso_c_binding`; deterministic compilation with `gfortran` (twice, for
   two compiler flavors) and `nvfortran -acc` validates the output.

A typical run consumes four LLM calls (~2 minutes wall-clock, ~0.04 USD on
Mistral `codestral-latest` or ~0.06 USD on Mistral-Large) and produces a
kernel that compiles for GPU and is importable from Python.

`fortranspire`'s contribution is the combination of deterministic AST
transformation, LLM-driven semantic refactoring, GPU pragma insertion,
Cython packaging, and optional JAX translation behind a single MCP
interface; the related-work section below positions this against the
three threads of prior art (deterministic source-to-source compilers,
monolithic neural transpilers, and agentic LLM workflows).

The MCP server makes the pipeline addressable from any MCP-aware client
(Claude Code, Claude Desktop, Cursor, mistral-vibe, VS Code agents), so
researchers can port a kernel without leaving their editor. The same code
runs unchanged in a CI job, an Apptainer container on an HPC login node, or
a sovereign-EU cloud VM, addressing the data-residency constraints common
in industrial R&D.

# Related work

Three threads of prior work overlap with `fortranspire`.

**Deterministic source-to-source compilers.** PSyclone
[@psyclone2024] (UK STFC) programmatically rewrites Fortran for
OpenACC, OpenMP, and Kokkos via user-provided transformation scripts;
it underpins the UK Met Office LFRic dynamical core and inserts GPU
offload pragmas into NEMO. CLAW [@clausecker2018claw] (ETH Zürich) is
an earlier Fortran DSL with similar source-to-source ambitions in
weather and climate models. GPUFORT [@gpufort2023] (AMD) emits CUDA
Fortran and HIP from rule-based transformation scripts. Loki
[@loki2024], on which `fortranspire` builds, plays the equivalent
role inside the ECMWF IFS modernisation. These tools assume the
input code is already modular: they do not lift `COMMON` blocks,
refactor `SAVE` state, or extract kernels from monolithic
`PROGRAM`s — precisely the steps where LLM guidance contributes.

**Monolithic neural transpilers.** CodeRosetta
[@tehranijamsaz2024coderosetta] is an encoder-decoder transformer
trained on parallel-code pairs covering Fortran↔C++ and C++↔CUDA.
HPC-Coder [@nichols2024hpccoder] fine-tunes a base LLM on HPC
sources and tasks. Fortran2CPP [@chen2024fortran2cpp] uses
multi-turn dialogues and a dual-agent setup to translate Fortran to
C++. Godoy et al. [@godoy2024llmhpc] benchmark off-the-shelf GPT
models across HPC kernels in C++, Fortran, Python, and Julia,
finding correctness strongly correlates with the maturity of the
target programming model. All of these operate on full files in
one shot, do not interleave deterministic AST analysis to bound
LLM responsibility, and do not expose the pipeline through a
tool-call interface usable from a developer's editor.

**Agentic LLM workflows.** Gupta et al. [@gupta2025kokkos] propose
the closest design to `fortranspire`: an autonomous agentic
workflow translating legacy Fortran to performance-portable
Kokkos C++. `fortranspire` differs in three ways: (i) it targets
OpenACC + Cython rather than Kokkos, keeping the Python ML and
data-science ecosystems reachable through `iso_c_binding` and
NumPy typed memoryviews, and adding an optional JAX (Phase 2)
output for differentiable use cases; (ii) it interleaves Loki
deterministic stages with the LLM stages, capping LLM
responsibility at the semantic edges and bounding token spend at
four calls per kernel by construction; (iii) it exposes the
pipeline as an MCP server so any conformant client (mistral-vibe,
Claude Code, Cursor) can drive the port from a developer's
editor without writing client-specific integration code.

# Functionality

The package exposes a unified CLI (`fortranspire <verb>` with subcommands
`analyze`, `explain`, `graph`, `doc`, `gpu`, `translate`, `profile`, `mcp`
and others) and an MCP server that publishes nine tools:
`analyze_kernels`, `explain_port_cost`, `build_call_graph`, `generate_docs`
(all four deterministic, no LLM call), `translate_kernel_gpu`,
`translate_kernel`, `profile_kernels`, `ask_agent`, and `agent_status`.
The MCP server supports both stdio (when the IDE owns the lifecycle) and
HTTP/SSE (for permanent service deployments) transports. All operations
are file-in / file-out, write intermediate results to `output/`, and emit
a structured log of LLM calls and validation steps for auditability.

Eleven recurring Fortran patterns — `INTENT`, `COMMON` blocks, `SAVE`,
`POINTER`, AoS → SoA + `collapse(2)`, stencil-vs-recurrence dependencies,
`ELEMENTAL` + `!$acc routine seq`, explicit `KIND` types,
`LOGICAL PARAMETER` flags → `#ifdef`, MPI halo exchange → GHEX (planned),
and Fortran I/O → xarray/zarr + DLPack (planned) — are documented in the
project documentation and covered by fixture kernels in the test suite.
These patterns cover the overwhelming majority of legacy scientific
Fortran encountered in seismic and atmospheric production codes.

## Where the LLM intervenes (and where it does not)

Fortran 90/2003 is a structured domain: the grammar is closed and
fully parseable, OpenACC is a versioned standard with a finite
directive vocabulary, the `iso_c_binding` mapping between Fortran
types and C is mechanical, and validation reduces to compiling with
two reference toolchains. The bulk of the transformation work is
therefore amenable to deterministic AST and template rules. Concretely,
six of the eight pipeline stages are LLM-free:

| Stage | Tool | LLM call? | What it does |
| ----- | ---- | --------- | ------------- |
| 1. Parse | Loki [@loki2024] | No | AST extraction, `INTENT` / `SAVE` / `COMMON` detection, loop and I/O census |
| 2. Extract | LLM | **Yes** | Lift kernels from monolithic `PROGRAM` into a `MODULE`; eliminate `COMMON`; surface `SAVE` as `INTENT(INOUT)` |
| 3. Purity | AST rules | No | Annotate `PURE`/`ELEMENTAL` where legal (no I/O, no `SAVE`, explicit `INTENT`) |
| 4. OpenACC | LLM | **Yes** | Insert `!$acc parallel loop collapse(...)` and `!$acc data copyin/copy` around the time loop |
| 5. Cython | LLM (×2) | **Yes** | Generate `.pyx` with typed memoryviews and `iso_c_binding` header |
| 6. CPU validation | `gfortran` [@gfortran] | No | Compile original and OpenACC variants; assert syntactic correctness |
| 7. GPU validation | `nvfortran -acc` (or `flang` [@flang] on roadmap) | No | Compile the OpenACC variant for the target architecture |
| 8. Equivalence | Test harness | No | Run both binaries on a deterministic input, assert `numpy.allclose` |

The LLM is the *minority partner*: it intervenes only at the three
semantic edges where deterministic rules cannot infer programmer
intent. Stage 2 must decide what constitutes a kernel inside a
five-thousand-line `PROGRAM` and how to expose its hidden state;
stage 4 must place the `!$acc data` region at the temporal granularity
that minimises host-device traffic without breaking the time-step
dependency; stage 5 must map Fortran `OPTIONAL` arguments and array
descriptors to a Cython interface that preserves the column-major
NumPy view. These are the parts where a structured rule-based system
would either fail (no closed-form rule covers the diversity of
production codes) or require so many special cases that the rule base
itself becomes a maintenance liability.

## Why an agent, not a one-shot prompt

A structured domain still requires an agent — that is, a loop with
state, validation, and retry — because the three LLM stages above are
not statistically independent. A wrong kernel boundary at stage 2
propagates into wrong `!$acc data` clauses at stage 4 and an
incompatible Cython signature at stage 5; a wrong OpenACC layout at
stage 4 causes a `nvfortran -acc` failure at stage 7 that the LLM has
no way to diagnose unless the compiler log is fed back into its
context. `fortranspire` therefore orchestrates the eight stages with
LangGraph [@langgraph2024]: each stage reads from and writes to a typed
state dictionary, and the validation stages 6–8 can route the pipeline
back to stage 4 (or 2) with the compiler log appended to the LLM
context, capped at three retries to bound token spend. Intermediate
artefacts are written to `output/` between stages so a human reviewer
can inspect or hand-edit them and re-run from any checkpoint without
re-issuing the upstream LLM calls. Together, the eight-stage typed
state machine, the validation-driven retry loop, the deterministic
majority, and the MCP exposure constitute the agent design that the
title of this paper refers to.

# Real-world demonstration

The pipeline is exercised end-to-end against
[PHYEX](https://github.com/UMR-CNRM/PHYEX), the public Météo-France
physics-parameterisations package shared with Meso-NH, the AROME-France
limited-area model, and other ACCORD-consortium NWP systems. On a Mac
mini (Apple Silicon, 16 GB RAM), invoking
`fortranspire_explain_port_cost` against
`src/common/turb/mode_compute_function_thermo.F90` (119 lines, 1
routine) from inside `mistral-vibe` returns the structural assessment
in approximately two seconds, with no LLM tokens consumed by
`fortranspire` itself (Loki AST only): 1 routine identified,
0 structural risks, an estimated 13 800 input + output tokens for a
full Phase-1 port, and an estimated cost of 0.04 USD on
`codestral-latest`. The same file is then portable to OpenACC and
Cython in a subsequent `translate_kernel_gpu` invocation.

# Quality control

Two test layers guard the pipeline. **Unit tests** (180+ pytest
functions) cover Loki adapters, the LangGraph state-machine
transitions, the MCP tool surface (a hand-rolled JSON-RPC handshake
verifying the nine tools are discoverable over stdio), the structured
CLI output formats (SARIF 2.1.0, Markdown, JSON), and the
deterministic stages independently of the LLM (using fixture inputs
and recorded outputs). **End-to-end equivalence tests** compile both
the original Fortran and the generated OpenACC variant with `gfortran`
(the latter via `gfortran -fopenacc`), execute both binaries on a
deterministic input, and assert `numpy.allclose` agreement within a
per-kernel tolerance (typically `atol=1e-12, rtol=1e-10`). The
diagnostic on failure reports the maximum absolute difference, the
worst-case probe index, and both values, enabling regression
attribution to a specific LLM stage. The first kernel covered is
`wave_kernels` — two two-dimensional finite-difference stencils run
for 20 time steps — and the harness is designed to accept additional
kernels by dropping a directory containing `original.f90`,
`openacc.f90`, `driver.f90`, and a documented tolerance into
`tests/fixtures/equivalence/`.

# Limitations and roadmap

Three known limitations bound the present scope. **Validation
dependencies.** Some production codes (PHYEX, IFS) depend on
ECMWF-IFS-specific support modules (e.g., `YOMHOOK` for instrumentation)
that are not vendored with the upstream public repositories; the
pipeline still emits the transformed Fortran and Cython artefacts in
those cases, but the final compilation gate against `nvfortran -acc` is
skipped. **JAX translation (Phase 2) is experimental** and currently
covers a subset of kernel shapes; complex time-loop patterns require
manual rewriting toward `jax.lax.scan` with explicit carry. **Multi-node
GPU communication** is on the roadmap — GHEX [@ghez2023] integration for
GPU-to-GPU halo exchange replaces CPU-roundtrip MPI but is not yet
implemented.

Planned extensions include a GT4Py.next [@gt4pynext2024] code-generation
target for stencil DSLs (relevant for ICON-LAM and finite-volume
unstructured grids), a managed deployment on the European Weather Cloud
via Morpheus / OpenStack to support data-residency-constrained users, and
neural-surrogate training pipelines (Fourier neural operators
[@li2020fourier], physics-informed neural networks [@raissi2019physics])
that consume the GPU-ported kernels as the reference solver.

# Acknowledgements

We thank the ECMWF Loki team for the Fortran transformation toolkit that
underpins the deterministic stages of this pipeline, the
Université Mixte de Recherche CNRM (UMR 3589, Météo-France / CNRS) for
publishing the PHYEX physics package under CeCILL-C, and the open-source
communities behind OpenACC, JAX, Cython, LangChain/LangGraph, and the
Model Context Protocol for the building blocks this work composes.

# References
