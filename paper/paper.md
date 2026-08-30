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

`fortranspire` is an open-source pipeline that transforms legacy
Fortran 90 scientific code into GPU-accelerated, Python-callable, and
optionally differentiable form. It combines deterministic
abstract-syntax-tree (AST) analysis with targeted Large Language Model
(LLM) calls, exposed through a Model Context Protocol (MCP) [@mcp2024]
server so the pipeline can be driven from an editor, a notebook, or a
continuous-integration job. A run produces `PURE`/`ELEMENTAL` kernels
with explicit `INTENT`, the same kernels annotated with OpenACC
[@openacc2024] pragmas, a Cython [@behnel2011cython] wrapper compiled
through `scikit-build`, and, on demand, a JAX [@jax2018github]
translation that is differentiable and JIT-compilable by XLA. The AST
stage uses Loki [@loki2024]; the LLM stage uses any OpenAI-compatible
endpoint — Mistral La Plateforme [@mistral2024] with Codestral
[@codestral2024] for code generation and Mistral Large 2
[@mistrallarge2024] for semantic reasoning by default, or a
self-hosted vLLM [@kwon2023efficient] / TGI / Ollama server.

# Statement of need

Scientific HPC codebases are dominated by Fortran 90 written between
the late 1980s and the early 2000s. Production codes for seismic
imaging, numerical weather prediction (ARPEGE, AROME, IFS)
[@ecmwf-ifs], computational fluid dynamics, and reservoir simulation
run for days on CPU clusters while modern GPU accelerators sit
underused because porting these codes by hand is slow (2–6 weeks per
kernel for a senior HPC engineer) and requires a rare combination of
skills — Fortran 90, OpenACC, MPI, Cython, and modern Python
packaging.

Two research directions are bottlenecked by this porting cost: (1)
generating training data for neural surrogates [@li2020fourier;
@raissi2019physics] and (2) physics-aware validation of those
surrogates during training. Both require the reference solver to be
fast (GPU) and callable from a Python machine-learning stack, which
legacy Fortran does not natively support.

`fortranspire` automates the transformations a human expert performs
by hand: kernel extraction from monolithic `PROGRAM`s into `MODULE`s
with explicit `INTENT`, OpenACC pragma insertion, Cython wrapping
with `iso_c_binding`, and validation through `gfortran` and
`nvfortran -acc`. A run consumes four LLM calls (approximately 2
minutes wall-clock and 0.04 USD on Codestral) and produces a kernel
that compiles for GPU and is importable from Python. Intermediate
artefacts are written to disk between stages so a reviewer can
inspect or hand-edit each step. The full pipeline architecture is
documented in [the project's Sphinx site](https://fortranspire.readthedocs.io/en/latest/concepts/architecture.html).

# Related work

`fortranspire` overlaps with three lines of prior work.

**Deterministic source-to-source compilers and stencil DSLs.**
PSyclone [@psyclone2024] programmatically rewrites Fortran for
OpenACC, OpenMP, and Kokkos and underpins the UK Met Office LFRic
dynamical core. Loki [@loki2024], on which `fortranspire` builds,
plays the equivalent role in the ECMWF IFS modernisation. GT4Py
[@paredes2023gt4py] provides a Python-embedded stencil DSL used by
Ubbiali et al. [@ubbiali2025cloudsc] to port the ECMWF CLOUDSC
microphysics scheme to GPU. These tools assume the input is already
modular: they do not lift `COMMON` blocks, refactor `SAVE` state, or
extract kernels from monolithic `PROGRAM`s.

**Monolithic neural transpilers.** CodeRosetta
[@tehranijamsaz2024coderosetta] is an encoder–decoder transformer
trained on parallel-code pairs (Fortran↔C++, C++↔CUDA). HPC-Coder
[@nichols2024hpccoder] fine-tunes a base LLM on HPC sources.
Fortran2CPP [@chen2024fortran2cpp] uses dual-agent multi-turn
dialogues. Godoy et al. [@godoy2024llmhpc] benchmark off-the-shelf
models across HPC kernels. These approaches operate on full files in
one shot, do not interleave deterministic AST analysis to bound LLM
responsibility, and do not expose a tool-call interface usable from a
developer's editor.

**Agentic LLM workflows.** Gupta et al. [@gupta2025kokkos] propose
the closest design: an agentic workflow translating legacy Fortran
to Kokkos C++. `fortranspire` differs in (i) target — OpenACC +
Cython + optional JAX rather than Kokkos, keeping the Python ML
ecosystem reachable; (ii) interleaving — six of the eight stages are
deterministic Loki AST work (see [architecture docs](https://fortranspire.readthedocs.io/en/latest/concepts/architecture.html)),
bounding LLM responsibility at three semantic edges and capping
spend at four LLM calls per kernel by construction; (iii) interface —
an MCP server, so any conformant client (mistral-vibe, Claude Code,
Cursor) drives the pipeline without bespoke integration.

# Functionality

The package exposes a unified `fortranspire <verb>` CLI (subcommands:
`analyze`, `explain`, `graph`, `doc`, `gpu`, `translate`, `profile`,
`mcp`) and a nine-tool MCP server with both stdio and HTTP/SSE
transports. Four of the nine tools are deterministic and require no
LLM call (`analyze_kernels`, `explain_port_cost`, `build_call_graph`,
`generate_docs`). Eleven recurring Fortran patterns — `INTENT`,
`COMMON`, `SAVE`, `POINTER`, AoS → SoA + `collapse`, stencil-vs-
recurrence dependencies, `ELEMENTAL`, explicit `KIND`, `LOGICAL
PARAMETER` flags, MPI halo exchange, and Fortran I/O — are
documented in the [Fortran patterns guide](https://fortranspire.readthedocs.io/en/latest/concepts/fortran-patterns.html).

# Real-world demonstration

The pipeline is exercised end-to-end against
[PHYEX](https://github.com/UMR-CNRM/PHYEX), the Météo-France
physics-parameterisations package. Calling
`explain_port_cost` on
`src/common/turb/mode_compute_function_thermo.F90` (119 lines, 1
routine) from inside `mistral-vibe` returns the structural assessment
in approximately two seconds with no LLM tokens consumed (Loki AST
only): 1 routine, 0 structural risks, an estimated 13 800 tokens for
a full Phase-1 port, and a cost of 0.04 USD on `codestral-latest`. A
subsequent `translate_kernel_gpu` call runs the OpenACC and Cython
stages on the same file.

# Quality control

Two test layers guard the pipeline. **Unit tests** (180+ pytest
functions) cover Loki adapters, the LangGraph state-machine
transitions, the MCP tool surface (a hand-rolled JSON-RPC handshake
verifying the nine tools are discoverable over stdio), the structured
CLI output formats (SARIF 2.1.0, Markdown, JSON), and the
deterministic stages independently of the LLM. **End-to-end
equivalence tests** compile the original Fortran and the generated
OpenACC variant with `gfortran -fopenacc`, execute both binaries on a
deterministic input, and assert `numpy.allclose` agreement within a
per-kernel tolerance (`atol=1e-12, rtol=1e-10`). The diagnostic on
failure reports the maximum absolute difference and the worst-case
probe index. One end-to-end fixture (`wave_kernels` — two 2-D
finite-difference stencils over 20 time steps) is currently shipped;
the harness accepts additional kernels by dropping a directory with
`original.f90`, `openacc.f90`, `driver.f90`, and a documented
tolerance into `tests/fixtures/equivalence/`.

# Limitations and roadmap

**Validation dependencies.** Some production codes (PHYEX, IFS)
depend on ECMWF-IFS-specific support modules (e.g., `YOMHOOK`) not
vendored with the upstream public repositories; the pipeline still
emits artefacts in those cases but the final `nvfortran -acc`
compilation gate is skipped. **JAX translation (Phase 2)** is
experimental and covers a subset of kernel shapes; complex time-loop
patterns require manual rewriting toward `jax.lax.scan` with explicit
carry. **Multi-node GPU communication** is on the roadmap — GHEX
[@ghez2023] integration for GPU-to-GPU halo exchange — but not yet
implemented.

Planned extensions include a GT4Py.next [@gt4pynext2024] code-
generation target for stencil DSLs, a managed deployment on the
European Weather Cloud via Morpheus / OpenStack, and neural-surrogate
training pipelines that consume the GPU-ported kernels as the
reference solver.

# Acknowledgements

We thank the ECMWF Loki team for the Fortran transformation toolkit
that underpins the deterministic stages of this pipeline, the
Université Mixte de Recherche CNRM (UMR 3589, Météo-France / CNRS)
for publishing the PHYEX physics package under CeCILL-C, and the
open-source communities behind OpenACC, JAX, Cython, LangChain /
LangGraph, and the Model Context Protocol.

# References
