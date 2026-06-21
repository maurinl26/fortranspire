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
  - name: External Lecturer, École Nationale de la Météorologie, Toulouse, France
    index: 1
date: 21 June 2026
bibliography: paper.bib
---

# Summary

`fortranspire` is an open-source pipeline that incrementally transforms
legacy Fortran scientific code into GPU-accelerated, Python-callable, and
optionally differentiable form. It combines deterministic abstract-syntax-tree
(AST) analysis with targeted Large Language Model (LLM) calls, exposed
through a Model Context Protocol (MCP) [@mcp2024] server so that the pipeline
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

A typical run consumes four LLM calls (~2 minutes wall-clock, ~0.06 USD in
Mistral-Large tokens) and produces a kernel that compiles for GPU and is
importable from Python.

To our knowledge no other open-source tool combines deterministic Fortran
AST transformation, LLM-driven kernel refactoring, GPU pragma insertion,
Cython packaging, and optional JAX translation behind a single MCP
interface. Related efforts cover individual stages: Loki [@loki2024]
provides the AST and transformation framework, F2PY [@f2py2009] and Cython
[@behnel2011cython] handle wrapping, OpenACC [@openacc2024] provides the
pragma model, and JAX [@jax2018github] provides the differentiable
back-end. `fortranspire` glues these together with an LLM acting as the
"semantic glue" exactly where deterministic rules fall short.

The MCP server makes the pipeline addressable from any MCP-aware client
(Claude Desktop, Cursor, VS Code agents, Mistral Le Chat connectors), so
researchers can port a kernel without leaving their editor. The same code
runs unchanged in a CI job, an Apptainer container on an HPC login node, or
a sovereign-EU cloud VM, addressing the data-residency constraints common
in industrial R&D.

# Functionality

The package exposes both a CLI (`agent-gpu`, `agent-pipeline`,
`agent-translate`, `agent-profile`) and an MCP server (`run-mcp`) that
publishes the same operations as MCP tools (`translate_kernel_gpu`,
`translate_kernel`, `profile_kernels`, `ask_agent`). All operations are
file-in / file-out, write their intermediate results to `output/`, and emit
a structured log of LLM calls and validation steps for auditability.

Nine recurring Fortran patterns — `INTENT`, `SAVE`, `COMMON`, `POINTER`,
implicit typing, fixed-form continuation, derived types, `MODULE PROCEDURE`
interfaces, and module-private state — are documented in the README and
covered by fixture kernels in the test suite. These patterns cover the
overwhelming majority of legacy scientific Fortran encountered in seismic
and atmospheric production codes.

# Acknowledgements

We thank the ECMWF Loki team for the Fortran transformation toolkit that
underpins the deterministic stages of this pipeline, and the open-source
communities behind OpenACC, JAX, Cython, LangChain/LangGraph, and the
Model Context Protocol for the building blocks this work composes.

# References
