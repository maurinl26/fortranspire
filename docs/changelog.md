# Changelog

All notable changes to `fortranspire` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Project renamed** from `coding-agent` / `local-code-agent` to
  `fortranspire`. Python package, PyPI distribution name, GitHub URLs,
  docs, and the inline `!> @generated_by` marker were all updated. Use
  `pip install fortranspire` and `from fortranspire.agent...` going
  forward.
- **Loki dependency** repointed from a hard-coded local path
  (`file://localhost/Users/loicmaurin/PycharmProjects/...`) to a pinned
  git tag (`git+https://github.com/ecmwf-ifs/loki@0.3.7`). The repo is
  now actually installable on a fresh machine.
- **Dependencies split into extras**. The default `uv sync` installs only
  what `agent-analyze` needs (~50 MB: Loki + NumPy + LangGraph). Pull in
  `[gpu]` (Phase 1 LLM stack + Cython + fprettify + fortls), `[mcp]`
  (FastMCP), `[jax]` (Phase 2), or `[all]` as needed.
- **LLM model selection per pipeline stage.** New env vars
  `MISTRAL_MODEL_REASONING` (default `mistral-large-latest`, used by
  `extractor` and `openacc` nodes) and `MISTRAL_MODEL_CODE` (default
  `codestral-latest`, used by `cython_wrapper`). The legacy
  `MISTRAL_MODEL` still works as a single fallback. Cuts cost per kernel
  ~50% on default settings.

### Added

- **`agent-analyze`** — standalone Fortran static analyzer (Loki-only,
  zero LLM calls). 11 rules covering COMMON blocks, SAVE, I/O in
  kernels, missing IMPLICIT NONE / KIND, POINTER, derived types,
  loop-carried deps, missing or non-OpenACC Fortran compiler on PATH.
  Outputs human-readable text, JSON, or SARIF 2.1.0. Includes a GitHub
  Actions workflow (`.github/workflows/analyze.yml`) that uploads SARIF
  to Code Scanning, and a lightweight Apptainer recipe
  (`Apptainer.analyze`) for HPC sites.
- **`agent-doc`** — LLM-driven documentation generator for legacy
  Fortran. Inserts idempotent `!>` Doxygen blocks above each routine
  (short `@brief` for stakeholders + multi-sentence `@details` for
  developers + `@param` per argument). With `--sphinx` / `--site-only`,
  also generates a self-contained Sphinx site (Furo + MyST +
  sphinx-design + sphinx-togglebutton) with a *Show source* dropdown
  per routine. `--no-llm` mode runs the signature extraction without
  any token cost.
- **`agent-format`** — Fortran source formatter wrapping `fprettify`
  with sensible defaults (lowercase keywords, 2-space indent). Works
  standalone and is intended to be called as a post-step by the Phase 1
  pipeline so generated OpenACC + Cython output isn't "flat" Fortran.
- **`fortranspire.agent.fortls_oracle`** — thin wrapper around the
  Fortran Language Server (`fortls`) for symbol lookup, neighbour-symbol
  context, and "does this name exist?" queries. Available for callers
  but not yet wired into the documenter prompts (deferred — see the
  follow-up issues).
- **Compiler detection** in `agent-analyze` — probes PATH for
  `nvfortran` / `gfortran` / `ifx` / `flang` / `lfortran`, classifies
  OpenACC capability (`native` / `experimental` / `unsupported`),
  surfaces the recommended compiler in every output format.
- **Mistral integration documentation** —
  `docs/concepts/mistral-integration.md` covers the four integration
  paths (LLM consumer, per-stage models, MCP provider, Le Chat connector
  directory). Connector manifest prepared in
  `integration/le-chat-connector.json` and documented in
  `docs/concepts/le-chat-connector.md`. Runnable smoke test at
  `examples/mistral_agents_api_smoke_test.py`.
- **JOSS submission package** — `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `paper.md`,
  `paper.bib`, `.zenodo.json` (Zenodo DOI metadata, ORCID
  `0009-0004-8117-4850`, affiliation ENM Toulouse), `.readthedocs.yaml`,
  and `.github/workflows/draft-paper.yml` to render the JOSS draft PDF
  on every push.
- **`CITATION.cff` + `codemeta.json`** at the repo root — GitHub picks
  up `CITATION.cff` for the "Cite this repository" button; `codemeta.json`
  is the semantic-web companion consumed by Zenodo, Software Heritage,
  and academic search engines. Both contain the same author/affiliation
  block as `.zenodo.json` and `paper.md`.
- **Sphinx documentation site** under `docs/` (Furo + MyST + extensions),
  with sections for installation, quickstart, configuration, pipeline
  architecture, Fortran patterns, LLM endpoints, Mistral integration,
  Le Chat connector, and the standalone documentation feature.

### Fixed

- **Parser missed module-contained routines.** `parser_phase1` only
  looked at `source.routines` (top-level), which is empty for modern
  Fortran 90 codes that wrap subroutines in `module ... contains ... end
  module`. Now walks `source.modules[].subroutines` too — Loki
  reports the same count, the LLM pipeline sees real routines instead of
  failing with "Loki found no routines in this file".
- **Off-by-one in `agent-doc` line numbers** due to `\s` matching `\n`
  in the routine-declaration regex; same bug caused the indent
  detection to capture a newline as whitespace. Switched to `[ \t]*`
  where appropriate, so generated docstring blocks now land above the
  right line and inherit the routine's indent correctly.
- **LangChain imports** in `translation_graph_phase1.py` were
  unconditionally loaded at module import, forcing `agent-analyze`
  (analyze-only) to depend on the full LLM stack. Moved into the three
  agent functions that actually call the LLM so `uv sync` (no extras)
  is enough to run analysis.

## [0.1.0] — 2026-06-17

### Added

- First public release. MCP server (`run-mcp`) exposing
  `translate_kernel_gpu`, `translate_kernel`, `profile_kernels`, and
  `ask_agent`.
- CLI: `agent-gpu`, `agent-translate`, `agent-profile`, `agent-pipeline`.
- Phase 1 LangGraph pipeline (Fortran → OpenACC GPU + Cython wrapper).
- Phase 2 LangGraph pipeline (Fortran → JAX, experimental).
- Loki-based deterministic Fortran AST analysis.
- Docker / docker-compose / Apptainer recipes; Azure Terraform deployment
  template.
- Pivoted to a sovereign Mistral endpoint (no Azure dependency on the LLM
  side).

[Unreleased]: https://github.com/maurinl26/fortranspire/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/maurinl26/fortranspire/releases/tag/v0.1.0
