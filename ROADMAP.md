# fortranspire — Roadmap

_Last updated: 2026-06-22 (v0.1.3 published)_

This roadmap captures the **strategic direction**, the **near-term concrete
work** tracked in GitHub issues, and the **deployment surfaces** the project
targets. The technical phases are versioned independently in
[`docs/changelog.md`](docs/changelog.md).

## Where we are

- **`fortranspire 0.1.3`** is live on [PyPI](https://pypi.org/project/fortranspire/) (single-step install).
- **`loki-ifs 0.3.7`** — community PyPI redistribution of ECMWF Loki — is live on PyPI ([source](https://github.com/maurinl26/loki-ifs)).
- **13 CLI verbs**, **9 MCP tools** (with [PR #40](https://github.com/maurinl26/fortranspire/pull/40) in flight), **234+ unit tests**.
- **JOSS submission** in preparation — `paper.md`, `paper.bib`, `CITATION.cff`, `codemeta.json`, `.zenodo.json` all in place.
- **Sphinx documentation** published on [Read the Docs](https://fortranspire.readthedocs.io/).

## Strategic posture

fortranspire is an **open-source R&D asset** — a public reference
implementation of LLM-driven Fortran modernization. The project is
maintained under Apache 2.0 with no paid feature gating.

The architecture explicitly separates three concerns:

1. **Deterministic backbone** — Loki AST + the harness + Pydantic
   structured outputs absorb ~60–70 % of the transformation work. This
   layer carries zero LLM risk.
2. **LLM layer** — sovereign EU endpoint (Mistral La Plateforme by
   default, self-hosted vLLM / TGI / Ollama on an OpenStack tenant
   or on-prem cluster as alternatives). The model choice is a runtime
   config, not a code change.
3. **Deployment surface** — local Docker, on-prem Apptainer, EU sovereign
   cloud on an OpenStack tenant or the European Weather Cloud. Always
   user-owned compute; no hyperscaler dependency.

This separation is what allows the same agent to be invoked from
[Claude Code](docs/integrations/claude-code.md) (US LLM) or
[mistral-vibe](docs/integrations/mistral-vibe.md) (EU LLM) without code
change. Same agent, fungible LLM, choice of sovereignty posture.

## Near-term tracked work

Each item below corresponds to a public GitHub issue or PR. **Order is
priority, not strict dependency.**

| # | Title | Status | Lands in |
| - | ----- | ------ | -------- |
| [#40](https://github.com/maurinl26/fortranspire/pull/40) | Claude Code skill + extended MCP surface for mistral-vibe | PR in review | 0.1.4 |
| [#41](https://github.com/maurinl26/fortranspire/pull/41) | Remove Azure-flavored deployment artifacts | PR in review | 0.1.4 |
| [#42](https://github.com/maurinl26/fortranspire/issues/42) | GT4Py.next as a Phase-1 transformation target | Open, scoped | 0.2.0 |
| [#43](https://github.com/maurinl26/fortranspire/issues/43) | Deploy MCP on European Weather Cloud (Morpheus) | Open, scoped | 0.2.0 |
| TBD | Loki upstream contribution path (pydantic pin, redistribution NOTICE) | Outreach in progress | n/a |
| TBD | Public Claude Code plugin packaging + `awesome-claude-code` listing | Planned | 0.2.0 |

## Technical phases

The phase numbering tracks **transformation depth**, not chronology — Phase 1.5 was a 0.1.x in-flight refinement, Phase 2 is partial today.

| Phase | What it does | Status | Lands |
| ----- | ------------ | ------ | ----- |
| 1 — OpenACC + Cython | Fortran → annotated GPU kernels + Python-callable wrapper | **Shipped** | 0.1.0 |
| 1.5 — OpenMP target alternative | Multi-vendor GPU pragma (gfortran 13+, nvfortran, ifx) | **Shipped** | 0.1.0 (issue #18) |
| 1.6 — GT4Py.next | Stencil-friendly Python DSL output (cartesian + unstructured) | Scoped — [#42](https://github.com/maurinl26/fortranspire/issues/42) | 0.2.0 |
| 2 — JAX | Functional, differentiable Python kernels | Partial | 0.2.x |
| 3 — GHEX (GPU-to-GPU MPI) | Cross-node halo exchange with RDMA + CUDA-aware MPI | Future | 0.3.x |
| 4 — Modern I/O (xarray / zarr) | Replace `OPEN/WRITE/CLOSE` with cloud-native I/O | Future | 0.3.x |
| 5 — Surrogate models (FNO / DeepONet) | Train neural operators from the ported kernels | Future | 0.4.x |

## Deployment surfaces

The MCP server (`fortranspire mcp`, port 8000, SSE) is the canonical
entry point for IDEs and agents. Today it deploys to:

| Surface | Status | Doc |
| ------- | ------ | --- |
| Local Docker | Ready | [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml) |
| Apptainer (HPC sites) | Ready | [`apptainer.def`](apptainer.def), [`Apptainer.analyze`](Apptainer.analyze) |
| Self-hosted vLLM / TGI / Ollama (LLM-side) | Ready | [`docs/concepts/llm-endpoints.md`](docs/concepts/llm-endpoints.md) |
| OpenStack-managed GPU tenant (per-kernel H100 spin-up) | Planned | TBD |
| European Weather Cloud (Morpheus) | Planned — [#43](https://github.com/maurinl26/fortranspire/issues/43) | [`docs/integrations/european-weather-cloud.md`](docs/integrations/european-weather-cloud.md) (TBD) |
| OVHcloud GPU instances | Planned | TBD |

## JOSS / academic milestones

- [x] `paper.md` drafted with [@joic-maurin](https://orcid.org/0009-0004-8117-4850) authorship
- [x] `paper.bib` populated with Loki, OpenACC, GT4Py, JAX, Mistral references
- [x] `.zenodo.json` + `CITATION.cff` ready for DOI minting
- [x] Read the Docs hosting active
- [ ] First Zenodo DOI minted (webhook fires on next tag)
- [ ] JOSS submission opened
- [ ] First independent reviewer feedback addressed
- [ ] Paper accepted, DOI cited in README badge

## How to contribute

- **Pick an open issue** — labelled `enhancement` / `marginal` / `good first issue`.
- **Try the Claude Code skill** (`skills/claude-code/fortranspire/`) and report friction.
- **Run `fortranspire analyze`** on your Fortran codebase and file the surprises as issues.
- **Submit a Mermaid call-graph** of your kernel — helps shape the Phase-1.6 GT4Py.next rules.
- **Run the MCP server** behind your editor of choice and tell us what's missing in the tool surface.

Direct contact: [maurin.loic.ac@gmail.com](mailto:maurin.loic.ac@gmail.com)
or GitHub issues. The repo is the source of truth — anything not tracked
here is best raised as an issue first.
