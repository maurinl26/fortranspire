# Changelog

All notable changes to `fortranspire` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — hosted Le Chat connector (issue #51, split into #76/#77)

- **Inline-source MCP tools** — `analyze_source`, `explain_source`,
  `build_call_graph_source` (`fortranspire/server.py`). The path-taking
  tools assume the client and server share a filesystem, which a hosted
  SSE endpoint does not: a Le Chat user's Fortran is on their machine.
  These take the source text, materialise it in a jailed temp file under
  the workspace, run the **same** deterministic code path, and delete it
  before returning. `filename` only selects the dialect (`.f` fixed form,
  `.F90` triggers cpp); a directory component in it is ignored, and an
  oversized paste (>1 MB) is refused. Zero LLM, zero token, zero persisted
  write.
- **The Le Chat connector now exposes only the safe surface.**
  `integration/le-chat-connector.json` previously listed
  `translate_kernel_gpu`, `translate_kernel`, `profile_kernels` and
  `ask_agent` — every one of which spends the operator's Mistral key or
  writes files, which is unacceptable on a public directory endpoint.
  Replaced by the three deterministic inline-source tools, with a
  `public_surface_note` recording why the token-spending ones are absent.
- **TLS deployment** at `deploy/le-chat/` — a Caddy reverse proxy
  (automatic Let's Encrypt) fronting the MCP container, plus compose and
  `.env.example`. A small EU VPS is enough; this does **not** depend on
  the European Weather Cloud tenancy (#43). Documented in
  `docs/integrations/le-chat.md`.
- Tests: `tests/server/test_inline_source.py` (inline tools, fixed-form
  and cpp handling, hosted-safety guards, and connector-surface guards
  asserting no token-spending or file-writing tool is publicly exposed).


### Fixed — found on the first real target (CMAQ, USEPA)

Three defects, all surfaced by running the free deterministic path over
`CCTM/src/gas/ros3/rbsolver.F` — 632 lines of 1990s fixed-form F77.

- **Uppercase-suffix files were never preprocessed.** By Fortran
  convention `.F` / `.F90` / `.FOR` mean "run cpp first". We did not, so
  Loki read the `#ifdef` lines as Fortran and returned **zero routines**,
  which surfaced as `FORT009` "failed to parse" rather than the missing
  pass it was. 525 of CMAQ's Fortran files carry that suffix and 199 hold
  live conditionals, so the gap blocked the whole target. New
  `agent/nodes/_preprocess.py`, wired into the parser.
  Removed lines are **blanked rather than deleted**: findings carry a line
  number, the composite action turns it into a GitHub annotation, and
  `cpp -P` would shift every line past the first `#ifdef`.
- **Directory scanning ignored fixed-form suffixes.** Six call sites each
  spelled their own `*.[fF]90` glob, so `.F`, `.f`, `.for` were invisible:
  `fortranspire explain` on CMAQ's tree reported "no .f90 / .F90 file
  found" for 525 files. Replaced by one `collect_fortran_files` helper in
  `nodes/_common.py` covering 16 suffixes.
- **Reported line numbers were too low.** `_line_of` takes patterns like
  `^\s*SUBROUTINE`, and `\s` matches newlines, so with `re.MULTILINE` the
  quantifier started the match on a preceding blank line. `RBSOLVER` was
  reported at line 25, which is blank; it is at 26. Fixed centrally rather
  than in each pattern, so the next caller cannot reintroduce it.

Not a defect on our side, recorded for the target: 32 CMAQ `.F` files use
`include SUBST_CONST`-style aliases that CMAQ's own `bldit_cctm.csh`
resolves at build time. They need that build step before any tool can
read them.


### Added — Phase 2 rebuilt as functional refactoring → JAX (issue #73)

- **New `functionalize` node** (`agent/nodes_jax/functionalize.py`), the
  step Phase 2 never had. Deterministic, no LLM. It derives the functional
  signature from the INTENT map — a subroutine mutates its arguments, a
  JAX function cannot, so every `INTENT(OUT)`/`INTENT(INOUT)` argument
  becomes a return value — and issues a purity verdict (`pure` /
  `threaded` / `blocked`) from the `has_io` / `has_save` flags Loki
  already computes. `PURE`/`ELEMENTAL` is Fortran's own word for
  functional purity, so the gate was free and Phase 2 was not using it.
  A routine that cannot be pure is reported, never silently ported.
- **New `gradcheck` node**, blocking. The previous Phase 2 validation ran
  syntax → bytecode → `exec` → `make_jaxpr`, which together prove the code
  *traces* — not that its gradients are right, though differentiability is
  the entire reason to target JAX. It now compares `jax.grad` against
  central finite differences, in float64, probing a few random entries per
  array. Catches the NaN-gradient from an unguarded `jnp.where` (the
  classic translated-Fortran defect, where an `IF` guarded a `sqrt`),
  silently detached gradients, and kernels that raise under `grad`.
  It also checks `jit`, deliberately: outside `jit`, reverse-mode traces
  with a tracer carrying a concrete primal, so a Python `if` on a traced
  value resolves against it and the gradient looks correct — then the
  kernel raises the first time anyone jits it.
  **Documented limitation**, pinned by a test: finite differences see the
  same function autodiff does, so a locally flat transformation (`floor`,
  `round`, an integer cast) yields zero on both sides and passes. A pass
  is not a proof of differentiability in general.
- **New graph** `translation_graph_phase2.py`:
  `init → parser → extractor → functionalize → jax_kernel → gradcheck`.
  The first three nodes are the Phase 1 ones, reused rather than forked —
  parsing is the same work, and the extractor's promotion of `COMMON` /
  `SAVE` state to explicit arguments is the first half of
  functionalisation whatever the target. Two LLM calls maximum.
- **Externalised JAX prompts** at `prompts/jax_kernel/{en,fr}/v1.md`.
  Phase 2's prompts were inline in a 1 601-line module, so they were
  neither versioned nor translatable, unlike every other prompt family
  (issue #3).
- **`FORT030` — JAX portability verdict** surfaced by `analyze` and
  `explain`, reusing the `functionalize` rules rather than restating them,
  so a quote cannot promise a port the pipeline then refuses. Emitted only
  for routines that are *not* pure: annotating the clean ones would bury
  real findings in Code Scanning.

### Added — differentiability by relaxation (issue #73, second lot)

- **`fortranspire/jax_smooth.py`** — guards and smooth relaxations the
  emitted kernels import instead of re-deriving. Three reasons it is a
  library: the stable form is easy to get wrong (the textbook softmax
  `log(exp(b*a) + exp(b*b))` overflows past `exp(709)` while `logaddexp`
  does not, and a model writes the textbook one); the limit is worth
  testing once rather than per generation; and a named import shows a
  reviewer *where* the model was relaxed and with which parameter.
- **The distinction the module is built around.** A **guard**
  (`safe_sqrt`, `safe_divide`, `safe_log`) repairs a translation: forward
  values unchanged wherever the original was defined, only the NaN or the
  infinite derivative removed. A **relaxation** (`smooth_max`,
  `smooth_abs`, `smooth_step`, `smooth_clamp`, `smooth_argmax`,
  `interp_table`) changes what the code computes — `MAX` replaced by a
  softmax no longer returns `max`. That is right for an adjoint and wrong
  for a flux limiter that must stay TVD, so it is a modelling decision,
  never a default. Each relaxation carries its convergence limit and its
  bias in the docstring; both are pinned by tests rather than asserted.
- **`--smoothing none|guarded|smooth`** on `fortranspire translate`,
  defaulting to `none`. A translation that silently changes the model is
  worse than one that is honestly non-differentiable. Under `smooth`,
  every applied relaxation is named in the emitted code.
- **`FORT031` — non-smooth construct**, detected by `analyze` and labelled
  by `explain`, from the same catalogue the emission prompt embeds, so a
  construct cannot be detected without a documented replacement or offered
  to the model without being detectable. Comment tails are stripped first,
  and each construct is reported once per file rather than once per use.
  Two entries have no replacement on purpose: `FLOOR`/`NINT` are the class
  finite differences provably cannot see — analytic and numerical both
  read zero — so the static rule is the only way to catch them; and
  `DO WHILE` becomes `lax.while_loop`, which is not reverse-mode
  differentiable in JAX.

### Changed

- **`fortranspire translate` now routes through the Phase 2 graph** and
  **returns a non-zero exit code when the gradient check fails**. A kernel
  that traces with a wrong gradient is silently wrong exactly where the
  caller relies on it, so the check gates the command.
- `agent/translation_graph.py` is marked superseded for the JAX path. It
  stays importable: it still carries the Phase 3-5 experiments (halo
  exchange, reproducibility, surrogate models) that have no other home.

### Fixed

- **`FORT010` and `FORT011` rendered unlabelled in the port-cost report.**
  `explain` keeps its risk-label map in sync with `analyze` by hand, and
  the two toolchain rules were never added. Found by a new test asserting
  the two stay in sync — the report is what a client reads before paying.


### Added — agent surfaces on GitHub

- **GitHub App** (`fortranspire github-app`, issue #50). A Starlette
  webhook receiver that turns `/fortranspire <verb> <path>` comments on
  issues and pull requests into pipeline runs, replies with the report,
  and opens a pull request for the verbs that rewrite code. New package
  `fortranspire/github_app/`, new `[github-app]` extra, new CLI verb.
  Every verb shells out to the same `fortranspire` console script the CLI
  and MCP server use — the App adds a trigger and an authorisation layer,
  not a second implementation. Two independent gates, both failing
  closed: an operator-configured installation allow-list (absent means
  refuse everything) and a repository write-access check on the commenter
  (an unresolvable lookup counts as no permission). Token-spending verbs
  are opted in per installation and default to off. Paths are jailed
  against the checkout with symlinks resolved first, and LLM keys are
  stripped from the subprocess environment for the verbs advertised as
  free. 85 tests in `tests/github_app/`.
- **Claude Code GitHub Action workflow** (`.github/workflows/claude-port.yml`,
  issue #48). `@fortranspire` on a pull request runs Claude Code with the
  fortranspire MCP server wired in over stdio. Notably, it sets
  `FORTRANSPIRE_DISABLE_JAIL=0` and `AGENT_WORKSPACE`: stdio mode disables
  the workspace jail by default, which is right for a local IDE and wrong
  when the agent acts on a third party's comment. The Mistral key reaches
  the MCP server through the step environment rather than `claude_args`,
  so it never lands in the process list. The prompt makes the free
  `explain_port_cost` call mandatory before any token-spending port.
- **GitHub Copilot cloud agent integration** (issue #49):
  `.github/workflows/copilot-setup-steps.yml` (the job name is
  load-bearing — GitHub matches on it literally) and
  `.github/agents/fortran-porter.agent.md`, the only mechanism that keeps
  the MCP configuration in Git. The allow-list is restricted to the three
  deterministic tools: Copilot invokes MCP tools autonomously with no
  approval prompt, so nothing that spends a token or writes a file is
  exposed there.
- **`tests/test_agent_surfaces.py`** — pins what only executes on someone
  else's infrastructure: the embedded MCP JSON parses, the jail variables
  are present, no secret is inlined into `claude_args`, the Copilot job
  keeps the name GitHub matches on, and the Copilot allow-list contains
  only tools the server actually registers.
- Documentation: `docs/integrations/github-app.md`,
  `claude-code-action.md`, `copilot-coding-agent.md`, plus a README
  section comparing the four GitHub surfaces by cost and trust.

### Security

- **The MCP server no longer fails open when auth cannot be
  installed.** `_install_auth()` attached the bearer/rate-limit/audit
  middleware through `mcp._app`, a private attribute FastMCP 3.x
  removed. It logged `[Erreur] Impossible de sécuriser l'API` and then
  started the server **anyway** — so `API_KEY=… fortranspire mcp`
  served every tool, the LLM ones included, to unauthenticated
  clients. The middleware is now passed to `mcp.run(middleware=…)`,
  and a configured-but-unattachable stack raises
  `AuthNotInstallable`, which aborts start-up with exit code 1.
  Behaviour with no token configured is unchanged (local OSS use stays
  unauthenticated). Verified over real HTTP in
  `tests/server/test_mcp_http.py`.

### Added

- **`/health` endpoint on the MCP server.** It was declared in
  `integration/le-chat-connector.json` (`health_check: /health`) and
  allow-listed in the auth middleware, but never registered as a
  route, so it answered 404 — blocking the hosted-deployment
  acceptance criteria of #43 (European Weather Cloud) and #51 (Le Chat
  connector directory). Answers without a token and reports service,
  version, transport and tool count.
- **Reusable composite GitHub Action** at the repository root
  (`action.yml`, issue #47). Any Fortran repository gets GPU-portability
  findings in Code Scanning, a port-cost estimate in the job summary or
  as a PR comment, and a Mermaid call-graph artifact from one `uses:`
  line — deterministic Loki path only, so no LLM, no secret and safe on
  forks. The severity gate runs last so every report is published even
  when the build fails. Documented in the README; contract pinned by
  `tests/test_github_action.py`.
- **`.github/workflows/action-selftest.yml`** — exercises the action the
  way an external repository would, including the minimum-permission
  case and a run where the gate is expected to fire.

### Changed

- **Unified the MCP path argument on `path`.** `translate_kernel_gpu`,
  `translate_kernel`, `profile_kernels` and `explain_port_cost` took
  `filepath` while `analyze_kernels`, `build_call_graph` and
  `generate_docs` took `path` — the same concept under two spellings,
  and `filepath` accepted directories anyway. An agent reading the
  schema could not tell which tool wanted which, and burned a turn on
  the validation error. `path` is now the single required argument on
  every path-taking tool; `integration/le-chat-connector.json` is
  updated to match.
- **`.github/workflows/analyze.yml` now consumes the action** it
  publishes (`uses: ./` with `version: local`), so CI exercises the
  working tree rather than the last published wheel.

### Fixed

- **README, JOSS paper and the mistral-vibe integration pages
  advertised MCP tools that do not exist.** The prose referred to
  `fortranspire_explain_port_cost`, `fortranspire_translate_kernel_gpu`
  and `fortranspire_build_call_graph`; the server registers those names
  without the `fortranspire_` prefix, so the README quick-start returned
  `Unknown tool`. The tables further down the same pages were already
  correct. `tests/server/test_mcp_http.py` now fails on any
  reintroduction, and cross-checks the connector manifest against the
  live schema.


## [0.2.1] — 2026-06-24

Patch release: pre-JOSS-submission hardening. No API breaks; the only
default-behaviour change is the security pass below.

### Security

- **`run_shell` MCP exposure is now opt-in** via the
  `FORTRANSPIRE_ENABLE_SHELL=1` environment variable
  (`fortranspire/tools/__init__.py`). The ReAct agent previously
  shipped `run_shell(shell=True)` in `ALL_TOOLS`, reachable from the
  `ask_agent` MCP tool — a path that would have exposed
  unauthenticated remote shell execution on any HTTP-deployed MCP
  server. Default-off closes the blast radius.
- **MCP file/directory arguments are now jailed to the workspace
  root** through a new `_jail()` helper in `fortranspire/server.py`,
  applied across the seven file-accepting tools
  (`translate_kernel_gpu`, `translate_kernel`, `profile_kernels`,
  `analyze_kernels`, `explain_port_cost`, `build_call_graph`,
  `generate_docs`). Resolves the directory listed in
  `config.workspace_dir` plus the colon-separated extra roots in
  `FORTRANSPIRE_WORKSPACE_EXTRA_ROOTS`. Auto-disabled in stdio mode
  (the IDE owns the trust boundary); enforced by default in
  HTTP/SSE mode. Override with `FORTRANSPIRE_DISABLE_JAIL=1`.

### Added

- **`.github/workflows/tests.yml`** — pytest runs on every push and
  PR. Fast job (no gfortran) on a Python 3.11 / 3.12 matrix; slow
  job (`gfortran -fopenacc` equivalence harness) on push and manual
  dispatch. README now shows a Tests badge alongside the JOSS draft
  one.
- **JOSS draft paper musclé** then trimmed for concision: Related
  Work section across three threads of prior art (deterministic
  source-to-source compilers and stencil DSLs / monolithic neural
  transpilers / agentic LLM workflows), Quality control section
  describing the equivalence harness, Limitations and roadmap
  section. Currently 1103 body words. The full pipeline architecture
  (8-stage table, LLM intervention rationale, agent-loop motivation)
  lives in `docs/concepts/architecture.md`.

### Changed

- **Repo reorganisation.** Paper artefacts moved from the root to
  `paper/` (`paper/paper.md`, `paper/paper.bib`). Container artefacts
  moved from the root to `containers/` (`containers/Dockerfile`,
  `containers/Dockerfile.hpc`, `containers/apptainer.def`,
  `containers/docker-compose.yml`). Two stray root-level markdown
  files moved into the Sphinx tree: `OPTIMIZATION_REVIEW.md` →
  `docs/concepts/jax-optimization.md` and `TUTORIAL_IDE.md` →
  `docs/integrations/ide-interactive-mode.md`. The repo root now
  holds only the five `.md` files (`README`, `ROADMAP`,
  `CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY`) expected by GitHub /
  OSSF conventions.
- **`uv tool install` install instructions corrected to
  `uv tool install 'fortranspire[mcp]'`** in README and
  `docs/getting-started/with-mistral-vibe.md`. The `[mcp]` extra is
  required for the stdio handshake; the previous instructions would
  have produced a `fastmcp`-missing crash on the first vibe message.
- **README walkthrough link** now points at the full PHYEX walkthrough
  (`docs/integrations/mistral-vibe.md`) rather than the 60-second
  quickstart, matching the link text.
- **Hardcoded `/opt/homebrew` brew path removed** from the vibe
  walkthrough — `vibe` is on `PATH` after `brew install`, which is
  `/opt/homebrew` on Apple Silicon and `/usr/local` on Intel Macs.
- **`agent-*` / `run-mcp` legacy CLI names swept out** of
  `docs/getting-started/quickstart.md`,
  `docs/getting-started/installation.md`,
  `docs/concepts/legacy-documentation.md`, and
  `docs/concepts/mistral-integration.md` in favour of the unified
  `fortranspire <verb>` form.
- **EU-regulation backing for "sovereign"** added to
  `docs/concepts/llm-endpoints.md`: GDPR (Regulation (EU) 2016/679),
  AI Act (Regulation (EU) 2024/1689), NIS2 Directive (Directive (EU)
  2022/2555) cited with permanent EUR-Lex URLs. The term is now
  used in the regulatory sense throughout the documentation.
- **Paper tone pass.** Dropped marketing-flavoured phrasing
  ("sovereign European", "leadership-class GPU-equipped
  supercomputers", "minority partner", "keeping the engineer in the
  loop", "in practice"); fixed the 6-vs-8-stage contradiction with
  the README diagram; reworded the orphan self-reference at the end
  of the agent-loop section; corrected the over-claim that all 11
  patterns are covered by fixtures (only `wave_kernels` ships
  today).
- **Pytest default gate.** `pyproject.toml`
  `[tool.pytest.ini_options]` now sets `testpaths = ["tests"]` (so
  the Loki redistribution under `vendor/` is no longer collected)
  and `addopts = "-m 'not slow'"` (so the slow gfortran-driven
  equivalence harness is opt-in by default).

### Removed

- **`fortranspire/main.py`** — dead REPL entry point referencing
  removed `CodeAgent` import path and a removed `config.ollama_base_url`
  attribute; no caller, no test exercised it.
- **`fortranspire/brain/<uuid>/scratch/`** — local Loki-introspection
  experiments that were accidentally shipped in the wheel.
- **`Dockerfile.ci` and `Apptainer.analyze`** — referenced the
  removed `local_code_agent/` path; replaced by the consolidated
  container files under `containers/`.
- **Bibliography orphans** dropped from `paper/paper.bib`:
  `@clausecker2018claw`, `@gpufort2023`, `@gfortran`, `@flang`,
  `@langgraph2024`. Their content moved to
  `docs/concepts/architecture.md` (plain-text mentions, not bibtex
  citations).

### Fixed

- **`codemeta.json` version field** synchronised to the actual
  release version (was stuck at `0.1.3` after the 0.2.0 cut).
- **`integration/le-chat-connector.json` connector version**
  synchronised to 0.2.0 / 0.2.1.
- **`docs/conf.py` Sphinx release fallback** for unbuilt envs:
  `0.1.0` → `0.2.1`.
- **Dangling reference labels** in `docs/changelog.md`: added the
  `[0.1.2]` / `[0.1.3]` / `[0.2.0]` link definitions and bumped the
  `[Unreleased]` compare base from `v0.1.1` to the current release.

## [0.2.0] — 2026-06-24

Minor release. Mistral-vibe becomes a first-class integration surface
alongside Claude Code, with end-to-end stdio support proven on real
Météo-France PHYEX kernels.

### Added

- **`fortranspire mcp --stdio` transport** (issue #39). The MCP server
  now speaks the JSON-RPC stdio framing in addition to SSE. mistral-vibe
  and Claude Code Desktop can spawn it as a subprocess — no port, no
  auth middleware, local-by-construction. `MISTRAL_API_KEY` is
  propagated to the spawned process via the host's `[mcp_servers.env]`
  block in `~/.vibe/config.toml`.
- **Stdio smoke test** at `tests/server/test_mcp_stdio.py` —
  pytest-driven `initialize` → `notifications/initialized` →
  `tools/list` handshake against the installed console script; asserts
  the 9-tool surface intact.
- **Equivalence harness on real kernels** (issue #45). Test suite at
  `tests/test_equivalence_real_kernels.py` and fixture directory at
  `tests/fixtures/equivalence/<kernel>/` (`original.f90`, `openacc.f90`,
  `driver.f90`, `TOLERANCE.md`). For each kernel, the harness compiles
  both the serial and OpenACC variants with `gfortran` (the OpenACC one
  via `gfortran -fopenacc`), runs both binaries on a deterministic
  input, and asserts `np.allclose` within a per-kernel tolerance. First
  kernel landed: `wave_kernels` (two 2D FD stencils, 20 time-steps).
  Tests are marked `slow` and skip when gfortran is not in PATH.
- **PEP 735 marker registration** for `slow` in `[tool.pytest.ini_options]`.
- **PHYEX walkthrough** in `docs/integrations/mistral-vibe.md` — guided
  3-step demo (triage → call-graph → Phase-1 port) on
  `src/common/turb/mode_compute_function_thermo.F90`. Validates the
  end-to-end natural-language flow on a real Météo-France kernel.
- **`docs/getting-started/with-mistral-vibe.md`** — 60-second quickstart
  page for readers coming from LinkedIn / external announcements.

### Changed

- Mistral-vibe integration documentation rewritten around the stdio
  transport (recommended path). HTTP/SSE moved to "alternative" section
  with LaunchAgent template for permanent-service deployments.
- README promotes the mistral-vibe quick-start above the architectural
  detail (better signal for first-time visitors).

## [0.1.3] — 2026-06-22

Patch release. Restores a single-step PyPI install by depending on the
new [`loki-ifs`](https://pypi.org/project/loki-ifs/) PyPI distribution.

### Added

- **`loki-ifs>=0.3.7` is now a core dependency.** `pip install
  fortranspire` resolves Loki in one step. `loki-ifs` is an unofficial
  PyPI redistribution of `ecmwf-ifs/loki@0.3.7` published under our
  control (source: https://github.com/maurinl26/loki-ifs). The Python
  import name remains `loki`, so all parser code is unchanged.

### Changed

- **Removed the PEP 735 `[dependency-groups] loki` block** and the
  `[tool.uv.sources]` Loki override. Both were workarounds for the
  v0.1.2 era when Loki had to be installed by hand. `uv sync` now
  resolves Loki transparently from PyPI.

### Documentation

- README + `docs/getting-started/installation.md` rewritten to reflect
  the single-step `pip install fortranspire`. The "two-step install"
  callout is replaced with an explanation of how `loki-ifs` relates to
  upstream ECMWF Loki and how to override it if you want a different
  Loki version.

## [0.1.2] — 2026-06-22

Patch release. Unblocks the first PyPI upload by removing the direct
git-URL dependency on ECMWF Loki, which PEP 715 forbids in published
metadata.

### Changed

- **ECMWF Loki is no longer a declared dependency.** PyPI rejected the
  v0.1.0 / v0.1.1 wheels with `400 Can't have direct dependency: loki @
  git+https://github.com/ecmwf-ifs/loki@0.3.7`. Loki is not on PyPI
  under that name (the PyPI `loki` is an unrelated astronomy package),
  and PEP 715 disallows `git+https://…` URLs in `Requires-Dist`.
  fortranspire now ships with no Loki entry; users install it in a
  second step:
  ```bash
  pip install fortranspire
  pip install "loki @ git+https://github.com/ecmwf-ifs/loki@0.3.7"
  ```
  The parser already falls back to a regex frontend when Loki is
  absent, so the package imports and runs without it — `analyze` is
  simply more accurate when Loki is available.
- **Local development uses a PEP 735 dependency group.** A new
  `[dependency-groups]` block declares `loki = ["loki"]` and
  `[tool.uv.sources]` pins it to the git tag. Developers run
  `uv sync --group loki` to pull Loki in for local work; this metadata
  is not published to PyPI.
- **`[tool.hatch.metadata] allow-direct-references = true` removed.**
  No longer required now that no direct URL dep remains.

### Documentation

- README + `docs/getting-started/installation.md` rewritten with the
  new two-step install path and an explanatory callout on why Loki is
  separate.

## [0.1.1] — 2026-06-22

Patch release. Fixes two cosmetic issues found by the post-tag smoke
test of the v0.1.0 wheel.

### Fixed

- **Unified `fortranspire <verb>` CLI no longer prints the legacy
  `agent-*` deprecation notice** when dispatching to `gpu`,
  `translate`, or `profile`. Root cause: the dispatch table pointed
  at `run_*` wrappers that include the deprecation message. Split
  each into an internal `_*_main()` function (called by the unified
  CLI, no deprecation) and a legacy `run_*` wrapper (called by the
  `agent-*` scripts, prints the notice). The other 10 subcommands
  were already wired this way.
- **Package docstring** in ``fortranspire/__init__.py`` was a relic
  from the pre-rename project (referenced "Local Code Agent —
  LangChain + Mistral NeMo 12B via Ollama"). Replaced with the
  current project description and added ``__version__ = "0.1.1"``.

## [0.1.0] — 2026-06-22

First public release on PyPI. The project was renamed from `coding-agent` /
`local-code-agent` to **fortranspire** and shipped under a single
`fortranspire <verb>` entry point covering 13 subcommands. The release
bundles the JOSS submission package, a Sphinx documentation site, and
Zenodo / `CITATION.cff` / `codemeta.json` metadata for academic
citation.

The release closed **19 issues** opened during the development sprint
(#1 → #20), summarised below by theme.

### Added

#### Unified CLI (#8)

- **`fortranspire <verb>`** console script — single entry point with
  subcommand dispatch (cargo / kubectl / git pattern). 13 subcommands:
  `analyze`, `doc`, `explain`, `format`, `graph`, `diff`, `report`,
  `bench`, `gpu`, `port-batch`, `translate`, `profile`, `mcp`.
- Legacy `agent-analyze`, `agent-doc`, `agent-explain`, `agent-format`,
  `agent-port-batch`, `agent-gpu`, `agent-translate`, `agent-profile`,
  `run-mcp` kept for backwards compatibility. Each prints a one-line
  stderr deprecation notice pointing at the new form. **Removal
  scheduled for 0.3** (pinned in the message so users can plan).

#### Standalone subcommands — no LLM, CI-friendly

- **`fortranspire analyze`** — Loki-only static analyzer. 11 rules
  covering COMMON blocks, SAVE, I/O in kernels, missing IMPLICIT NONE
  / KIND, POINTER, derived types, suspected loop-carried deps, plus a
  toolchain check (compiler on PATH + OpenACC capability classification
  for `nvfortran` / `gfortran` / `ifx` / `flang` / `lfortran`). Outputs
  human-readable text, JSON, or SARIF 2.1.0. Ships a GitHub Actions
  workflow that uploads SARIF to Code Scanning, plus a lightweight
  `Apptainer.analyze` recipe for HPC sites.
- **`fortranspire explain`** (#14) — pre-flight cost + risk estimator.
  Runs Loki only and sums per-stage token estimates against the price
  table from #5 to produce a stakeholder-ready Markdown report
  (`Summary` + per-file breakdown + FORT001-009 risk roll-up). Zero
  LLM calls, zero tokens.
- **`fortranspire format`** — Fortran source formatter wrapping
  `fprettify` with sensible defaults (lowercase keywords, 2-space
  indent). Works standalone and as a post-step of the Phase 1 pipeline.
- **`fortranspire graph`** (#15) — module-level call-graph report.
  One Mermaid `flowchart LR` per file, external callees marked with a
  dashed border so cross-file edges are obvious. `--narrate` adds a
  1-paragraph LLM intro per file.
- **`fortranspire diff`** (#19) — semantic before/after viewer.
  Classifies each changed line into `[pragma]`, `[purity]`, `[type]`,
  `[refactor]`, `[common]`, `[save]`, `[other]`. Terminal output with
  per-category ANSI colors; `--html` writes a self-contained
  single-file page (no CDN, no JS) suitable for PR review or
  air-gapped reviewer machines.
- **`fortranspire report`** (#20) — single-file HTML audit dashboard
  per kernel. Embeds original source, refactored MODULE, OpenACC
  driver, Cython wrapper + header, validation log, and equivalence
  harness from #11; each section collapsible via `<details>` (pure
  HTML, no JavaScript). Pygments syntax highlighting if installed.
- **`fortranspire bench`** (#17) — pipeline-output metrics + regression
  detector. Counts files, routines, pragmas, generated bytes, LLM
  tokens (from observability traces), gfortran compilation timing.
  `--compare baseline.json` exits 1 if any metric regresses beyond
  `--tolerance` (default ±10 %, `--strict` = ±5 %).

#### LLM-driven subcommands

- **`fortranspire doc`** — documentation generator. Idempotent inline
  Doxygen `!>` blocks above each routine (`@brief` + `@details` +
  `@param` per argument). With `--sphinx` / `--site-only`, also
  generates a self-contained Sphinx site (Furo + MyST + sphinx-design
  + sphinx-togglebutton) with a *Show source* dropdown per routine.
  `--no-llm` runs Loki signature extraction only.
- **`fortranspire gpu --gpu-pragma {acc,omp}`** (#18) — Phase 1 port,
  now supporting two GPU directive families. `acc` (default,
  backwards-compatible) emits `!$acc parallel loop` / `!$acc data`.
  `omp` emits `!$omp target teams distribute parallel do` /
  `!$omp target data` — broader compiler audience (gfortran 13+,
  nvfortran -mp=gpu, ifx, AMD AOMP, IBM XL).
- **`fortranspire port-batch`** (#13) — parallel Phase 1 port across
  many files. Per-file output isolation via `ContextVar` so concurrent
  workers don't clobber `output/fortran_gpu/`. Default concurrency
  `min(4, cpu/2)`, configurable via `--concurrency`.

#### Server, observability, security

- **`fortranspire mcp`** — FastMCP server exposing every transformation
  as an MCP tool over HTTP/SSE.
- **Telemetry + cost tracking (#5)** — `fortranspire/observability/`
  package with a JSONL `Tracer`, per-model price table (Mistral La
  Plateforme rates), and a LangChain `BaseCallbackHandler` that
  captures token usage even when `with_structured_output()` swallows
  the raw message. Per-tenant accounting via `FORTRANSPIRE_TENANT_ID`.
  Optional OpenTelemetry export via `FORTRANSPIRE_OTEL_ENDPOINT`.
- **MCP server hardening (#10)** — `fortranspire/security/` package
  with a JSON-file token registry (legacy `API_KEY` env auto-promoted),
  per-tenant rate limiter (sliding window), HMAC-signed audit log
  (`FORTRANSPIRE_AUDIT_SECRET`). Backwards-compatible fallback to
  no-auth when nothing is configured.
- **Content-addressed LLM cache (#7)** — `fortranspire/cache/` package
  wiring an SQLite-backed `BaseCache` as LangChain's process-wide LLM
  cache. Identical re-runs return instantly with **zero token cost**.
  LRU eviction at configurable byte cap (default 1 GiB,
  `FORTRANSPIRE_CACHE_MAX_GB`). Disable with `FORTRANSPIRE_CACHE=off`.

#### LLM backend dispatch (#9)

- **Native `mistralai` SDK** auto-selected when the endpoint matches
  Mistral La Plateforme — gets first-class function calling, JSON
  mode, streaming, Mistral safety guards. Falls back gracefully to
  `ChatOpenAI` for self-hosted vLLM / TGI / Ollama on an OpenStack tenant.
  Override via `FORTRANSPIRE_LLM_BACKEND={mistral,openai}`.

#### Equivalence harness (#11)

- **Auto-generated CPU↔GPU test file** per ported kernel
  (`output/tests/test_<kernel>_equivalence.py`). Loads f2py-wrapped
  CPU reference and Cython-wrapped GPU port, runs random inputs,
  asserts `np.testing.assert_allclose` on every INTENT(OUT|INOUT).
  Skips cleanly when either build is missing. Tolerances configurable
  via `FORTRANSPIRE_TOLERANCE_ATOL/RTOL`.

#### Prompts, schemas, i18n

- **Externalized prompts (#3)** — every system prompt moved out of
  Python source into `fortranspire/prompts/<name>/<lang>/<version>.md`.
  Loader `load_prompt(name, version=, lang=, **vars)` with versioning
  (so prompt evolution is auditable), locale fallback (FR → EN), and
  per-deployment override via `FORTRANSPIRE_PROMPTS_DIR`. LRU-cached
  file reads.
- **French translations (#16)** — 11 new FR prompt variants covering
  all 6 LLM-emitting stages. Activate with `FORTRANSPIRE_LANG=fr` or
  `load_prompt(..., lang="fr")`. Initial audience (Météo-France, CEA,
  EDF R&D, ENM Toulouse) is francophone; Mistral handles French
  exceptionally well at zero quality cost.
- **Pydantic structured outputs (#4)** — every LLM-emitting stage now
  returns a typed Pydantic object via
  `llm.with_structured_output(SchemaModel)` instead of regex-parsing
  free-form text. Schemas: `ExtractorOutput`, `OpenACCKernelOutput`,
  `OpenACCDriverOutput`, `CythonPyxOutput`, `CythonHeaderOutput`,
  `DocRoutineOutput`. v2 prompts emit JSON; v1 kept as legacy fallback
  for backends without JSON-schema mode (triple fallback in the
  extractor for robustness).
- **`fortls` oracle (#12)** — the Fortran Language Server is now wired
  into the documenter prompts. Neighbouring-symbol context fetched per
  routine reduces hallucinations on names/relationships. Graceful
  no-op when `fortls` is missing.

#### Architecture refactors

- **LangGraph nodes split (#2)** — `translation_graph_phase1.py`
  (~1400 lines) decomposed into per-node modules under
  `fortranspire/agent/nodes/` (`init`, `parser`, `extractor`,
  `pure_elemental`, `openacc`, `cython_wrapper`, `equivalence_harness`,
  `validation`). The main file is now ~80 lines of graph wiring.
  Public surface preserved via re-exports for backwards compat.

#### Documentation & publishing

- **JOSS submission package** — `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `paper.md`,
  `paper.bib`, `.zenodo.json` (DOI metadata, ORCID
  `0009-0004-8117-4850`, affiliation: Independent Researcher),
  `.readthedocs.yaml`, and `.github/workflows/draft-paper.yml` to
  render the JOSS draft PDF on every push.
- **`CITATION.cff` + `codemeta.json`** at the repo root — GitHub
  surfaces the "Cite this repository" button; `codemeta.json` is the
  semantic-web companion consumed by Zenodo, Software Heritage, and
  academic search engines.
- **Sphinx documentation site** under `docs/` (Furo + MyST +
  extensions), with sections for installation, quickstart,
  configuration, pipeline architecture, Fortran patterns, LLM
  endpoints, Mistral integration, Le Chat connector preparation, and
  the standalone documentation feature.
- **Mistral integration documentation** —
  `docs/concepts/mistral-integration.md` covers four integration
  paths (LLM consumer, per-stage models, MCP provider, Le Chat
  connector directory). Connector manifest prepared in
  `integration/le-chat-connector.json` and documented in
  `docs/concepts/le-chat-connector.md`. Runnable smoke test at
  `examples/mistral_agents_api_smoke_test.py`.

### Changed

- **Project renamed** from `coding-agent` / `local-code-agent` to
  **fortranspire**. Python package directory, PyPI distribution name,
  GitHub URLs, docs, CI workflows, and the inline
  `!> @generated_by` marker were all updated. Use
  `pip install fortranspire` and `from fortranspire.agent...`.
- **Loki dependency** repointed from a hard-coded local path
  (`file://localhost/Users/loicmaurin/PycharmProjects/...`) to a
  pinned git tag (`git+https://github.com/ecmwf-ifs/loki@0.3.7`). The
  repo is now installable on any fresh machine.
- **Dependencies split into extras**: default `uv sync` installs
  ~50 MB (Loki + NumPy + LangGraph + python-dotenv) — enough for
  `analyze`, `explain`, `format` (`--no-llm`), `graph`, `diff`,
  `report`, `bench`. Pull `[gpu]` for the LLM stack + Cython +
  fprettify + fortls, `[mcp]` for the FastMCP server, `[jax]` for
  Phase 2, or `[all]`.
- **LLM model selection per pipeline stage** — `MISTRAL_MODEL_REASONING`
  (default `mistral-large-latest`, used by `extractor` and `openacc`)
  and `MISTRAL_MODEL_CODE` (default `codestral-latest`, used by
  `cython_wrapper`). Legacy `MISTRAL_MODEL` still works as a single
  fallback. Cuts cost per kernel ~50 % on default settings.
- **Pipeline node order** — the equivalence-harness node from #11 is
  inserted between `cython_wrapper` and `validation`, generating the
  test harness file before the final validation step.

### Fixed

- **Parser missed module-contained routines** — `parser_phase1` only
  looked at `source.routines` (top-level), which is empty for modern
  Fortran 90 codes that wrap subroutines in
  `module ... contains ... end module`. Now also walks
  `source.modules[].subroutines`.
- **Off-by-one in `agent-doc` line numbers** — `\s` in the routine
  declaration regex was matching `\n`, letting the match start on the
  preceding blank line. Same bug caused the indent capture to take a
  newline as whitespace. Switched to `[ \t]*` so docstring blocks
  land above the right line with the routine's own indent.
- **LangChain imports** in `translation_graph_phase1.py` were
  unconditionally loaded at module import — forcing `agent-analyze`
  and friends to require `[gpu]`. Now lazy inside the three
  LLM-using node functions, so `uv sync` (core only) suffices for
  every no-LLM command.
- **`agent-port-batch --help`** crashed on the core-only install for
  the same reason. Same lazy-import fix applies in
  `fortranspire/agent/cli.py`.
- **Cache `langchain.globals` import path** — modern langchain moved
  `set_llm_cache` to `langchain_core.globals`; the install hook
  follows the new location.

## [0.0.x] — pre-rename snapshot (2026-06-17)

The pre-rename project was published informally as `coding-agent`. It
shipped the initial MCP server (`run-mcp`), the legacy CLI
(`agent-gpu`, `agent-translate`, `agent-profile`, `agent-pipeline`),
the original Phase 1 and Phase 2 LangGraph pipelines, Loki-based
deterministic Fortran AST analysis, Docker / docker-compose / Apptainer
recipes, and the pivot from Azure to a sovereign Mistral endpoint
(commit `ccfe221`). All of that is preserved under fortranspire 0.1.0.

[Unreleased]: https://github.com/maurinl26/fortranspire/compare/v0.2.0...HEAD
[0.2.0]:      https://github.com/maurinl26/fortranspire/releases/tag/v0.2.0
[0.1.3]:      https://github.com/maurinl26/fortranspire/releases/tag/v0.1.3
[0.1.2]:      https://github.com/maurinl26/fortranspire/releases/tag/v0.1.2
[0.1.1]:      https://github.com/maurinl26/fortranspire/releases/tag/v0.1.1
[0.1.0]:      https://github.com/maurinl26/fortranspire/releases/tag/v0.1.0
[0.0.x]:      https://github.com/maurinl26/fortranspire/commits/v0.1.0/
