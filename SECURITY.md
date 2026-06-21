# Security Policy

## Supported versions

`fortranspire` is in active development on `main`. Security fixes are
applied to `main` and to the most recent tagged release.

| Version       | Supported           |
| ------------- | ------------------- |
| `main`        | :white_check_mark:  |
| latest tag    | :white_check_mark:  |
| older tags    | :x:                 |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report vulnerabilities privately by either:

1. **GitHub Security Advisory** — preferred. Open a private advisory at
   <https://github.com/maurinl26/fortranspire/security/advisories/new>.
   This gives you a private channel with the maintainer and a tracked CVE
   workflow.
2. **Email** — write to `maurin.loic.ac@gmail.com` with the subject line
   `[security] fortranspire: <short description>`. Please include:
   - a description of the vulnerability and its impact,
   - the affected version (commit SHA or release tag),
   - reproduction steps or a proof-of-concept, and
   - any suggested mitigation.

You can encrypt your report — request the maintainer's PGP key in a first
email if you need it.

## Response expectations

- **Acknowledgement** within 5 business days.
- **Initial assessment** (confirmed / not reproducible / out of scope)
  within 14 days.
- **Fix or mitigation timeline** communicated within 30 days of
  acknowledgement. Critical issues are prioritized.
- **Public disclosure** is coordinated with the reporter. We aim to publish
  an advisory and a patched release on the same day, with credit to the
  reporter unless they prefer to remain anonymous.

## Scope

In scope:

- The Python package `fortranspire/`, including the MCP server,
  LangGraph pipelines, and CLI entry points.
- Container images built from `Dockerfile`, `Dockerfile.ci`,
  `Dockerfile.hpc`, and `apptainer.def`.
- The deployment recipes under `infrastructure/` and `scripts/`.

Out of scope:

- Vulnerabilities in upstream dependencies (Loki, LangChain, FastMCP,
  Mistral SDK, JAX, …) unless `fortranspire` makes them exploitable in a
  configuration we ship by default. Please report those to the upstream
  project; we are happy to coordinate.
- Issues that require an attacker to already have local shell access or
  write access to your `.env` file.
- DoS through resource exhaustion on the LLM endpoint — this is a property
  of your endpoint provider, not of this project.

## Operational notes for users

`fortranspire` executes generated Fortran, Cython, and Python code and
invokes external compilers (`gfortran`, `nvfortran`). Treat it as you would
any code-generation tool:

- Run it inside an isolated environment (container, devcontainer, VM)
  whenever the input Fortran or the LLM endpoint is not fully trusted.
- The MCP server listens on `0.0.0.0:8000` by default. Set `API_KEY` in your
  environment to enable bearer-token authentication, or front it with a
  reverse proxy that enforces auth and TLS.
- Do not commit `.env` or any file containing API keys. The repository
  ships an `.env.example` that lists every variable the agent reads.
