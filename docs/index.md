# fortranspire

**An LLM + MCP pipeline that ports legacy Fortran HPC kernels to GPU
(OpenACC) and differentiable JAX.**

`fortranspire` automates the four transformations a senior HPC engineer
would perform by hand when porting a Fortran 90 kernel to GPU and Python:

1. **Parse** the source with [Loki](https://github.com/ecmwf-ifs/loki)
   (ECMWF) — deterministic AST analysis of `INTENT`, `SAVE`, `COMMON`,
   `POINTER`, and inline loops.
2. **Extract** the kernels into a `MODULE` with explicit `INTENT(INOUT)`
   arguments — one LLM call against any OpenAI-compatible endpoint.
3. **Annotate** purity (`PURE`/`ELEMENTAL`) and insert OpenACC pragmas
   (`!$acc parallel loop collapse(...)`, `!$acc data copyin/copy`).
4. **Wrap** in Cython (NumPy typed memoryviews, `iso_c_binding` header) and
   validate with `gfortran` and `nvfortran -acc`.

The pipeline is exposed as a CLI **and** as a Model Context Protocol (MCP)
server, so it is callable from any MCP-aware client (Claude Desktop,
Cursor, VS Code, Mistral Le Chat connectors) or any CI runner.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/with-mistral-vibe
getting-started/installation
getting-started/quickstart
getting-started/configuration
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/architecture
concepts/fortran-patterns
concepts/llm-endpoints
concepts/mistral-integration
concepts/le-chat-connector
concepts/legacy-documentation
concepts/jax-optimization
concepts/gt4py-next-patterns
concepts/gt4py-next-cheatsheet
```

```{toctree}
:maxdepth: 2
:caption: Integrations

integrations/claude-code
integrations/mistral-vibe
integrations/ide-interactive-mode
integrations/claude-code-action
integrations/copilot-coding-agent
integrations/github-app
integrations/le-chat
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
security
code-of-conduct
changelog
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
