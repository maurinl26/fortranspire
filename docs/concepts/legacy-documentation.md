# Documenting a legacy Fortran codebase

`agent-doc` is a **standalone documentation generator** for legacy
Fortran 90 code. It can run completely outside the GPU porting pipeline —
its only goal is to produce readable, structured documentation for codes
that have lost their original authors.

Two outputs are available, individually or together:

- **Inline Doxygen-style docstrings** (`!>` blocks) injected directly above
  each `subroutine` / `function`. Idempotent — re-runs *replace* the
  previous block rather than stacking new ones.
- **A self-contained Sphinx site** (`--sphinx` or `--site-only`) with one
  `.rst` per file and per-routine sections, including a *Show source*
  toggle when the togglebutton extension is present.

Every routine is documented at **two levels** in a single LLM call:

- `short_summary` (≤ 1 line) — stakeholder / project-manager view.
- `detailed` (2–4 sentences) — developer view, mentioning `INTENT`
  semantics, invariants, and known gotchas (hidden `SAVE`, `COMMON`
  blocks, index ordering).

## Quick start

```bash
# Annotate every kernel in src/ with inline !> docstrings
uv run agent-doc src/

# In addition, generate a Sphinx site under documentation/<project>/
uv run agent-doc --sphinx src/

# Sphinx site only — leave the source files alone
uv run agent-doc --site-only src/

# Show what would be inserted, do not modify the source
uv run agent-doc --dry-run src/kernel.f90

# No LLM call — useful in CI / for offline runs / to verify the plumbing
uv run agent-doc --no-llm src/
```

The generated Sphinx site builds with:

```bash
cd documentation/<project>
pip install -r requirements.txt
make html
open build/html/index.html
```

## What the inline docstring looks like

For a routine `update_vx` with three arguments, the generated block above
the `subroutine` keyword looks like this:

```fortran
!> @generated_by fortranspire v1 routine=update_vx body=8c9f1e2a3b4c
!> @brief   Update the horizontal velocity component using a one-sided FD
!>          stencil on the σxx field.
!> @details Reads sigma_xx at (i,j) and (i-1,j); writes vx in place.
!>          INTENT(INOUT) on vx is required because the time loop calls
!>          this routine N times accumulating updates. Assumes a regular
!>          Cartesian grid; dx must be > 0.
!> @param[inout] vx        Horizontal velocity field, modified in place.
!> @param[in]    sigma_xx  Stress tensor xx component, read-only.
!> @param[in]    dx        Grid spacing in the x direction.
subroutine update_vx(vx, sigma_xx, dx, nx, ny)
  ...
```

The `@generated_by fortranspire` marker on the first line is the
**idempotency anchor**. Re-running `agent-doc` detects the existing block,
strips it, and emits a fresh one — your source file never accumulates
duplicate documentation.

The trailing `body=<hash>` lets a future incremental mode skip routines
whose body has not changed since the last documentation pass.

## Operating modes

| Flag             | Effect                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| *(default)*      | Inline `!>` docstrings, source files modified in place                    |
| `--sphinx`       | Inline + generate `documentation/<project>/` Sphinx site                  |
| `--site-only`    | Generate the Sphinx site only, leave source files untouched               |
| `--no-llm`       | Skip the LLM calls (signatures / argument list only, no narrative)        |
| `--dry-run`      | Print the rewritten source to stdout, do not write anything               |
| `--project NAME` | Project name used in the Sphinx site title (default: inferred from path)  |
| `--output DIR`   | Where the Sphinx site is written (default: `documentation/`)              |

## Cost model

One LLM call per routine, against the Codestral model
(`MISTRAL_MODEL_CODE`, which defaults to `codestral-latest`). At
Codestral tariffs:

| Codebase size   | LLM calls | Wall-clock | Cost   |
| --------------- | --------- | ---------- | ------ |
| 10 routines     | 10        | ~30 s      | ~0.02 USD |
| 50 routines     | 50        | ~3 min     | ~0.10 USD |
| 200 routines    | 200       | ~10 min    | ~0.40 USD |

Re-runs on unchanged routines will become free in a future iteration
(the body hash is already recorded in the inline block — only the cache
lookup is missing). For now, prefer `--dry-run` on a single file when
iterating on prompts.

## Pairing with the analyzer

`agent-doc` and `agent-analyze` complement each other:

- `agent-analyze` answers *"is this code GPU-ready?"* — deterministic,
  zero LLM, runs in CI.
- `agent-doc` answers *"what does this code do?"* — LLM-driven,
  one-shot, run by the maintainer or as a release artifact.

Run the analyzer first to surface the structural issues
(`COMMON`, `SAVE`, missing `INTENT`); then run `agent-doc` to capture
the *intent* the analyzer cannot guess.

## Limitations

- The Loki AST extraction handles modules and free-form Fortran 90 well;
  fixed-form sources (Fortran 77 with `*` columns) may need a manual
  conversion first.
- LLM-generated narratives describe what the routine *looks like it
  does*, not what it *was meant to do*. Always review for domain-specific
  accuracy before publishing.
- The Sphinx site uses `furo` and pure-RST per-routine sections. It does
  not currently use `sphinxfortran` (the project is largely unmaintained);
  a `Show source` toggle is added when `sphinx-togglebutton` is installed.
