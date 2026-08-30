# Claude Code GitHub Action — porting from a PR comment

*Issue [#48](https://github.com/maurinl26/fortranspire/issues/48).*

A maintainer comments on a pull request:

```
@fortranspire porte src/common/turb/mode_thermo.F90 en GPU
```

A CI job starts Claude Code with the fortranspire MCP server wired in, runs
the Phase-1 transformation through the same tools the CLI uses, and pushes
the result to a branch.

This is the cheapest agent surface to wire, because it reuses the MCP server
as-is. It is also the one that **spends tokens and writes code**, so keep it
to repositories you trust.

Workflow: [`.github/workflows/claude-port.yml`](https://github.com/maurinl26/fortranspire/blob/main/.github/workflows/claude-port.yml).

## Setup

Two repository (or organisation) secrets:

| Secret | Purpose |
| ------ | ------- |
| `ANTHROPIC_API_KEY` | Runs Claude Code itself |
| `MISTRAL_API_KEY` | Used by the fortranspire MCP server for the transformation |

Both LLMs appear because they do different jobs: Claude drives the session
and decides which tool to call; codestral performs the kernel translation
inside the pipeline. To keep the whole thing on an EU endpoint, point the MCP
server at a self-hosted vLLM instead — see
[LLM endpoints](../concepts/llm-endpoints.md).

Then copy the workflow into your repository. No other configuration.

## Who can trigger it

The action checks that the commenter has **write access** before Claude runs;
a comment from anyone else is ignored. The `if:` condition on the job is only
a cheap pre-filter to avoid spinning up a runner.

Do not add `allowed_non_write_users` unless you have thought hard about it:
it lets anyone who can comment on a public PR spend your tokens.

## The workspace jail

This is the part worth understanding before you deploy it.

`fortranspire mcp --stdio` **disables the workspace jail by default**. That
is the right default for a local IDE — the user owns the trust boundary and
wants to analyse a checkout anywhere on disk. It is the wrong default here,
because the agent is acting on a comment somebody else wrote.

The workflow therefore sets, both on the action's own process and inside the
MCP server's environment:

```yaml
FORTRANSPIRE_DISABLE_JAIL: '0'
AGENT_WORKSPACE: ${{ github.workspace }}
```

With those, every path the tools accept is resolved and checked against the
checkout. `tests/test_agent_surfaces.py` fails if either is dropped.

## Secrets stay off the command line

The MCP configuration references `${MISTRAL_API_KEY}`, and the workflow puts
the real value in the step's `env:`. Claude Code substitutes it when it
launches the server. Interpolating `${{ secrets.MISTRAL_API_KEY }}` straight
into `claude_args` would put the key in the process list instead — a test
guards against that too.

## What Claude is told to do

The prompt makes the free step mandatory:

1. `explain_port_cost` first — it costs nothing and reports the structural
   risks. A blocking finding (I/O inside the kernel, a loop-carried
   dependency) stops the run before any tokens are spent on a port that
   would not have worked.
2. `analyze_kernels` for the detail.
3. `translate_kernel_gpu` (or `translate_kernel` for JAX) only if the file
   is actually portable.

It is also told **not** to hand-write OpenACC pragmas. The point of the MCP
server is that the Loki AST backbone does that deterministically; an agent
that "helps" by editing the Fortran directly has bypassed the guarantee you
installed it for.

## Cost control

- `--max-turns 20` bounds a runaway session.
- `concurrency` serialises runs per thread — two concurrent ports would race
  on the same branch and on `output/`.
- `timeout-minutes: 30` bounds the job.
- Step 1 of the prompt is the real saver: it refuses unportable files before
  the expensive call.

## Relationship to the other surfaces

This complements the [Claude Code skill](claude-code.md) rather than
replacing it: the skill runs on a developer's machine, this runs on GitHub.
For unattended analysis with no key at all, use the
[composite Action](../../README.md#-github-action-zero-token-ci).
