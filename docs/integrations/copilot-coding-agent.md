# GitHub Copilot cloud agent

*Issue [#49](https://github.com/maurinl26/fortranspire/issues/49).*

Teams already on Copilot can reach fortranspire without adopting a second
tool: the Copilot cloud agent (formerly "coding agent") speaks MCP, so the
existing server plugs straight in.

This integration deliberately exposes **only the deterministic tools** —
`analyze_kernels`, `explain_port_cost`, `build_call_graph`. No LLM call, no
token, no secret, no file written.

That restraint is not caution for its own sake. Copilot invokes MCP tools
**autonomously, with no approval prompt**. Anything you allow-list runs
unattended, so a token-spending, code-writing tool does not belong here
until you have decided you want that.

## Two files, one setting

### 1. `.github/workflows/copilot-setup-steps.yml`

Installs fortranspire into the agent's environment. Already in this
repository; copy it into yours.

Three things about it are load-bearing:

- **The job must be called `copilot-setup-steps`.** Any other name and
  GitHub silently ignores the file.
- **It only takes effect from the default branch.**
- Only `steps`, `permissions`, `runs-on`, `services`, `snapshot` and
  `timeout-minutes` (max 59) are honoured. Everything else is dropped
  without a warning.

It installs into a dedicated venv at `/opt/fortranspire`, which the MCP
configuration names by absolute path. Two reasons: whether `PATH` changes
made in setup steps are inherited by the process that launches MCP servers
is not documented, and installing into the runner's Debian-managed system
Python simply fails — pip cannot uninstall distro-owned packages
(`Cannot uninstall typing_extensions … RECORD file not found`). A venv owns
its own dependencies and avoids both problems.

### 2. The MCP configuration

Either commit an agent profile, or paste JSON into the repository settings.

**Committed** — [`.github/agents/fortran-porter.agent.md`](https://github.com/maurinl26/fortranspire/blob/main/.github/agents/fortran-porter.agent.md).
This is the only mechanism that keeps the configuration in Git, and it is
scoped to one custom agent:

```yaml
mcp-servers:
  fortranspire:
    type: 'local'
    command: '/opt/fortranspire/bin/fortranspire'
    args: ['mcp', '--stdio']
    tools: ['analyze_kernels', 'explain_port_cost', 'build_call_graph']
```

**Repository-wide** — Settings → Copilot → MCP servers, then paste:

```json
{
  "mcpServers": {
    "fortranspire": {
      "type": "local",
      "command": "/opt/fortranspire/bin/fortranspire",
      "args": ["mcp", "--stdio"],
      "tools": ["analyze_kernels", "explain_port_cost", "build_call_graph"]
    }
  }
}
```

`type: "local"` is the canonical spelling for a stdio server; Copilot maps
`stdio` onto it for compatibility with Claude Code and VS Code.

Always list `tools` explicitly. The documented behaviour when it is omitted
is ambiguous, and the setting decides what runs without asking you.

## Smoke test

1. Open an issue: *"How much would a GPU port of `src/kernel.f90` cost?"*
2. Assign it to Copilot.
3. When the PR appears, open **View session** → **…** → **Copilot** →
   **Start MCP Servers**. A server that started lists its tools at the
   bottom of that log; that is the check that the config took.

The agent should call `explain_port_cost` and answer with the routine count,
the structural risks and the estimated token cost — with no token spent
reaching that answer.

## Secrets, if you ever need them

Only needed if you later expose an LLM tool. Copilot reads a dedicated
**Agents** secret type (Settings → Security → Secrets and variables →
Agents), not Actions secrets, and only names prefixed `COPILOT_MCP_` reach
the MCP configuration:

```json
"env": { "MISTRAL_API_KEY": "$COPILOT_MCP_MISTRAL_API_KEY" }
```

## Limits worth knowing

- **Tools only.** MCP resources and prompts are not supported.
- **No OAuth** for remote MCP servers. Not an issue for a local one.
- **The agent firewall does not apply to MCP servers.** Processes Copilot
  starts through its Bash tool are filtered; your MCP server is not. It has
  unrestricted network access — worth weighing if that matters to you.
- **Ubuntu and Windows runners only.** No macOS.
- A failing setup step does not abort the run: Copilot skips the rest and
  starts anyway, so a broken install shows up as "the tools aren't there"
  rather than as an error.

## Why not the porting tools too

Porting is a deliberate act with a cost attached, so it lives behind an
explicit trigger — `@fortranspire` on a PR, via the
[Claude Code Action](claude-code-action.md), or `/fortranspire port` through
the [GitHub App](github-app.md). Both gate on write access first.
