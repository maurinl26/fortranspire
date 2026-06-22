# Use fortranspire from Claude Code

`fortranspire` ships a ready-to-install [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)
so Claude can analyze, port, and document Fortran 90 code from inside
the IDE — no MCP plumbing required.

## Local skill (recommended)

The skill is `skills/claude-code/fortranspire/` in the repo. Copy it
into your project's `.claude/skills/`:

```bash
pip install fortranspire           # the actual binary
# In whatever Claude-Code project you want to use it from:
mkdir -p .claude/skills
cp -r <fortranspire-repo>/skills/claude-code/fortranspire .claude/skills/
```

…then reload Claude Code. The `/fortranspire` slash command appears,
and Claude auto-invokes the skill on Fortran-related requests.

The skill's [`SKILL.md`](https://github.com/maurinl26/fortranspire/blob/main/skills/claude-code/fortranspire/SKILL.md)
documents every subcommand and includes a decision flow — Claude always
runs `fortranspire explain` first (zero LLM cost, tells you whether the
file is portable) before recommending the paid `gpu` or `translate`
verbs.

### What needs the LLM, and what doesn't

| Verb              | LLM call? | API key needed? |
| ----------------- | --------- | --------------- |
| `analyze`         | No        | No              |
| `explain`         | No        | No              |
| `format`          | No        | No              |
| `graph`           | No        | No              |
| `diff`            | No        | No              |
| `report`          | No        | No              |
| `bench`           | No        | No              |
| `doc --no-llm`    | No        | No              |
| `doc --with-llm`  | Yes       | Yes             |
| `gpu`             | Yes       | Yes             |
| `port-batch`      | Yes       | Yes             |
| `translate`       | Yes       | Yes             |

For LLM verbs, see [LLM endpoints](../concepts/llm-endpoints.md).

## Remote MCP (advanced)

If your Claude Code is configured to talk to an MCP server (e.g., on a
shared GPU host), point it at a running `fortranspire mcp`:

```bash
# On the server
fortranspire mcp --port 8000
# or with env vars for the container
MCP_HOST=0.0.0.0 MCP_PORT=8000 fortranspire mcp
```

…and register `http://<host>:8000/sse` in Claude Code's MCP settings.
The server exposes nine tools — the same surface as the CLI minus the
`format` / `diff` / `report` verbs which are best run locally on the
user's filesystem:

- `analyze_kernels` — static analysis, optional SARIF output
- `explain_port_cost` — pre-flight estimate (no LLM)
- `build_call_graph` — Mermaid call-graph
- `generate_docs` — `!>` docstring injection (no-LLM or LLM-driven)
- `translate_kernel_gpu` — Phase 1 port
- `translate_kernel` — Phase 2 JAX translation
- `profile_kernels` — performance comparison
- `ask_agent` — natural-language query
- `agent_status` — server config dump

When `FORTRANSPIRE_TOKENS_FILE` or `API_KEY` is set, the MCP server runs
authenticated with rate-limiting + HMAC-signed audit logs
(`docs/concepts/security.md`).

## Troubleshooting

- **`/fortranspire` not in the slash-command menu** — verify
  `.claude/skills/fortranspire/SKILL.md` exists and Claude Code has been
  reloaded (close and re-open the project).
- **Claude says "fortranspire: command not found"** — `pip install
  fortranspire` in the same venv Claude Code uses, or in `~/.local/bin`
  if you're using system Python.
- **LLM verb fails with "no MISTRAL_API_KEY"** — add the key to
  `.env` at the project root, or export it in the shell Claude Code
  spawns (`~/.zshrc`, `~/.bashrc`).
