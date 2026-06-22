# Use fortranspire from mistral-vibe

[`mistral-vibe`](https://chat.mistral.ai/) — Mistral's hosted IDE / agent
surface — speaks the [Model Context Protocol](https://modelcontextprotocol.io)
natively. fortranspire ships its FastMCP server pre-configured for this
use case, so the **same agent** can run from Claude Code or from
mistral-vibe with a model swap, no code change.

This is the strongest demonstration of the sovereignty story (see
[`Architecture vs LLM`](../concepts/llm-endpoints.md)): Loki + the
deterministic harness absorb 60-70% of the work, and the LLM (Mistral
on mistral-vibe, Claude on Claude Code) is a fungible component.

## Start the MCP server

```bash
pip install "fortranspire[mcp]"
fortranspire mcp --port 8000
# or with env vars for a container / hosted deploy
MCP_HOST=0.0.0.0 MCP_PORT=8000 fortranspire mcp
```

The server listens on SSE at `http://<host>:8000/sse`.

For a public deploy, enable auth:

```bash
export FORTRANSPIRE_TOKENS_FILE=/etc/fortranspire/tokens.json
export FORTRANSPIRE_AUDIT_SECRET=<random-256-bit-secret>
fortranspire mcp
```

See [security](../security.md) for the token registry format
and the HMAC-signed audit log.

## Register the server in mistral-vibe

In your mistral-vibe project settings, add an MCP server:

```jsonc
{
  "mcpServers": {
    "fortranspire": {
      "url": "http://<host>:8000/sse",
      "headers": {
        "Authorization": "Bearer <token-from-FORTRANSPIRE_TOKENS_FILE>"
      }
    }
  }
}
```

mistral-vibe will discover the nine exposed tools:

| Tool                     | What it does                                          | LLM call?       |
| ------------------------ | ----------------------------------------------------- | --------------- |
| `analyze_kernels`        | Loki AST analysis, optional SARIF output              | No              |
| `explain_port_cost`      | Pre-flight cost & risk estimate                       | No              |
| `build_call_graph`       | Mermaid call-graph                                    | No              |
| `generate_docs`          | `!>` docstring injection (no-LLM or LLM-driven)       | Optional        |
| `translate_kernel_gpu`   | Phase 1 — Fortran → OpenACC + Cython                  | Yes             |
| `translate_kernel`       | Phase 2 — Fortran → JAX (experimental)                | Yes             |
| `profile_kernels`        | Performance benchmarking                              | No              |
| `ask_agent`              | Natural-language query against the code               | Yes             |
| `agent_status`           | Dump server config                                    | No              |

## Pointing the LLM at Mistral

The MCP server itself does not call the LLM — the **client** (Claude
Code, mistral-vibe) drives the conversation. The LLM verbs (`gpu`,
`translate`, `doc --with-llm`) call Mistral from inside the server
process, so the server's `.env` needs:

```bash
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<your-mistral-key>"
MISTRAL_MODEL="codestral-latest"           # or mistral-large-latest
```

For fully on-prem inference, point the server at a self-hosted vLLM /
TGI / Ollama (see
[LLM endpoints](../concepts/llm-endpoints.md)).

## Verify the integration

In mistral-vibe, ask:

> Run `explain_port_cost` on `path/to/kernel.f90`.

Mistral should call the MCP tool, surface the routine count + port-cost
table, and offer to `translate_kernel_gpu` next. If `explain_port_cost`
isn't listed, check the MCP registration in mistral-vibe settings.

## Why both Claude Code and mistral-vibe?

| Surface         | LLM            | Sovereignty                                   |
| --------------- | -------------- | --------------------------------------------- |
| Claude Code     | Claude (US)    | Mixed — code stays local, model is US-hosted  |
| mistral-vibe    | Mistral (EU)   | Full — code + inference both EU-resident      |
| `fortranspire mcp` self-hosted | Mistral self-hosted | Air-gapped possible |

Same agent, three sovereignty postures. Choose per project / per
customer requirement.

## Known gaps

- mistral-vibe MCP connection has not been smoke-tested end-to-end at
  the time of writing (issue [#39](https://github.com/maurinl26/fortranspire/issues/39))
  — the integration follows the MCP spec strictly, so it should work,
  but please file an issue with the tool-call payload if anything
  diverges.
- The MCP server returns plain-text responses (Markdown-friendly).
  Structured-output support (Pydantic JSON schemas) is on the roadmap
  — track progress in the [issue tracker](https://github.com/maurinl26/fortranspire/issues).
