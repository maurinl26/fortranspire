# Mistral integration

`fortranspire` is built **Mistral-first** but **Mistral-only is not
required**. The agent talks to *any* OpenAI-compatible endpoint, so the
same code paths work against:

- **Mistral La Plateforme** (`https://api.mistral.ai/v1`) — sovereign EU
  hosting, the default.
- **Self-hosted vLLM / TGI / Ollama** — full sovereignty, on an OpenStack
  tenant, an HPC site (GENCI Jean Zay, EuroHPC LUMI / MareNostrum), or
  any private datacenter.
- **EU OpenAI-compatible gateways** — any third-party gateway exposing
  the `/chat/completions` schema and respecting EU data residency.

**Azure is not part of the integration.** A previous version of the project
proxied LLM calls through Azure OpenAI; this dependency was removed in
commit `ccfe221`. There is no `AZURE_*` environment variable to set, no
Azure-specific code path, and no Azure resource in the install
instructions.

This page walks through the four integration paths.

## Path 1 — Agent as a Mistral La Plateforme consumer

The most common setup. The agent uses your Mistral API key to call any
Mistral chat model.

### One-time setup

1. Create an API key at <https://console.mistral.ai/api-keys>.
2. Copy it into `.env` at the repo root:

```bash
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<paste-your-key>"
```

3. Verify the key works:

```bash
curl https://api.mistral.ai/v1/models \
  -H "Authorization: Bearer $MISTRAL_API_KEY" | jq '.data[].id' | head
```

You should see `mistral-large-latest`, `codestral-latest`, `mistral-nemo`,
etc.

### Run a transformation

```bash
uv run agent-gpu path/to/kernel.f90
```

The pipeline issues up to four LLM calls (~0.06 USD at Mistral-Large
tariffs, ~2 min wall-clock).

## Path 2 — Per-stage model selection (Codestral + Mistral-Large)

The pipeline has two kinds of LLM work:

- **Reasoning** stages (`extractor`, `openacc`) — refactoring a monolithic
  `PROGRAM` into a `MODULE`, placing `!$acc data copyin/copy` clauses
  around the time loop. Wrong placement = silent GPU corruption. **Needs
  a large reasoning model.**
- **Code-gen** stages (`cython_wrapper`) — emitting a `.pyx` wrapper and
  an `iso_c_binding` header. Mostly boilerplate. **Codestral is faster and
  cheaper** for this kind of work.

The agent picks the right model per stage automatically. Override with
two environment variables:

```bash
MISTRAL_MODEL_REASONING="mistral-large-latest"   # extractor, openacc
MISTRAL_MODEL_CODE="codestral-latest"            # cython_wrapper
```

The legacy `MISTRAL_MODEL` variable still works as a single-model
override for both roles, so older `.env` files keep working unchanged.

### Cost comparison

For a typical kernel (4 LLM calls), against the November 2024 Mistral
tariffs:

| Setup                                              | Wall-clock | Cost   |
| -------------------------------------------------- | ---------- | ------ |
| All-Mistral-Large (legacy)                         | ~2 min     | ~0.06 USD |
| Mistral-Large reasoning + Codestral code-gen (default) | ~90 s      | ~0.03 USD |
| All-Codestral (cheapest)                           | ~75 s      | ~0.02 USD |

`Codestral` reasoning is good enough on simple kernels but degrades
visibly on monolithic codes with deep `COMMON` / `SAVE` patterns — keep
Mistral-Large on the reasoning stages unless you know your kernels are
clean.

## Path 3 — Agent as an MCP provider for Le Chat / Mistral Agents API

The agent ships an MCP server (`run-mcp`) that publishes the
transformation tools (`translate_kernel_gpu`, `translate_kernel`,
`profile_kernels`, `ask_agent`) over HTTP/SSE on port 8000. **Anything
that speaks MCP can drive it** — Claude Desktop, Cursor, VS Code agents,
Mistral Le Chat, the Mistral Agents API.

### Start the server

```bash
uv sync --extra mcp     # FastMCP + the [gpu] transformation stack
uv run run-mcp          # listens on http://0.0.0.0:8000/sse
```

Set `API_KEY` to require a bearer token on every request:

```bash
API_KEY=$(openssl rand -hex 32) uv run run-mcp
```

### Use from Le Chat (connector — beta)

Le Chat's connector directory is in beta. The mechanics:

1. **Public endpoint** — Le Chat needs to reach your MCP server. Locally,
   tunnel with `cloudflared` / `ngrok`; in production, deploy on an
   OpenStack tenant or on-prem cluster — see the
   [deployment roadmap](https://github.com/maurinl26/fortranspire#-roadmap).
2. **Manifest** — submit the connector descriptor. A draft is shipped in
   [`integration/le-chat-connector.json`](https://github.com/maurinl26/fortranspire/blob/main/integration/le-chat-connector.json),
   ready to send when the directory opens to general submissions.
3. **Auth** — Le Chat passes a bearer token; the agent's MCP server checks
   it against `API_KEY`.

### Use from the Mistral Agents API (now)

The Agents API supports MCP servers natively. Minimal Python client:

```python
from mistralai import Mistral
client = Mistral(api_key=...)
agent = client.beta.agents.create(
    name="fortran-gpu-port",
    model="mistral-large-latest",
    instructions="You port Fortran kernels to GPU. Always call translate_kernel_gpu.",
    tools=[{
        "type": "mcp",
        "server": {
            "url": "https://<your-public-mcp-host>/sse",
            "auth": {"type": "bearer", "token": "<API_KEY>"},
        },
    }],
)
```

A runnable smoke-test that exercises this pattern lives at
[`examples/mistral_agents_api_smoke_test.py`](https://github.com/maurinl26/fortranspire/blob/main/examples/mistral_agents_api_smoke_test.py).
Run it with your own Mistral API key — it consumes tokens.

## Path 4 — Submission to the Le Chat connector directory

Goal: get the agent listed as a public connector in Le Chat so any
Mistral Enterprise customer can plug it in with one click.

**Status:** the directory is in beta; an HPC / scientific vertical does
not yet exist, so `fortranspire` is a credible first-mover. The
submission package is prepared in
[`integration/le-chat-connector.json`](https://github.com/maurinl26/fortranspire/blob/main/integration/le-chat-connector.json)
and documented in [Le Chat connector](le-chat-connector).

**Prerequisites before submission:**

- A stable public MCP endpoint (HTTPS, valid certificate, monitored).
  Today this means deploying on an OpenStack tenant or an on-prem
  cluster — see the
  [deployment roadmap](https://github.com/maurinl26/fortranspire#-roadmap).
- A privacy notice (the connector receives user-provided Fortran source).
- A demo video / screenshots (a typical kernel transformation run).

When those are in place, submit through the directory portal (currently
invite-only; Mistral partner managers handle the onboarding).

## Cheat sheet

| Goal                                            | Command                                                         |
| ----------------------------------------------- | --------------------------------------------------------------- |
| Validate your API key                           | `curl -H "Authorization: Bearer $MISTRAL_API_KEY" https://api.mistral.ai/v1/models` |
| Port a kernel (default models)                  | `uv run agent-gpu kernel.f90`                                   |
| Use Codestral everywhere (cheapest)             | `MISTRAL_MODEL=codestral-latest uv run agent-gpu kernel.f90`    |
| Run the MCP server with auth                    | `API_KEY=… uv run run-mcp`                                       |
| Quick API smoke-test                            | `uv run python examples/mistral_agents_api_smoke_test.py --key $MISTRAL_API_KEY` |
| Build the Apptainer analyze image (no LLM cost) | `apptainer build analyze.sif Apptainer.analyze`                 |
