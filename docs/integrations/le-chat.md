# Le Chat connector — hosted MCP for Mistral's agent surface

*Issue [#51](https://github.com/maurinl26/fortranspire/issues/51).*

Le Chat, Mistral's cloud agent surface, can call an MCP server listed in
its connector directory. That reaches every Le Chat user from a menu, on
Mistral models, EU-resident — a strong fit for the sovereignty posture.

Le Chat talks only to a **public HTTPS SSE endpoint with a bearer token**,
so this needs a hosted server. It does **not** need the European Weather
Cloud tenancy (#43): a small EU VPS is enough, and the deploy config below
provisions one.

## What the public connector exposes — and what it deliberately does not

A directory listing is reachable by anyone who adds it in Le Chat. Two
consequences shaped the tool surface:

- **No shared filesystem.** The path-taking tools (`analyze_kernels`,
  `translate_kernel_gpu`, …) assume the client and server see the same
  files. A Le Chat user's Fortran is on their machine, not on the server,
  so a path means nothing. The connector uses **inline-source** tools
  instead — the caller sends the Fortran text.
- **No token spend, no writes, no LLM.** The public surface is limited to
  the deterministic tools. A token-spending tool on a public endpoint runs
  the operator's Mistral key on anyone's request; a writing tool touches
  the host. Those stay off this surface entirely — reachable only over a
  local stdio connection or an authenticated private deployment.

| Tool | What it does | LLM | Writes |
| ---- | ------------ | --- | ------ |
| `explain_source` | Port-cost + risk estimate for pasted Fortran | No | No |
| `analyze_source` | GPU-portability findings (`FORT0xx`) | No | No |
| `build_call_graph_source` | Mermaid call-graph | No | No |

Each takes `source` (the Fortran text) and an optional `filename` used only
to pick the dialect — `.f` for fixed form, `.F90` to trigger cpp
preprocessing.

## Deploy on a small EU VPS

Everything is under [`deploy/le-chat/`](https://github.com/maurinl26/fortranspire/tree/main/deploy/le-chat).
Caddy fronts the MCP container and obtains a Let's Encrypt certificate
automatically, so there are no certificate files to manage.

```bash
# 1. Point an A record at the host:  mcp.example.eu -> <host IP>
# 2. Configure
cd deploy/le-chat
cp .env.example .env
#    FORTRANSPIRE_DOMAIN=mcp.example.eu
#    API_KEY=$(openssl rand -hex 32)
#    FORTRANSPIRE_AUDIT_SECRET=$(openssl rand -hex 32)

# 3. Launch
docker compose --env-file .env up -d

# 4. Verify the public probe
curl https://mcp.example.eu/health      # {"status":"ok","service":"fortranspire",...}
```

The MCP container is only `expose`d on the internal network; Caddy is the
single public door. `API_KEY` makes the SSE endpoint **fail closed** —
every request without a matching bearer token is refused and audited (the
behaviour is verified over real HTTP in `tests/server/test_mcp_http.py`).
`/health` answers without a token, because the connector probes it.

## Register with the directory

[`integration/le-chat-connector.json`](https://github.com/maurinl26/fortranspire/blob/main/integration/le-chat-connector.json)
is the manifest. Before submitting:

1. Set `transport.default_url` to `https://<your-domain>/sse`.
2. Re-check it against the current directory schema — the file itself notes
   the beta schema may have moved.
3. Submit through the Le Chat connector directory.

The manifest already declares the three deterministic tools, the bearer
auth, and the `/health` probe, and it carries a `public_surface_note`
explaining why the token-spending tools are absent.

## Acceptance

- [ ] Public SSE endpoint reachable over TLS + bearer; `/health` returns 200.
- [ ] `le-chat-connector.json` validated against the current schema.
- [ ] Connector submitted, and (if accepted) visible in the directory.
- [ ] A Le Chat user calls `explain_source` on a pasted kernel end-to-end.

## Relationship to the other MCP surfaces

`mistral-vibe` reaches the same server over **local stdio** (no hosting),
covered in [mistral-vibe](mistral-vibe.md) and issue #39. That local path
keeps the full tool set, including the token-spending ones, because the
trust boundary there is the user's own machine. This page is the hosted,
public path, which is why its surface is narrower.
