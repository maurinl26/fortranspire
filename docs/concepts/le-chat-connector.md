# Le Chat connector (preparation kit)

This page documents the submission package for the
**Le Chat connector directory** (beta). The package is *prepared* but not
yet submitted — submission is gated on having a stable public MCP
endpoint, which depends on the deployment-roadmap items.

> **Status as of 2026-06-21:** Le Chat's connector directory is in beta;
> there is no scientific / HPC vertical yet, so `fortranspire` is a
> credible first-mover. Submission is invite-only — Mistral partner
> managers handle onboarding.

## Submission package

The machine-readable manifest lives at
[`integration/le-chat-connector.json`](https://github.com/maurinl26/fortranspire/blob/main/integration/le-chat-connector.json).
It declares:

- The four MCP tools exposed (`translate_kernel_gpu`, `translate_kernel`,
  `profile_kernels`, `ask_agent`) with their input schemas, expected
  latency, and side effects.
- Transport (`mcp` over `sse`) and auth (`bearer`, mapped to the agent's
  `API_KEY` env var).
- Privacy details (sub-processors, data residency, deletion policy).
- Vendor and licensing metadata.

When the directory schema lands as a stable spec, regenerate the manifest
against it — the current draft is conservative and only uses widely
expected keys.

## Checklist before submission

Mistral expects connector operators to satisfy a baseline of operational
hygiene. The current gap list:

- [ ] **Public HTTPS endpoint** with valid certificate, reachable from
      Le Chat's egress. Today this requires deploying the MCP server to
      Scaleway, OVH, or an on-prem OpenStack tenant — see the deployment
      roadmap in [README](https://github.com/maurinl26/fortranspire#-roadmap).
- [ ] **Health check** at `/health` (already shipped — bypasses bearer
      auth so probes don't need credentials).
- [ ] **Bearer auth enforced** (set `API_KEY` on the server; the
      connector passes Le Chat's token in the `Authorization` header).
- [ ] **Privacy notice published** — link from the directory listing.
- [ ] **Demo video** showing a real kernel transformation
      (~2 minutes, recommended fixture: a small seismic CPML 2-D stencil).
- [ ] **On-call procedure** documented (issues vs. security advisory
      routing).

## Operating recommendations

- Run a dedicated MCP instance per tenant if data isolation is a
  contractual requirement; per-tenant subdomains keep TLS and rate-limit
  configuration simple.
- Configure the server's LLM endpoint to point at the *tenant's* Mistral
  workspace key (env var injection at container start), not a shared
  vendor key — this keeps billing and audit trails clean.
- Monitor the four-call-per-kernel budget; spikes above six calls per
  kernel usually mean the input is hitting a pattern the pipeline doesn't
  cover yet — open an issue with the offending fixture.

## Related

- The agent's full Mistral integration story is in
  [Mistral integration](mistral-integration).
- The OpenStack / Scaleway deployment recipes are tracked in the
  [deployment roadmap](https://github.com/maurinl26/fortranspire#-roadmap).
