# LLM endpoints

`fortranspire` calls any **OpenAI-compatible chat endpoint** — it never
talks to a hyperscaler. The choice of endpoint is governed by your
sovereignty requirements.

## A — Mistral La Plateforme (EU-hosted, operated by Mistral AI)

```bash
MISTRAL_ENDPOINT="https://api.mistral.ai/v1"
MISTRAL_API_KEY="<key-from-console.mistral.ai>"
MISTRAL_MODEL="mistral-large-latest"   # or codestral-latest, mistral-nemo, …
```

Create the key at <https://console.mistral.ai/> under *API Keys*. Billing
is per-token and the infrastructure stays in Europe.

## B — Self-hosted (full sovereignty)

For an internal cluster (Pangea, GENCI, a private datacenter), expose any
Mistral model via [vLLM](https://github.com/vllm-project/vllm),
[TGI](https://github.com/huggingface/text-generation-inference) or
[Ollama](https://ollama.com/) — all three serve an OpenAI-compatible API.

```bash
# vLLM example
MISTRAL_ENDPOINT="http://vllm-host:8000/v1"
MISTRAL_API_KEY="any-non-empty-string"
MISTRAL_MODEL="mistralai/Mistral-Large-Instruct-2411"
```

The agent does not care which model serves the endpoint as long as it
accepts the OpenAI `/chat/completions` schema and returns deterministic
output at `temperature=0`.

## C — Other compatible backends

- **EU OpenAI-compatible gateways on OpenStack tenants** — any
  OpenAI-compatible endpoint hosted on an EU sovereign OpenStack
  infrastructure accepts the same configuration as Mistral La Plateforme.
- **OpenAI / Azure OpenAI** — works but defeats the sovereignty story and
  is therefore not the default. If you must, set `MISTRAL_ENDPOINT` to the
  appropriate URL and pick a chat-completions model.
- **Ollama** for laptop dev — convenient for iterating on prompts without
  network round-trips, but model quality at small parameter counts is too
  low for production pipelines.

## Quality vs. cost tradeoff

The pipeline issues four calls per kernel. Mistral-Large produces the most
reliable extractor and OpenACC outputs in our internal benchmarks; Codestral
and Mistral-Nemo are cheaper and adequate on simple kernels but more often
require a manual fix-up step on monolithic codes with deep `COMMON` /
`SAVE` patterns.
