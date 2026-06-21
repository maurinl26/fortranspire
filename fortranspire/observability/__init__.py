"""Telemetry and cost-tracking for fortranspire pipeline runs.

Three thin layers, no heavy external dep:

- :mod:`fortranspire.observability.tracer` — `Tracer` + `Span` writing
  one JSONL record per LLM call to ``output/traces.jsonl`` (override via
  ``FORTRANSPIRE_TRACE_PATH``). Disable with ``FORTRANSPIRE_TRACE=0``.
- :mod:`fortranspire.observability.pricing` — per-model price table
  (Mistral La Plateforme as of 2026-06). Used to compute ``cost_usd``
  on each span.
- :mod:`fortranspire.observability.llm_callback` — a
  ``BaseCallbackHandler`` that captures LangChain's token usage from
  ``response.llm_output["token_usage"]`` even when
  ``with_structured_output()`` swallows the raw message. Lazy-imports
  langchain so the analyze-only image keeps working.

Per-tenant accounting via ``FORTRANSPIRE_TENANT_ID`` env var (defaults
to ``"default"``). Every record carries the tenant id so a single trace
file can serve a multi-tenant deployment.

Optional OpenTelemetry export when ``FORTRANSPIRE_OTEL_ENDPOINT`` is
set — see :mod:`fortranspire.observability.exporters.otel`.
"""
from fortranspire.observability.pricing import estimate_cost_usd
from fortranspire.observability.tracer import Span, Tracer, tracer

__all__ = ["Span", "Tracer", "tracer", "estimate_cost_usd"]
