"""JSONL tracer — one record per LLM call.

Designed to never crash the pipeline:

- Disabled by default in CI / analyze-only when no LLM is fired.
- Tolerates missing write permission, full disk, missing fields.
- Output is append-only JSONL so it's safe to tail / grep / pipe to
  jq / load into a DataFrame.

Per-tenant accounting via ``FORTRANSPIRE_TENANT_ID`` env var. The same
trace file can carry multiple tenants in a hosted deployment; downstream
billing splits by the ``tenant_id`` field.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fortranspire.observability.pricing import estimate_cost_usd

_TRACE_ENABLED_DEFAULT = "1"
_TRACE_PATH_DEFAULT = "output/traces.jsonl"
_TENANT_DEFAULT = "default"


@dataclass
class Span:
    """One LLM call (or in-progress LLM call)."""

    node: str
    model: str | None = None
    tenant_id: str = _TENANT_DEFAULT
    start_ts: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def record_tokens(self, prompt: int, completion: int) -> None:
        """Accumulate token counts (callbacks may fire multiple times)."""
        self.prompt_tokens += max(0, int(prompt or 0))
        self.completion_tokens += max(0, int(completion or 0))

    def record_error(self, exc: BaseException) -> None:
        """Capture exception so the JSONL record carries the failure reason."""
        self.error = f"{type(exc).__name__}: {exc}"

    def annotate(self, **kwargs: Any) -> None:
        """Attach free-form keys to the ``extra`` field of the record."""
        self.extra.update(kwargs)

    def _finalize(self) -> None:
        self.duration_ms = (time.time() - self.start_ts) * 1000.0
        self.cost_usd = estimate_cost_usd(
            self.model, self.prompt_tokens, self.completion_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        record = {
            "ts":                self.start_ts,
            "tenant_id":         self.tenant_id,
            "node":              self.node,
            "model":             self.model,
            "duration_ms":       round(self.duration_ms, 2),
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "error":             self.error,
        }
        if self.extra:
            record["extra"] = self.extra
        return record


class _NullSpan(Span):
    """No-op span returned when tracing is disabled — keeps callers branchless."""

    def record_tokens(self, prompt: int, completion: int) -> None:  # noqa: D401
        return

    def record_error(self, exc: BaseException) -> None:
        return

    def annotate(self, **kwargs: Any) -> None:
        return


class Tracer:
    """One Tracer per process. Reads env vars at instantiation."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        tenant_id: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.output_path = Path(
            output_path
            or os.getenv("FORTRANSPIRE_TRACE_PATH", _TRACE_PATH_DEFAULT)
        )
        self.tenant_id = tenant_id or os.getenv("FORTRANSPIRE_TENANT_ID", _TENANT_DEFAULT)
        if enabled is None:
            enabled = os.getenv("FORTRANSPIRE_TRACE", _TRACE_ENABLED_DEFAULT) != "0"
        self.enabled = enabled

    @contextmanager
    def span(self, node: str, model: str | None = None) -> Iterator[Span]:
        """Open a span; auto-writes a JSONL record on exit (even on failure)."""
        if not self.enabled:
            yield _NullSpan(node=node, model=model)
            return

        span = Span(node=node, model=model, tenant_id=self.tenant_id)
        try:
            yield span
        except Exception as exc:
            span.record_error(exc)
            raise
        finally:
            span._finalize()
            self._write_safely(span.to_dict())
            self._maybe_emit_otel(span.to_dict())

    def _write_safely(self, record: dict[str, Any]) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Telemetry never crashes the pipeline. Surface once on stderr.
            print(f"fortranspire.observability: cannot write trace ({exc})",
                  file=sys.stderr)

    def _maybe_emit_otel(self, record: dict[str, Any]) -> None:
        endpoint = os.getenv("FORTRANSPIRE_OTEL_ENDPOINT")
        if not endpoint:
            return
        try:
            from fortranspire.observability.exporters.otel import emit
            emit(endpoint, record)
        except Exception:
            # Same rule: observability MUST NOT throw.
            pass


# Shared module-level instance. Tests can swap in their own via
# fortranspire.observability.tracer.tracer = Tracer(...).
tracer = Tracer()
