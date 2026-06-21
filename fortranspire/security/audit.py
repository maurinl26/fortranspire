"""Signed-by-default audit log — one JSONL record per HTTP request.

Lives alongside the observability tracer but is independent: auth events
must be auditable even when LLM tracing is disabled.

Records are HMAC-signed with ``FORTRANSPIRE_AUDIT_SECRET`` when set, so
post-hoc tampering is detectable. Compliance teams at TotalEnergies /
EDF / CEA-class clients require this for any externally-reachable
endpoint that touches their codebases.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_AUDIT_PATH_DEFAULT = "output/audit.jsonl"


@dataclass
class AuditEvent:
    ts: float
    tenant_id: str | None
    path: str
    method: str
    status: int
    duration_ms: float
    outcome: str          # "ok" | "unauthorized" | "rate_limited" | "error"
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        # Round float fields for stable signatures
        record["duration_ms"] = round(self.duration_ms, 2)
        return record


def _sign(record: dict[str, Any], secret: str) -> str:
    """HMAC-SHA256 over the JSON-canonical record (sans signature field)."""
    payload = {k: v for k, v in record.items() if k != "signature"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def write_event(event: AuditEvent) -> None:
    """Append `event` to the audit log. Never raises — telemetry must not crash auth."""
    path = Path(os.getenv("FORTRANSPIRE_AUDIT_PATH", _AUDIT_PATH_DEFAULT))
    record = event.to_dict()

    secret = os.getenv("FORTRANSPIRE_AUDIT_SECRET")
    if secret:
        record["signature"] = _sign(record, secret)
    else:
        # Drop the default-None signature so the record stays clean when
        # signing isn't configured (avoids `"signature": null` everywhere).
        record.pop("signature", None)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Same rule as the tracer: never crash the pipeline on I/O failure.
        print(f"fortranspire.security.audit: cannot write event ({exc})",
              file=sys.stderr)


def verify_record(record: dict[str, Any], secret: str) -> bool:
    """Re-compute the HMAC and compare. Use for post-hoc tamper detection."""
    expected = _sign(record, secret)
    actual = record.get("signature")
    return bool(actual) and hmac.compare_digest(expected, actual)


def emit(
    tenant_id: str | None,
    path: str,
    method: str,
    status: int,
    duration_ms: float,
    outcome: str,
    reason: str | None = None,
    **extra: Any,
) -> AuditEvent:
    """Convenience constructor + writer in one call."""
    event = AuditEvent(
        ts=time.time(),
        tenant_id=tenant_id,
        path=path,
        method=method,
        status=status,
        duration_ms=duration_ms,
        outcome=outcome,
        reason=reason,
        extra=dict(extra),
    )
    write_event(event)
    return event
