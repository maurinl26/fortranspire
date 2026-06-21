"""MCP-server security primitives.

Three thin layers, no external dependency beyond ``starlette`` (already
pulled in via ``fastmcp``):

- :class:`TokenRegistry` — loads a JSON file mapping bearer tokens to
  a tenant id + per-tenant quota. Hot-reloadable on SIGHUP.
- :class:`RateLimiter` — in-memory sliding-window counter, configurable
  per tenant.
- :class:`AuthMiddleware` — Starlette middleware that does bearer auth,
  rate limiting, tenant assignment, and audit logging in one pass.

Audit log is written via :func:`fortranspire.security.audit.write_event`
so the same JSONL pipeline carries auth events and (later) MCP tool
invocations.

Backward-compatible fallback: if neither ``FORTRANSPIRE_TOKENS_FILE`` nor
``API_KEY`` is set, the middleware is not installed and the server runs
without auth — same as the legacy behavior before this change.
"""
from fortranspire.security.audit import AuditEvent, write_event
from fortranspire.security.auth import (
    AuthMiddleware,
    RateLimiter,
    Token,
    TokenRegistry,
)

__all__ = [
    "AuthMiddleware",
    "RateLimiter",
    "Token",
    "TokenRegistry",
    "AuditEvent",
    "write_event",
]
