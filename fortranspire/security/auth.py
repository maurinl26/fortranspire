"""Bearer-token auth + per-tenant rate limit + audit log middleware.

Replaces the original single-`API_KEY` middleware that shipped in the
first version of the MCP server. Backwards-compatible: if neither
``FORTRANSPIRE_TOKENS_FILE`` nor ``API_KEY`` is set, the middleware is
simply not installed and the server runs unauthenticated (same as
legacy behavior).

Per-tool scopes are NOT enforced at the HTTP layer (FastMCP routes
every tool through a single SSE endpoint, so the tool name isn't in the
URL). The TokenRegistry still records scopes so a future FastMCP-level
middleware can enforce them; today they're documented as
"defense-in-depth, not load-bearing".
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Iterable

from fortranspire.security.audit import emit

# Default rate limit per token when the registry doesn't override.
_DEFAULT_RATE_PER_HOUR = 100
# Endpoints excluded from auth (health checks etc.).
_OPEN_PATHS: set[str] = {"/health", "/ping", "/metrics"}


@dataclass(frozen=True)
class Token:
    """One entry in the registry — bearer-token-to-tenant binding."""

    secret: str
    tenant_id: str
    scopes: tuple[str, ...] = field(default_factory=tuple)
    rate_limit_per_hour: int = _DEFAULT_RATE_PER_HOUR


@dataclass
class TokenRegistry:
    """In-memory mapping from bearer token secret → Token.

    Loaded from ``FORTRANSPIRE_TOKENS_FILE`` (JSON), with backward
    compatibility for a single ``API_KEY`` env var.
    """

    tokens: dict[str, Token] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "TokenRegistry":
        registry = cls()
        path = os.getenv("FORTRANSPIRE_TOKENS_FILE")
        if path:
            registry.load_file(path)
        legacy = os.getenv("API_KEY")
        if legacy and legacy not in registry.tokens:
            # Legacy single-token deployment — give it the default tenant.
            registry.tokens[legacy] = Token(
                secret=legacy,
                tenant_id=os.getenv("FORTRANSPIRE_TENANT_ID", "default"),
                scopes=(),  # full access
                rate_limit_per_hour=_DEFAULT_RATE_PER_HOUR,
            )
        return registry

    def load_file(self, path: str) -> None:
        """Replace the registry from a JSON file.

        Expected shape:
            {
                "<bearer-secret>": {
                    "tenant_id": "client-a",
                    "scopes": ["translate_kernel_gpu", "ask_agent"],
                    "rate_limit_per_hour": 100
                },
                ...
            }
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        new_tokens: dict[str, Token] = {}
        for secret, attrs in data.items():
            if not isinstance(attrs, dict):
                continue
            new_tokens[secret] = Token(
                secret=secret,
                tenant_id=str(attrs.get("tenant_id", "default")),
                scopes=tuple(attrs.get("scopes", [])),
                rate_limit_per_hour=int(
                    attrs.get("rate_limit_per_hour", _DEFAULT_RATE_PER_HOUR)
                ),
            )
        self.tokens = new_tokens

    def lookup(self, secret: str | None) -> Token | None:
        if not secret:
            return None
        return self.tokens.get(secret)

    def __len__(self) -> int:
        return len(self.tokens)


class RateLimiter:
    """In-memory sliding-window rate limiter — one bucket per tenant.

    Keeps the last hour of timestamps per tenant in a `deque` and accepts
    the request when the bucket is under the per-tenant cap. Thread-safe
    via an `RLock` so it composes with Starlette's async runtime
    (Starlette serializes per-event-loop, but this stays correct under
    threaded test runners).
    """

    def __init__(self, window_seconds: float = 3600.0) -> None:
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._lock = RLock()

    def allow(self, tenant_id: str, cap: int, now: float | None = None) -> bool:
        """Try to record one call. Returns False when the cap is hit."""
        if cap <= 0:
            return True  # No cap configured
        now = now if now is not None else time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(tenant_id, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= cap:
                return False
            bucket.append(now)
            return True

    def usage(self, tenant_id: str) -> int:
        """Number of recent calls — useful for `/metrics`."""
        with self._lock:
            return len(self._buckets.get(tenant_id, ()))


# ── Starlette middleware ────────────────────────────────────────────────────


def build_middleware(registry: TokenRegistry, limiter: RateLimiter | None = None):
    """Return a ready-to-mount Starlette middleware bound to the registry.

    Lazy-imports starlette so this module loads on the analyze-only image.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    rate_limiter = limiter or RateLimiter()

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            start = time.time()
            path = request.url.path
            method = request.method

            if path in _OPEN_PATHS:
                response = await call_next(request)
                return response

            auth = request.headers.get("Authorization", "")
            secret = auth.removeprefix("Bearer ").strip() if auth.lower().startswith("bearer ") else None
            token = registry.lookup(secret)

            if token is None:
                duration_ms = (time.time() - start) * 1000
                emit(tenant_id=None, path=path, method=method, status=401,
                     duration_ms=duration_ms, outcome="unauthorized",
                     reason="invalid_or_missing_token")
                return JSONResponse({"detail": "Unauthorized."}, status_code=401)

            if not rate_limiter.allow(token.tenant_id, token.rate_limit_per_hour):
                duration_ms = (time.time() - start) * 1000
                emit(tenant_id=token.tenant_id, path=path, method=method, status=429,
                     duration_ms=duration_ms, outcome="rate_limited",
                     reason=f"cap={token.rate_limit_per_hour}/h")
                return JSONResponse(
                    {"detail": "Rate limit exceeded."},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )

            # Attach tenant to the request for downstream handlers.
            request.scope["tenant_id"] = token.tenant_id
            request.scope["token_scopes"] = token.scopes

            try:
                response = await call_next(request)
                duration_ms = (time.time() - start) * 1000
                emit(tenant_id=token.tenant_id, path=path, method=method,
                     status=response.status_code, duration_ms=duration_ms,
                     outcome="ok")
                return response
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000
                emit(tenant_id=token.tenant_id, path=path, method=method, status=500,
                     duration_ms=duration_ms, outcome="error",
                     reason=f"{type(exc).__name__}: {exc}")
                raise

    return AuthMiddleware


# Convenience alias for code that wants to import the class directly without
# pre-loading starlette (callers must instantiate the registry first).
def AuthMiddleware(*args, **kwargs):  # type: ignore[no-redef]
    raise NotImplementedError(
        "Use build_middleware(registry, limiter=...) to obtain the middleware class. "
        "Direct construction requires starlette + a bound TokenRegistry."
    )
