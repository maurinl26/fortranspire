"""Tests for the auth / rate-limit / audit MCP security layer — issue #10."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fortranspire.security.auth import RateLimiter, Token, TokenRegistry


# ── TokenRegistry ───────────────────────────────────────────────────────────

def test_registry_loads_json_file(tmp_path: Path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({
        "abc123": {"tenant_id": "client-a", "scopes": ["translate_kernel_gpu"],
                   "rate_limit_per_hour": 50},
        "xyz789": {"tenant_id": "client-b"},
    }))
    reg = TokenRegistry()
    reg.load_file(str(path))
    assert len(reg) == 2

    tok_a = reg.lookup("abc123")
    assert tok_a is not None
    assert tok_a.tenant_id == "client-a"
    assert tok_a.scopes == ("translate_kernel_gpu",)
    assert tok_a.rate_limit_per_hour == 50

    tok_b = reg.lookup("xyz789")
    assert tok_b is not None
    assert tok_b.tenant_id == "client-b"
    assert tok_b.scopes == ()    # default empty
    assert tok_b.rate_limit_per_hour == 100   # default cap


def test_registry_from_env_legacy_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FORTRANSPIRE_TOKENS_FILE", raising=False)
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("FORTRANSPIRE_TENANT_ID", "the-tenant")
    reg = TokenRegistry.from_env()
    assert len(reg) == 1
    tok = reg.lookup("legacy-key")
    assert tok is not None
    assert tok.tenant_id == "the-tenant"


def test_registry_from_env_empty_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FORTRANSPIRE_TOKENS_FILE", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    reg = TokenRegistry.from_env()
    assert len(reg) == 0  # No tokens → server runs unauthenticated (legacy)


def test_registry_lookup_returns_none_for_unknown():
    reg = TokenRegistry()
    assert reg.lookup("unknown") is None
    assert reg.lookup(None) is None
    assert reg.lookup("") is None


# ── RateLimiter ─────────────────────────────────────────────────────────────

def test_rate_limiter_allows_under_cap():
    rl = RateLimiter()
    for _ in range(5):
        assert rl.allow("tenant-x", cap=10) is True


def test_rate_limiter_blocks_over_cap():
    rl = RateLimiter()
    for _ in range(3):
        rl.allow("tenant-y", cap=3)
    assert rl.allow("tenant-y", cap=3) is False
    assert rl.usage("tenant-y") == 3


def test_rate_limiter_resets_after_window():
    rl = RateLimiter(window_seconds=10.0)
    base_time = 1_000_000_000.0
    for i in range(5):
        rl.allow("tenant-z", cap=5, now=base_time + i)
    # Cap reached
    assert rl.allow("tenant-z", cap=5, now=base_time + 9) is False
    # Slide window past the first event
    assert rl.allow("tenant-z", cap=5, now=base_time + 11) is True


def test_rate_limiter_per_tenant_isolation():
    rl = RateLimiter()
    for _ in range(5):
        rl.allow("a", cap=5)
    # Tenant b should be unaffected.
    for _ in range(5):
        assert rl.allow("b", cap=5) is True


def test_rate_limiter_cap_zero_means_unlimited():
    rl = RateLimiter()
    for _ in range(1000):
        assert rl.allow("tenant", cap=0) is True


# ── Token frozen-dataclass invariants ──────────────────────────────────────

def test_token_is_immutable():
    tok = Token(secret="s", tenant_id="t")
    with pytest.raises((AttributeError, Exception)):
        tok.tenant_id = "other"   # type: ignore[misc]
