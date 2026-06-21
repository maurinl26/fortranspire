"""Integration tests for AuthMiddleware against a Starlette TestClient."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from fortranspire.security.auth import (
    RateLimiter,
    TokenRegistry,
    build_middleware,
)


def _make_app(registry: TokenRegistry, limiter: RateLimiter | None = None) -> Starlette:
    async def hello(request):
        tenant = request.scope.get("tenant_id", "?")
        return PlainTextResponse(f"hello {tenant}")

    async def health(request):
        return PlainTextResponse("ok")

    middleware_cls = build_middleware(registry, limiter)
    app = Starlette(routes=[
        Route("/", hello),
        Route("/health", health),
    ])
    app.add_middleware(middleware_cls)
    return app


def _build_registry(tmp_path: Path, secret: str, *, rate_cap: int = 100) -> TokenRegistry:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({
        secret: {"tenant_id": "client-a", "rate_limit_per_hour": rate_cap}
    }))
    reg = TokenRegistry()
    reg.load_file(str(path))
    return reg


def test_unauthorized_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = _make_app(_build_registry(tmp_path, "good-token"))
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 401


def test_unauthorized_with_wrong_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = _make_app(_build_registry(tmp_path, "good-token"))
    client = TestClient(app)
    r = client.get("/", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_authorized_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = _make_app(_build_registry(tmp_path, "good-token"))
    client = TestClient(app)
    r = client.get("/", headers={"Authorization": "Bearer good-token"})
    assert r.status_code == 200
    assert "hello client-a" in r.text


def test_health_bypasses_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = _make_app(_build_registry(tmp_path, "good-token"))
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_rate_limit_blocks_after_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = _make_app(_build_registry(tmp_path, "tok", rate_cap=3), RateLimiter())
    client = TestClient(app)
    for _ in range(3):
        r = client.get("/", headers={"Authorization": "Bearer tok"})
        assert r.status_code == 200
    blocked = client.get("/", headers={"Authorization": "Bearer tok"})
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After") == "60"


def test_audit_log_records_every_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(audit_path))
    app = _make_app(_build_registry(tmp_path, "tok"))
    client = TestClient(app)
    client.get("/", headers={"Authorization": "Bearer tok"})
    client.get("/")   # unauthorized
    lines = audit_path.read_text().splitlines()
    outcomes = [json.loads(line)["outcome"] for line in lines]
    assert "ok" in outcomes
    assert "unauthorized" in outcomes
