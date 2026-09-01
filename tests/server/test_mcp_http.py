"""HTTP-transport regression tests for the MCP server.

These cover two failures that shipped silently in 0.2.1 and blocked every
hosted-deployment issue (#43 European Weather Cloud, #51 Le Chat connector):

1. ``_install_auth`` attached the bearer middleware through ``mcp._app``,
   a private attribute FastMCP 3.x removed. It printed an error and then
   started the server **anyway** — so ``API_KEY=... fortranspire mcp``
   served every tool, including the LLM ones, to unauthenticated clients.
2. ``/health`` was declared in ``integration/le-chat-connector.json`` and
   allow-listed in the auth middleware, but never registered as a route,
   so it answered 404 and no load balancer could probe the instance.

Both are network-level behaviours: unit-testing the helpers would not have
caught either, so these tests drive a real server over real HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

TOKEN = "test-token-please-ignore"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, token: str | None = None, timeout: float = 5.0):
    """Return ``(status, body)``, mapping an HTTP error onto its status."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # An SSE stream never ends; read only the status for those.
            body = b"" if "event-stream" in resp.headers.get("content-type", "") else resp.read()
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except socket.timeout:
        # Reading an open SSE stream timed out => the request was accepted.
        return 200, b""


@pytest.fixture(scope="module")
def authed_server():
    """Spawn `fortranspire mcp` (SSE) with a bearer token configured."""
    port = _free_port()
    env = {
        **os.environ,
        "API_KEY": TOKEN,
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": str(port),
        "FASTMCP_SHOW_SERVER_BANNER": "0",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "fortranspire.cli", "mcp"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited early:\n{proc.stdout.read() if proc.stdout else ''}")
        try:
            if _get(f"{base}/health")[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("server did not become healthy within 60s")

    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.slow
def test_health_is_public_and_describes_the_instance(authed_server):
    """`/health` must answer 200 without a token — the connector probes it."""
    status, body = _get(f"{authed_server}/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert payload["service"] == "fortranspire"
    from fortranspire.server import _TOOL_NAMES

    assert payload["tools"] == len(_TOOL_NAMES)


@pytest.mark.slow
def test_sse_rejects_missing_and_wrong_tokens(authed_server):
    """The regression that mattered: an authed server must not serve anons."""
    assert _get(f"{authed_server}/sse")[0] == 401
    assert _get(f"{authed_server}/sse", token="wrong-token")[0] == 401


@pytest.mark.slow
def test_sse_accepts_the_configured_token(authed_server):
    assert _get(f"{authed_server}/sse", token=TOKEN)[0] == 200


def test_auth_builder_fails_closed_when_middleware_cannot_be_built(monkeypatch):
    """Tokens configured + unattachable middleware => refuse to start.

    Pre-fix the code printed an error and returned, leaving `main()` to
    start an open server. It must raise instead.
    """
    from fortranspire import server

    monkeypatch.setenv("API_KEY", TOKEN)
    monkeypatch.delenv("FORTRANSPIRE_TOKENS_FILE", raising=False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("starlette unavailable")

    monkeypatch.setattr("fortranspire.security.auth.build_middleware", _boom)
    with pytest.raises(server.AuthNotInstallable):
        server._build_auth_middleware()


def test_no_token_configured_stays_unauthenticated(monkeypatch):
    """Backwards compat: localhost OSS use with no token keeps working."""
    from fortranspire import server

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("FORTRANSPIRE_TOKENS_FILE", raising=False)
    assert server._build_auth_middleware() == []


def test_declared_tool_names_match_the_registered_surface():
    """`_TOOL_NAMES` is what the docs and the Le Chat connector promise."""
    from fortranspire.server import _TOOL_NAMES, mcp

    registered = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    assert registered == sorted(_TOOL_NAMES)


# ── Docs / connector drift guard ──────────────────────────────────────────


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def test_docs_never_advertise_a_tool_the_server_does_not_expose():
    """Guard the drift that broke the README quick-start.

    README.md, the JOSS paper and two integration pages all told users to
    call `fortranspire_explain_port_cost`. The server exposes
    `explain_port_cost` — the prefixed name returned "Unknown tool", so the
    flagship demo in the README did not work.
    """
    import re

    from fortranspire.server import _TOOL_NAMES

    root = _repo_root()
    # The changelog is a historical record: the entry that fixes this drift
    # has to quote the names that were wrong, so it is scanned out.
    changelog = root / "docs" / "changelog.md"
    docs = [root / "README.md", root / "paper" / "paper.md"]
    docs += [p for p in sorted((root / "docs").rglob("*.md")) if p != changelog]

    pattern = re.compile(r"`fortranspire_(" + "|".join(_TOOL_NAMES) + r")`")
    offenders = [
        f"{p.relative_to(root)}: {m.group(0)}"
        for p in docs
        if p.exists()
        for m in pattern.finditer(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "docs reference MCP tools with a `fortranspire_` prefix the server "
        f"does not register: {offenders}"
    )


def test_le_chat_connector_lists_only_real_tools():
    """`integration/le-chat-connector.json` is a published contract (#51)."""
    from fortranspire.server import _TOOL_NAMES

    path = _repo_root() / "integration" / "le-chat-connector.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared = {t["id"] for t in manifest.get("tools", [])}
    assert declared <= set(_TOOL_NAMES), (
        f"connector advertises unknown tools: {sorted(declared - set(_TOOL_NAMES))}"
    )


def test_le_chat_connector_health_path_is_served():
    """The connector's `health_check` must match a route we actually register."""
    path = _repo_root() / "integration" / "le-chat-connector.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared = manifest["transport"]["health_check"]

    from fortranspire.server import mcp

    routes = {r.path for r in mcp._additional_http_routes}
    assert declared in routes, (
        f"connector probes {declared!r} but the server registers {sorted(routes)}"
    )


def test_every_path_taking_tool_uses_the_same_argument_name():
    """One spelling for one concept, and it must be required.

    The surface used to be split — `analyze_kernels`/`build_call_graph`/
    `generate_docs` took ``path`` while `explain_port_cost` and the three
    LLM tools took ``filepath``, for the same concept. An agent reading the
    schema cannot guess which is which, so it mis-calls and burns a turn on
    a validation error.
    """
    import asyncio

    from fortranspire.server import mcp

    for tool in asyncio.run(mcp.list_tools()):
        schema = tool.parameters or {}
        props = schema.get("properties", {})
        if tool.name in {"ask_agent", "agent_status",
                         "domain_geometries", "domain_decomposition"}:
            # ask_agent/agent_status take a query/nothing; the domain tools
            # take a resolution + inline source, not a server path.
            continue
        if tool.name.endswith("_source"):
            # Hosted variants take inline `source`, not a server path —
            # that is the whole reason they exist (issue #51).
            assert "source" in props, f"{tool.name} should take `source`: {list(props)}"
            assert "path" not in props
            continue
        assert "path" in props, f"{tool.name} does not take `path`: {list(props)}"
        assert "filepath" not in props, f"{tool.name} still exposes the legacy `filepath`"
        assert "path" in schema.get("required", []), f"{tool.name}: `path` must be required"


def test_connector_inputs_match_the_server_schema():
    """The Le Chat connector declares the arguments Le Chat will send (#51)."""
    import asyncio

    from fortranspire.server import mcp

    manifest = json.loads(
        (_repo_root() / "integration" / "le-chat-connector.json").read_text(encoding="utf-8")
    )
    live = {t.name: (t.parameters or {}) for t in asyncio.run(mcp.list_tools())}

    for tool in manifest.get("tools", []):
        schema = live[tool["id"]]
        props = set(schema.get("properties", {}))
        for declared in tool.get("inputs", []):
            assert declared["name"] in props, (
                f"connector declares input {declared['name']!r} for {tool['id']}, "
                f"server accepts {sorted(props)}"
            )
