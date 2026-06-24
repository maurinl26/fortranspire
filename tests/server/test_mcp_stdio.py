"""Stdio transport smoke test for `fortranspire mcp --stdio`.

mistral-vibe and Claude Code Desktop both spawn the MCP server as a
subprocess and speak JSON-RPC over its stdin/stdout. This test exercises
that exact handshake — initialize → notifications/initialized →
tools/list — and asserts the 9 tools surface intact.

Marked ``slow`` because it shells out to the installed ``fortranspire``
console script and waits up to 15s for stdio responses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.slow


EXPECTED_TOOLS = {
    "ask_agent",
    "agent_status",
    "translate_kernel_gpu",
    "translate_kernel",
    "profile_kernels",
    "analyze_kernels",
    "explain_port_cost",
    "build_call_graph",
    "generate_docs",
}


def test_fortranspire_mcp_stdio_lists_all_tools():
    if shutil.which("fortranspire") is None:
        pytest.skip("`fortranspire` console script not on PATH")

    proc = subprocess.Popen(
        ["fortranspire", "mcp", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout

    requests = [
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    for req in requests:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

    responses: list[dict] = []
    deadline = time.monotonic() + 15.0
    try:
        while len(responses) < 2 and time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"non-JSON on stdout: {line!r}", file=sys.stderr)
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert len(responses) >= 2, f"got {len(responses)} responses, expected ≥2"

    tools_result = next(
        (r["result"] for r in responses if r.get("id") == 2 and "result" in r),
        None,
    )
    assert tools_result is not None, "tools/list returned no result"

    tool_names = {t["name"] for t in tools_result["tools"]}
    missing = EXPECTED_TOOLS - tool_names
    assert not missing, f"stdio tool surface missing: {missing}"
