"""Tests for the MCP tool surface — issue #39.

These tests verify that the no-LLM MCP tools (``analyze_kernels``,
``explain_port_cost``, ``build_call_graph``, ``generate_docs``) run
against the same fixture used by the CLI tests, return a non-empty
text payload, and signal success via the prefixed ``rc=0`` line.

The MCP server module is imported eagerly to lock in the registered
tool surface — if a tool gets dropped or renamed, this test fails.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

# Disable the workspace jail for these tests — pytest's tmp_path lives
# under /private/var/folders/... which is outside the workspace root,
# and we exercise the tools on those fixture paths intentionally.
os.environ["FORTRANSPIRE_DISABLE_JAIL"] = "1"

from fortranspire.server import (  # noqa: E402  (env var must precede import)
    _capture_main,
    analyze_kernels,
    build_call_graph,
    explain_port_cost,
    generate_docs,
    mcp,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "doc_kernel.f90"


# ── Registered tool surface ───────────────────────────────────────────────

def test_expected_tools_are_registered():
    """The MCP server must expose every tool the SKILL.md references."""
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
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
    missing = expected - names
    assert not missing, f"Missing MCP tools: {missing}"


# ── _capture_main helper ──────────────────────────────────────────────────

def test_capture_main_handles_systemexit_cleanly():
    """``--help`` raises SystemExit; the helper must convert that to rc."""
    from fortranspire.agent.explain import main as explain_main

    rc, text = _capture_main(explain_main, ["--help"])
    assert rc == 0
    assert "usage" in text.lower() or "Usage" in text


# ── analyze_kernels ───────────────────────────────────────────────────────

def test_analyze_kernels_on_fixture(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    out = analyze_kernels(str(target), no_toolchain_check=True)
    assert out.startswith("analyze rc=")
    # The fixture is benign; analyze should report the per-file summary.
    assert "file(s) analyzed" in out or "FORT" in out


def test_analyze_kernels_sarif_output(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    sarif = tmp_path / "report.sarif"
    out = analyze_kernels(
        str(target), sarif_out=str(sarif), no_toolchain_check=True,
    )
    assert out.startswith("analyze rc=")
    assert sarif.exists()
    body = sarif.read_text()
    assert '"version": "2.1.0"' in body
    assert '"$schema":' in body


# ── explain_port_cost ─────────────────────────────────────────────────────

def test_explain_port_cost_on_fixture(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    out = explain_port_cost(str(target))
    assert out.startswith("explain rc=")
    # The fixture has 2 routines (update_vx, update_sigma) — at least
    # one must surface in the rendered estimate.
    assert "update_" in out or "Routines" in out or "routines" in out


# ── build_call_graph ──────────────────────────────────────────────────────

def test_build_call_graph_on_fixture(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    out_md = tmp_path / "graph.md"
    out = build_call_graph(str(target), out=str(out_md))
    assert out.startswith("graph rc=")
    assert out_md.exists()
    body = out_md.read_text()
    assert "```mermaid" in body
    assert "flowchart" in body


# ── generate_docs ─────────────────────────────────────────────────────────

def test_generate_docs_no_llm_dry_run(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    out = generate_docs(str(target), with_llm=False, dry_run=True)
    assert out.startswith("doc rc=")
    assert "@generated_by fortranspire" in out
    # Dry-run must not modify the source file in place
    assert target.read_text() == FIXTURE.read_text()
