"""Tests for ``fortranspire graph`` — module call-graph report (issue #15)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from fortranspire.agent.call_graph import (
    FileGraph,
    RoutineNode,
    _mermaid_id,
    extract_graphs,
    main,
    render_mermaid,
    render_report,
)

FIXTURE = Path(__file__).parent / "fixtures" / "doc_kernel.f90"


# ── Mermaid id sanitization ─────────────────────────────────────────────────

def test_mermaid_id_passes_through_clean_names():
    assert _mermaid_id("update_vx") == "update_vx"
    assert _mermaid_id("Compute123") == "Compute123"


def test_mermaid_id_sanitizes_special_chars():
    assert _mermaid_id("foo bar") == "foo_bar"
    assert _mermaid_id("a-b.c") == "a_b_c"


def test_mermaid_id_prefixes_leading_digit():
    # Mermaid identifiers cannot start with a digit.
    out = _mermaid_id("9routine")
    assert out[0].isalpha()
    assert "9routine" in out


# ── Loki extraction on the seismic fixture ──────────────────────────────────

def test_extract_graphs_finds_module_routines():
    graphs = extract_graphs([str(FIXTURE)])
    assert len(graphs) == 1
    g = graphs[0]
    assert g.parse_error is None
    names = {r.name for r in g.routines}
    assert "update_vx" in names
    assert "update_sigma" in names


def test_extract_graphs_walks_directory(tmp_path: Path):
    shutil.copy(FIXTURE, tmp_path / "a.f90")
    shutil.copy(FIXTURE, tmp_path / "b.F90")
    graphs = extract_graphs([str(tmp_path)])
    assert len(graphs) == 2
    assert all(g.parse_error is None for g in graphs)


def test_extract_graphs_returns_error_record_on_garbage(tmp_path: Path):
    bad = tmp_path / "broken.f90"
    bad.write_text("not fortran @#$%")
    graphs = extract_graphs([str(bad)])
    assert len(graphs) == 1
    # Either flagged as parse_error, or simply zero routines — both acceptable
    assert graphs[0].parse_error is not None or graphs[0].routines == []


# ── Mermaid rendering ───────────────────────────────────────────────────────

def test_render_mermaid_emits_flowchart_block():
    g = FileGraph(file="x.f90", routines=[
        RoutineNode(name="caller",  file="x.f90", calls=["callee_a", "callee_b"]),
        RoutineNode(name="callee_a", file="x.f90", calls=[]),
    ])
    md = render_mermaid(g)
    assert md.startswith("```mermaid")
    assert md.endswith("```")
    assert "flowchart LR" in md
    assert "caller --> callee_a" in md
    assert "caller --> callee_b" in md


def test_render_mermaid_marks_external_callees():
    g = FileGraph(file="x.f90", routines=[
        RoutineNode(name="local",  file="x.f90", calls=["mpi_init", "mpi_send"]),
    ])
    md = render_mermaid(g)
    # External callees get the dashed class
    assert "external" in md
    assert "stroke-dasharray" in md


def test_render_mermaid_handles_empty_graph():
    g = FileGraph(file="x.f90", routines=[])
    md = render_mermaid(g)
    assert "(no routines)" in md


def test_render_mermaid_handles_parse_error():
    g = FileGraph(file="x.f90", parse_error="boom")
    md = render_mermaid(g)
    assert "boom" in md


# ── Full report ────────────────────────────────────────────────────────────

def test_render_report_lists_each_file_with_routines(tmp_path: Path):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    graphs = extract_graphs([str(tmp_path)])
    report = render_report(graphs)
    assert "# fortranspire — module call-graph report" in report
    assert "Scanned **1 file(s)**" in report
    assert "update_vx" in report
    assert "update_sigma" in report
    assert "```mermaid" in report


def test_render_report_without_narrate_skips_paragraph(tmp_path: Path):
    """`narrate=False` (the default) must never hit the LLM stack."""
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    graphs = extract_graphs([str(tmp_path)])
    report = render_report(graphs, narrate=False)
    # No paragraph signature words — narrate_graph isn't called.
    # We just check the file structure is consistent.
    assert report.count("## `") == 1


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_writes_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    rc = main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fortranspire — module call-graph report" in out
    assert "```mermaid" in out


def test_cli_writes_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    target = tmp_path / "graph.md"
    rc = main([str(tmp_path), "-o", str(target)])
    assert rc == 0
    assert target.is_file()
    assert "```mermaid" in target.read_text()


def test_cli_returns_2_when_no_files(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main([str(empty)])
    assert rc == 2
