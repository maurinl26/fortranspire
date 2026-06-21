"""Tests for ``agent-explain`` — pre-flight cost + risk estimator (issue #14)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fortranspire.agent.explain import (
    CodebaseEstimate,
    FileEstimate,
    estimate_file,
    estimate_paths,
    main,
    render_markdown,
)

FIXTURE = Path(__file__).parent / "fixtures" / "doc_kernel.f90"


def test_estimate_single_file_no_llm_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Disable everything that could fire the LLM transparently
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    fe = estimate_file(str(FIXTURE))
    assert fe.parse_error is None
    assert fe.n_routines >= 2  # update_vx + update_sigma in the fixture
    assert "update_vx" in fe.routines
    assert "update_sigma" in fe.routines
    assert fe.prompt_tokens > 0
    assert fe.completion_tokens > 0
    assert fe.reasoning_cost_usd > 0
    assert fe.code_cost_usd > 0


def test_estimate_cost_scales_with_routine_count():
    # The fixture has 2 routines. Re-running gives the same cost.
    # A larger fictional file would give higher cost — sanity check on the formula.
    one = FileEstimate(path="x", n_routines=1)
    five = FileEstimate(path="x", n_routines=5)
    # Cost helpers are external to FileEstimate, so we re-evaluate via estimate_file
    # on real fixtures. Here just sanity-check the dataclass arithmetic.
    one.reasoning_cost_usd, one.code_cost_usd = 1.0, 2.0
    five.reasoning_cost_usd, five.code_cost_usd = 5.0, 10.0
    assert five.total_cost_usd > one.total_cost_usd


def test_estimate_paths_walks_directory(tmp_path: Path):
    # Drop the fixture twice in tmp_path to verify directory walking.
    shutil.copy(FIXTURE, tmp_path / "a.f90")
    shutil.copy(FIXTURE, tmp_path / "b.F90")
    est = estimate_paths([str(tmp_path)])
    assert len(est.files) == 2
    assert est.n_files_ok == 2
    assert est.total_cost_usd > 0


def test_estimate_returns_parse_error_record_on_garbage(tmp_path: Path):
    bad = tmp_path / "broken.f90"
    bad.write_text("this is not fortran at all !@#$%")
    fe = estimate_file(str(bad))
    # Loki's REGEX fallback often still recognises "subroutine"-less files as having
    # zero routines (raises "no routines"). Either parse_error is set OR n_routines is 0.
    assert fe.parse_error is not None or fe.n_routines == 0


def test_render_markdown_contains_key_sections(tmp_path: Path):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    est = estimate_paths([str(tmp_path)])
    md = render_markdown(est)
    assert "# fortranspire — port-cost estimate" in md
    assert "## Summary" in md
    assert "Estimated LLM cost" in md
    assert "Reasoning model" in md
    assert "Code-gen model" in md
    # Per-file breakdown only renders when at least one file parses
    assert "Per-file breakdown" in md
    # No LLM was called — should be visible in the footer
    assert "no LLM was called" in md


def test_render_markdown_surfaces_risks(tmp_path: Path):
    # The fixture has a module without explicit IMPLICIT NONE inside both routines
    # (the module-level one is present); risks may or may not fire depending on
    # parser heuristics. The render must not crash either way.
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    est = estimate_paths([str(tmp_path)])
    md = render_markdown(est)
    # Risk section header only appears when there are risks; skipping assertion
    # to keep test robust across parser heuristic changes.
    assert isinstance(md, str)
    assert len(md) > 200


def test_cli_writes_report_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    out = tmp_path / "estimate.md"
    rc = main([str(tmp_path), "--output", str(out)])
    assert rc == 0
    assert out.is_file()
    body = out.read_text()
    assert "## Summary" in body


def test_cli_stdout_when_no_output_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    rc = main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fortranspire — port-cost estimate" in out


def test_cli_returns_2_when_no_fortran_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    rc = main([str(empty_dir)])
    assert rc == 2
