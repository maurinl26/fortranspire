"""Tests for ``fortranspire bench`` — pipeline output metrics + regression (#17)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fortranspire.agent.bench import (
    Metrics,
    _count_acc_pragmas,
    collect_metrics,
    compare,
    main,
    render_text,
)


def _populate(root: Path, *, n_routines: int = 2, n_pragmas: int = 3) -> None:
    """Build a synthetic Phase-1 output tree."""
    (root / "fortran_gpu").mkdir(parents=True, exist_ok=True)
    body = ""
    for i in range(n_routines):
        body += f"subroutine k{i}\n"
        for _ in range(n_pragmas if i == 0 else 0):
            body += "  !$acc parallel loop\n"
        body += "  do j=1,n\n  enddo\n"
        body += f"end subroutine k{i}\n\n"
    (root / "fortran_gpu" / "module_kernels_gpu.f90").write_text(body)

    (root / "cython").mkdir(parents=True, exist_ok=True)
    (root / "cython" / "k.pyx").write_text("# cython wrapper\n" * 30)
    (root / "cython" / "kernel_c.h").write_text("/* header */\n")


# ── _count_acc_pragmas ──────────────────────────────────────────────────────

def test_count_acc_pragmas_basic():
    src = "subroutine f\n  !$acc parallel\n  do i=1,n\n  !$acc end parallel\n  enddo\nend"
    assert _count_acc_pragmas(src) == 2


def test_count_acc_pragmas_ignores_non_acc():
    src = "  !$omp target\n  ! plain comment\n  !$ACC routine seq\n"
    # !$acc is case-insensitive (matches !$ACC too)
    assert _count_acc_pragmas(src) == 1


# ── collect_metrics ─────────────────────────────────────────────────────────

def test_collect_metrics_counts_routines_and_files(tmp_path: Path):
    _populate(tmp_path, n_routines=3)
    m = collect_metrics(tmp_path)
    assert m.n_routines_extracted == 3
    assert m.n_files_generated >= 3   # at least: 1 module + 1 pyx + 1 .h
    assert m.fortran_total_bytes > 0
    assert m.cython_total_bytes > 0


def test_collect_metrics_counts_pragmas(tmp_path: Path):
    _populate(tmp_path, n_pragmas=5)
    m = collect_metrics(tmp_path)
    assert m.n_acc_pragmas == 5


def test_collect_metrics_reads_traces(tmp_path: Path):
    _populate(tmp_path)
    (tmp_path / "traces.jsonl").write_text(
        '{"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.005}\n'
        '{"prompt_tokens": 200, "completion_tokens": 100, "cost_usd": 0.012}\n'
    )
    m = collect_metrics(tmp_path)
    assert m.llm_calls == 2
    assert m.llm_prompt_tokens == 300
    assert m.llm_completion_tokens == 150
    assert abs(m.llm_cost_usd - 0.017) < 1e-9


def test_collect_metrics_gfortran_optional(tmp_path: Path):
    # When gfortran isn't available OR the sources are broken, gfortran_ok
    # is just False — never raises.
    _populate(tmp_path)
    m = collect_metrics(tmp_path)
    # Either gfortran ran and reported OK/NOT-OK, or wasn't available.
    assert isinstance(m.gfortran_ok, bool)
    assert m.gfortran_seconds >= 0


# ── compare ────────────────────────────────────────────────────────────────

def test_compare_detects_routine_drop():
    baseline = {"n_routines_extracted": 10}
    current  = Metrics(output_root="x", n_routines_extracted=8)
    report = compare(baseline, current, tolerance=0.10)
    rows = {r.metric: r for r in report.rows}
    assert rows["n_routines_extracted"].severity == "regression"
    assert rows["n_routines_extracted"].delta_pct == -20.0
    assert report.has_regressions is True


def test_compare_detects_cost_rise():
    baseline = {"llm_cost_usd": 0.05}
    current  = Metrics(output_root="x", llm_cost_usd=0.08)
    report = compare(baseline, current, tolerance=0.10)
    rows = {r.metric: r for r in report.rows}
    assert rows["llm_cost_usd"].severity == "regression"
    assert rows["llm_cost_usd"].delta_pct == 60.0


def test_compare_within_tolerance_is_ok():
    baseline = {"n_routines_extracted": 10, "llm_cost_usd": 0.05}
    current  = Metrics(output_root="x", n_routines_extracted=10, llm_cost_usd=0.052)
    report = compare(baseline, current, tolerance=0.10)
    assert report.has_regressions is False


def test_compare_ignores_missing_or_zero_baseline():
    # zero baseline → no meaningful delta → skipped
    baseline = {"n_acc_pragmas": 0, "n_routines_extracted": 0}
    current  = Metrics(output_root="x", n_acc_pragmas=5, n_routines_extracted=3)
    report = compare(baseline, current, tolerance=0.10)
    assert report.rows == []


def test_compare_skips_bool_fields():
    # gfortran_ok is bool — must NOT crash compare()
    baseline = {"gfortran_ok": True}
    current = Metrics(output_root="x")
    report = compare(baseline, current, tolerance=0.10)
    assert report.rows == []


# ── render_text ─────────────────────────────────────────────────────────────

def test_render_text_includes_metrics_block(tmp_path: Path):
    _populate(tmp_path)
    m = collect_metrics(tmp_path)
    out = render_text(m)
    assert "## Metrics" in out
    assert "n_routines_extracted" in out


def test_render_text_includes_regression_check_when_compared(tmp_path: Path):
    _populate(tmp_path, n_routines=2)
    m = collect_metrics(tmp_path)
    baseline = {"n_routines_extracted": 5}
    report = compare(baseline, m, tolerance=0.10)
    out = render_text(m, report)
    assert "Regression check" in out
    assert "REGRESS" in out


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _populate(tmp_path)
    rc = main([str(tmp_path), "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_routines_extracted"] == 2


def test_cli_emits_text_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _populate(tmp_path)
    rc = main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# fortranspire bench" in out


def test_cli_writes_to_file(tmp_path: Path):
    _populate(tmp_path)
    target = tmp_path / "metrics.json"
    rc = main([str(tmp_path), "--format", "json", "-o", str(target)])
    assert rc == 0
    assert target.is_file()
    json.loads(target.read_text())   # validates


def test_cli_compare_passes_when_baseline_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    _populate(tmp_path, n_routines=2)
    # Save baseline first
    baseline_path = tmp_path / "baseline.json"
    main([str(tmp_path), "--format", "json", "-o", str(baseline_path)])
    capsys.readouterr()
    # Now compare against itself — identical, no regression
    rc = main([str(tmp_path), "--compare", str(baseline_path)])
    assert rc == 0


def test_cli_compare_fails_on_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    # Baseline: 5 routines. Current build: 2 routines. -60 % drop > 10 % tol.
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"n_routines_extracted": 5}))
    _populate(tmp_path, n_routines=2)
    rc = main([str(tmp_path), "--compare", str(baseline_path)])
    assert rc == 1


def test_cli_returns_2_for_non_directory(tmp_path: Path):
    f = tmp_path / "regular.txt"
    f.write_text("x")
    rc = main([str(f)])
    assert rc == 2


def test_cli_returns_2_for_bad_baseline(tmp_path: Path):
    _populate(tmp_path)
    rc = main([str(tmp_path), "--compare", str(tmp_path / "does-not-exist.json")])
    assert rc == 2
