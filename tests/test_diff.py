"""Tests for ``fortranspire diff`` — semantic before/after viewer (#19)."""
from __future__ import annotations

from pathlib import Path

import pytest

from fortranspire.agent.diff import (
    classify_line,
    compute_diff,
    main,
    render_html,
    render_text,
)


# ── classify_line ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,expected", [
    ("  !$acc parallel loop collapse(2)", "pragma"),
    ("!$omp target", "pragma"),
    ("  PURE subroutine foo(x)", "purity"),
    ("ELEMENTAL function bar(y)", "purity"),
    ("  real(dp) :: a", "type"),
    ("integer(kind=4) :: n", "type"),
    ("  COMMON /grid/ dx, dy", "common"),
    ("  real, save :: psi = 0.0", "save"),
    ("subroutine update_vx(vx, sx, nx)", "refactor"),
    ("integer, intent(in) :: nx", "refactor"),
    ("  vx(i, j) = vx(i, j) + dt * a", "other"),
])
def test_classify_line(line: str, expected: str):
    assert classify_line(line) == expected


# ── compute_diff ────────────────────────────────────────────────────────────

def test_compute_diff_picks_up_added_pragma():
    before = "subroutine k\n  do i=1,n\n  enddo\nend subroutine"
    after  = "subroutine k\n  !$acc parallel loop\n  do i=1,n\n  enddo\nend subroutine"
    report = compute_diff(before, after, a_path="b.f90", b_path="a.f90")
    pragma_adds = [e for e in report.entries
                   if e.kind == "+" and e.category == "pragma"]
    assert len(pragma_adds) == 1
    assert "!$acc" in pragma_adds[0].line


def test_compute_diff_picks_up_removed_pure():
    before = "PURE subroutine k(x)\nend subroutine"
    after  = "subroutine k(x)\nend subroutine"
    report = compute_diff(before, after, a_path="b.f90", b_path="a.f90")
    # Both removal and addition appear (purity vs refactor) — we just
    # check at least one removal is tagged purity.
    purity_dels = [e for e in report.entries
                   if e.kind == "-" and e.category == "purity"]
    assert len(purity_dels) >= 1


def test_compute_diff_counts_summary():
    before = "real :: a\n  COMMON /g/ x\n  do i=1,n\n  enddo"
    after  = "real(dp) :: a\n  !$acc parallel loop\n  do i=1,n\n  enddo"
    report = compute_diff(before, after, a_path="b.f90", b_path="a.f90")
    assert report.counts["type"] >= 1
    assert report.counts["common"] >= 1
    assert report.counts["pragma"] >= 1
    assert report.total_changes == sum(report.counts.values())


def test_compute_diff_zero_changes_when_identical():
    src = "subroutine k\n  return\nend subroutine\n"
    report = compute_diff(src, src, a_path="b.f90", b_path="a.f90")
    assert report.total_changes == 0
    assert sum(report.counts.values()) == 0


# ── render_text ─────────────────────────────────────────────────────────────

def test_render_text_includes_header_and_summary():
    before = "real :: a"
    after  = "real(dp) :: a"
    report = compute_diff(before, after, a_path="x.f90", b_path="y.f90")
    out = render_text(report, use_color=False)
    assert "--- x.f90" in out
    assert "+++ y.f90" in out
    assert "1 changed line(s)" in out or "changed line(s)" in out
    assert "[type]" in out


def test_render_text_collapses_unchanged_runs():
    before = "\n".join(["unchanged"] * 10 + ["old"])
    after  = "\n".join(["unchanged"] * 10 + ["new"])
    out = render_text(compute_diff(before, after, a_path="a", b_path="b"),
                      use_color=False)
    assert "unchanged line(s)" in out


def test_render_text_no_color_omits_ansi_codes():
    before = "old\n"
    after  = "new\n"
    out = render_text(compute_diff(before, after, a_path="a", b_path="b"),
                      use_color=False)
    assert "\033[" not in out


# ── render_html ─────────────────────────────────────────────────────────────

def test_render_html_self_contained():
    before = "real :: a\n"
    after  = "real(dp) :: a\n"
    html_out = render_html(compute_diff(before, after, a_path="a.f90", b_path="b.f90"))
    assert html_out.startswith("<!DOCTYPE html>")
    assert "<style>" in html_out
    # No CDN / external assets
    assert "https://" not in html_out
    assert "<table>" in html_out
    assert "type" in html_out  # tag name on at least one row


def test_render_html_escapes_special_chars():
    before = "a < b\n"
    after  = "a > b\n"
    html_out = render_html(compute_diff(before, after, a_path="a", b_path="b"))
    assert "a &lt; b" in html_out
    assert "a &gt; b" in html_out
    assert "a < b" not in html_out  # raw chars must NOT leak into output


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_text_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    a = tmp_path / "a.f90"
    b = tmp_path / "b.f90"
    a.write_text("PURE subroutine k(x)\nend subroutine\n")
    b.write_text("subroutine k(x)\n  !$acc parallel\nend subroutine\n")
    rc = main([str(a), str(b), "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[purity]" in out
    assert "[pragma]" in out


def test_cli_html_to_file(tmp_path: Path):
    a = tmp_path / "a.f90"
    b = tmp_path / "b.f90"
    a.write_text("real :: a\n")
    b.write_text("real(dp) :: a\n")
    out = tmp_path / "diff.html"
    rc = main([str(a), str(b), "--html", "-o", str(out)])
    assert rc == 0
    body = out.read_text()
    assert body.startswith("<!DOCTYPE html>")
    assert "<table>" in body


def test_cli_missing_file_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rc = main([str(tmp_path / "missing.f90"), str(tmp_path / "also-missing.f90")])
    assert rc == 2
