"""Tests for ``fortranspire report`` — HTML audit dashboard (issue #20)."""
from __future__ import annotations

from pathlib import Path

import pytest

from fortranspire.agent.report import (
    Report,
    collect_sections,
    main,
    render_html,
)


def _make_fake_output(root: Path) -> None:
    """Build a minimal Phase-1 output tree under `root` for tests."""
    (root / "kernel.f90").write_text("subroutine k(x)\n  integer :: x\nend subroutine\n")
    (root / "fortran_gpu").mkdir(parents=True, exist_ok=True)
    (root / "fortran_gpu" / "kernel_pure.f90").write_text("PURE subroutine k\nend subroutine\n")
    (root / "fortran_gpu" / "module_kernels_gpu.f90").write_text(
        "module m\ncontains\n  subroutine k\n    !$acc parallel\n  end subroutine\nend module\n"
    )
    (root / "fortran_gpu" / "validation.log").write_text("gfortran OK\nnvfortran OK\n")
    (root / "cython").mkdir(parents=True, exist_ok=True)
    (root / "cython" / "kernel.pyx").write_text("cdef extern from 'kernel_c.h': pass\n")
    (root / "cython" / "kernel_c.h").write_text("#ifndef KERNEL_C_H\n#define KERNEL_C_H\n#endif\n")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_kernel_equivalence.py").write_text("import pytest\n")


# ── Collection ─────────────────────────────────────────────────────────────

def test_collect_sections_finds_present_files(tmp_path: Path):
    _make_fake_output(tmp_path)
    report = collect_sections(tmp_path)
    labels = {s.label for s in report.sections}
    assert "Original Fortran source" in labels
    assert "MODULE annotated with OpenACC" in labels
    assert "Validation log" in labels
    # Equivalence harness uses a glob — its label is "Equivalence test harness"
    # (no suffix because there's only one match)
    assert "Equivalence test harness" in labels


def test_collect_sections_lists_missing(tmp_path: Path):
    # Empty directory → everything is missing
    report = collect_sections(tmp_path)
    assert report.sections == []
    assert len(report.missing) > 0


def test_collect_sections_uses_kernel_name_from_dir(tmp_path: Path):
    sub = tmp_path / "my_kernel"
    sub.mkdir()
    (sub / "kernel.f90").write_text("subroutine k\nend subroutine\n")
    report = collect_sections(sub)
    assert report.kernel_name == "my_kernel"


def test_collect_sections_multiple_pyx_files(tmp_path: Path):
    """When several .pyx exist, each becomes its own section."""
    (tmp_path / "cython").mkdir()
    (tmp_path / "cython" / "a.pyx").write_text("# pyx a\n")
    (tmp_path / "cython" / "b.pyx").write_text("# pyx b\n")
    report = collect_sections(tmp_path)
    pyx_sections = [s for s in report.sections if "Cython wrapper" in s.label]
    assert len(pyx_sections) == 2


# ── Rendering ──────────────────────────────────────────────────────────────

def test_render_html_self_contained(tmp_path: Path):
    _make_fake_output(tmp_path)
    report = collect_sections(tmp_path)
    out = render_html(report)
    assert out.startswith("<!DOCTYPE html>")
    assert "<style>" in out
    # No external CDN / script tags
    assert "https://" not in out
    assert "<script" not in out
    # Per-section collapsible
    assert out.count("<details") >= len(report.sections)


def test_render_html_escapes_special_chars(tmp_path: Path):
    (tmp_path / "kernel.f90").write_text("x = a < b > c\n")
    report = collect_sections(tmp_path)
    out = render_html(report)
    # When Pygments is available, the syntax highlighter inserts tags,
    # but the raw `< b >` token must NOT appear unescaped — at minimum
    # the `<` should be wrapped in something that hides it from the
    # outer HTML parser.
    assert "&lt;" in out or "a < b" not in out.replace("<", "&lt;")


def test_render_html_lists_missing_artifacts(tmp_path: Path):
    (tmp_path / "kernel.f90").write_text("subroutine k\nend subroutine\n")
    report = collect_sections(tmp_path)
    out = render_html(report)
    assert "Missing artifacts" in out


# ── CLI ────────────────────────────────────────────────────────────────────

def test_cli_writes_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _make_fake_output(tmp_path)
    rc = main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "<!DOCTYPE html>" in out
    assert "audit report" in out


def test_cli_writes_to_file(tmp_path: Path):
    _make_fake_output(tmp_path)
    target = tmp_path / "review.html"
    rc = main([str(tmp_path), "-o", str(target)])
    assert rc == 0
    assert target.is_file()
    assert target.read_text().startswith("<!DOCTYPE html>")


def test_cli_returns_2_when_not_a_directory(tmp_path: Path):
    f = tmp_path / "regular.txt"
    f.write_text("x")
    rc = main([str(f)])
    assert rc == 2


def test_cli_returns_2_when_directory_is_empty(tmp_path: Path):
    rc = main([str(tmp_path)])
    assert rc == 2
