"""Tests for the cpp pass on uppercase-suffix Fortran (issue #73 fieldwork).

Found on the first real file we tried: CMAQ's `rbsolver.F`, 632 lines of
1990s fixed-form F77. Loki returned **zero routines** and the analyzer
reported a parse failure. The file parses perfectly — it had simply never
been preprocessed, which its uppercase suffix has been asking for all
along. 525 of CMAQ's Fortran files carry that suffix and 199 of them hold
live `#ifdef` blocks, so the gap blocked the whole target.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fortranspire.agent.nodes._preprocess import (
    PREPROCESSED_SUFFIXES,
    needs_preprocessing,
    preprocess,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "patterns" / "fixed_form_cpp.F"


class TestSuffixConvention:
    @pytest.mark.parametrize("name", ["a.F", "a.F90", "a.FOR", "a.FPP"])
    def test_uppercase_asks_for_preprocessing(self, name):
        assert needs_preprocessing(name)

    @pytest.mark.parametrize("name", ["a.f", "a.f90", "a.for", "a.txt"])
    def test_lowercase_does_not(self, name):
        """The distinction is the whole convention — it must stay case-sensitive."""
        assert not needs_preprocessing(name)

    def test_the_two_sets_do_not_overlap_case_insensitively(self):
        assert all(s == s.upper() for s in PREPROCESSED_SUFFIXES)


class TestPreprocessing:
    def test_directives_are_removed(self):
        raw = FIXTURE.read_text()
        assert "#ifdef" in raw

        out, note = preprocess(FIXTURE, raw)
        assert note is None
        assert "#ifdef" not in out
        assert "#endif" not in out

    def test_inactive_branch_is_dropped(self):
        """`rbdebug` is not defined, so its WRITE must not survive.

        It matters beyond tidiness: that WRITE is I/O, and I/O in a kernel
        is FORT001, which blocks a GPU port. Leaving it in would make the
        analyzer refuse a portable kernel.
        """
        out, _ = preprocess(FIXTURE, FIXTURE.read_text())
        assert "entering ROS3_STEP" not in out

    def test_line_numbers_are_preserved(self):
        """Findings carry a line; SARIF turns it into a GitHub annotation.

        `cpp -P` deletes inactive branches and shifts everything below, so
        every annotation past the first `#ifdef` would land on the wrong
        statement. Removed lines are blanked instead.
        """
        raw = FIXTURE.read_text()
        out, _ = preprocess(FIXTURE, raw)

        assert out.count("\n") == raw.count("\n")

        raw_line = next(i for i, l in enumerate(raw.splitlines(), 1) if "SUBROUTINE ROS3_STEP" in l)
        out_line = next(i for i, l in enumerate(out.splitlines(), 1) if "SUBROUTINE ROS3_STEP" in l)
        assert raw_line == out_line

    def test_surviving_code_is_untouched(self):
        out, _ = preprocess(FIXTURE, FIXTURE.read_text())
        assert "Y( I ) = Y( I ) + DT * MAX( Y( I ), 0.0D0 )" in out

    def test_lowercase_file_is_returned_unchanged(self, tmp_path):
        source = tmp_path / "kernel.f90"
        body = "subroutine k(x)\n  real :: x\nend subroutine k\n"
        source.write_text(body)
        out, note = preprocess(source, body)
        assert out == body and note is None

    def test_missing_cpp_reports_rather_than_pretending(self, tmp_path, monkeypatch):
        """A silent no-op would look like a successful pass."""
        monkeypatch.setattr("shutil.which", lambda _name: None)
        raw = FIXTURE.read_text()
        out, note = preprocess(FIXTURE, raw)
        assert out == raw
        assert note and "PATH" in note


class TestEndToEnd:
    def test_loki_finds_the_routine_only_after_preprocessing(self):
        """The finding itself: unpreprocessed, Loki returns nothing."""
        pytest.importorskip("loki")
        from loki import Sourcefile

        raw = FIXTURE.read_text()
        out, _ = preprocess(FIXTURE, raw)

        import tempfile

        def routine_count(text: str) -> int | None:
            """Routines Loki reports, or None when it cannot read the file.

            Both outcomes are failures to parse — which one you get depends
            on where the first offending line sits — so the test asserts
            "nothing usable" rather than one particular flavour.
            """
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "probe.f90"
                path.write_text(text)
                try:
                    return len(Sourcefile.from_file(path).all_subroutines)
                except Exception:  # noqa: BLE001 - a raise is a failure too
                    return None

        assert routine_count(raw) in (0, None)
        assert routine_count(out) == 1

    def test_analyze_parses_the_file(self):
        from fortranspire.agent.analyze import analyze_paths

        reports = analyze_paths([str(FIXTURE)])
        assert reports[0].parse_error is None
        codes = {f.rule_id for f in reports[0].findings}
        assert "FORT009" not in codes, "parse failure on a file that only needed cpp"


class TestLineNumbers:
    """`\\s` includes newlines, so `^\\s*SUBROUTINE` starts on a blank line."""

    def test_leading_blank_lines_do_not_shift_the_reported_line(self):
        from fortranspire.agent.analyze import _line_of_routine

        source = "MODULE m\n\n\n\n      SUBROUTINE foo(a)\n      END SUBROUTINE\n"
        assert _line_of_routine(source, "foo") == 5

    def test_patterns_without_leading_whitespace_are_unaffected(self):
        from fortranspire.agent.analyze import _line_of

        source = "line one\nx = y\n      COMMON / blk / a\n"
        assert _line_of(source, r"COMMON\s*/\s*blk\s*/") == 3
