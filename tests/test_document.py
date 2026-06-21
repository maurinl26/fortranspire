"""Smoke tests for `agent-doc` — exercise the LLM-free code paths only.

These tests run without an `MISTRAL_API_KEY`; they cover:

- routine extraction through Loki (deterministic),
- inline `!>` docstring injection (idempotent),
- Sphinx site scaffolding.

LLM narrative generation (`generate_narrative`) is not covered here — it
needs a real endpoint, so it lives under the `@pytest.mark.llm` selector
in the integration suite.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from fortranspire.agent.document import (
    GENERATED_MARKER,
    extract_routines,
    generate_sphinx_site,
    inject_inline_docstrings,
    main,
)

FIXTURE = Path(__file__).parent / "fixtures" / "doc_kernel.f90"


def test_extract_routines_finds_two_kernels():
    doc = extract_routines(str(FIXTURE))
    names = sorted(r.name for r in doc.routines)
    assert "update_vx" in names
    assert "update_sigma" in names
    assert doc.parse_error is None


def test_inject_inline_docstrings_idempotent(tmp_path: Path):
    target = tmp_path / "doc_kernel.f90"
    shutil.copy(FIXTURE, target)

    doc1 = extract_routines(str(target))
    assert inject_inline_docstrings(doc1) is True

    contents_after_first = target.read_text()
    assert GENERATED_MARKER in contents_after_first
    # One generated block per routine.
    assert contents_after_first.count(GENERATED_MARKER) == len(doc1.routines)

    # Second run on the same file: blocks must be replaced, not duplicated.
    doc2 = extract_routines(str(target))
    inject_inline_docstrings(doc2)
    contents_after_second = target.read_text()
    assert contents_after_second.count(GENERATED_MARKER) == len(doc2.routines)


def test_sphinx_site_scaffold(tmp_path: Path):
    doc = extract_routines(str(FIXTURE))
    output = tmp_path / "documentation"
    generate_sphinx_site([doc], output / "demo", project="demo")

    assert (output / "demo" / "source" / "conf.py").is_file()
    assert (output / "demo" / "source" / "index.rst").is_file()
    assert (output / "demo" / "requirements.txt").is_file()
    assert (output / "demo" / "Makefile").is_file()
    # One .rst per source file.
    rsts = list((output / "demo" / "source").glob("*.rst"))
    assert any(p.name != "index.rst" for p in rsts)


def test_cli_no_llm_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    rc = main([
        "--no-llm",
        "--dry-run",
        str(target),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Dry-run prints the rewritten source — must contain our marker.
    assert GENERATED_MARKER in out
    # Original file must remain unchanged in --dry-run mode.
    assert GENERATED_MARKER not in target.read_text()


def test_cli_site_only(tmp_path: Path):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    rc = main([
        "--no-llm",
        "--site-only",
        "--output", str(tmp_path / "docs_out"),
        "--project", "fixture",
        str(target),
    ])
    assert rc == 0
    assert (tmp_path / "docs_out" / "fixture" / "source" / "index.rst").is_file()
    # Source file untouched in --site-only mode.
    assert GENERATED_MARKER not in target.read_text()
