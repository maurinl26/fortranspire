"""Tests for the unified `fortranspire <verb>` dispatcher — issue #8."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from fortranspire.cli import _DISPATCH, _HELP, main


FIXTURE = Path(__file__).parent / "fixtures" / "doc_kernel.f90"


# ── Top-level dispatcher ────────────────────────────────────────────────────

def test_help_prints_command_list(capsys: pytest.CaptureFixture[str]):
    rc = main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fortranspire — Fortran → GPU/JAX pipeline" in out
    for cmd in _DISPATCH:
        assert cmd in out


def test_no_argv_prints_help(capsys: pytest.CaptureFixture[str]):
    rc = main([])
    assert rc == 0
    assert "Usage:" in capsys.readouterr().out


def test_unknown_command_returns_2(capsys: pytest.CaptureFixture[str]):
    rc = main(["definitely-not-a-command"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_dispatch_table_lists_expected_commands():
    # 16 subcommands. Update this set when a new verb lands.
    expected = {
        "analyze", "doc", "explain", "format",
        "graph", "diff", "report", "bench",     # added during the 0.1.0 sprint
        "gpu", "port-batch", "translate", "profile", "mcp",
        "github-app",                           # issue #50
        "gt4py",                                # issue #42
        "domain",                               # issue #88
    }
    assert set(_DISPATCH.keys()) == expected


# ── End-to-end dispatch on no-LLM commands ─────────────────────────────────

def test_dispatch_analyze_help_works(capsys: pytest.CaptureFixture[str]):
    rc = main(["analyze", "--help"])
    # argparse `--help` exits 0 via SystemExit; the dispatcher converts it
    assert rc == 0
    out = capsys.readouterr().out
    assert "agent-analyze" in out or "Fortran" in out


def test_dispatch_explain_runs_on_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    rc = main(["explain", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "port-cost estimate" in out


def test_dispatch_doc_no_llm_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    target = tmp_path / "k.f90"
    shutil.copy(FIXTURE, target)
    rc = main(["doc", "--no-llm", "--dry-run", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "@generated_by fortranspire" in out


def test_dispatch_port_batch_empty_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    empty = tmp_path / "nothing"
    empty.mkdir()
    rc = main(["port-batch", str(empty)])
    assert rc == 2


def test_argv_restored_after_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    sentinel = ["myscript", "original", "args"]
    monkeypatch.setattr(sys, "argv", list(sentinel))
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    main(["explain", str(tmp_path)])
    # sys.argv must be back to what it was before
    assert sys.argv == sentinel


# ── Deprecation notice on legacy `agent-*` aliases ─────────────────────────

def test_legacy_run_analyze_prints_deprecation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
):
    """The agent-analyze legacy entry must emit a stderr deprecation line."""
    from fortranspire.agent.cli import run_analyze
    monkeypatch.setattr(sys, "argv", ["agent-analyze", "--help"])
    with pytest.raises(SystemExit):
        run_analyze()
    err = capsys.readouterr().err
    assert "agent-analyze` is deprecated" in err
    assert "fortranspire analyze" in err
    assert "0.3" in err


def test_unified_dispatch_does_not_print_deprecation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    """The unified `fortranspire analyze` route must NOT show deprecation."""
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    main(["analyze", "--no-toolchain-check", str(tmp_path)])
    err = capsys.readouterr().err
    assert "deprecated" not in err


# Regression guard for #0.1.1: the unified CLI used to dispatch
# `gpu`/`translate`/`profile` through the legacy `run_*` wrappers,
# which always print the `agent-*` deprecation notice. The 0.1.1 fix
# splits each into a `_*_main()` shared entry; the dispatch table now
# points at those. This test would fail against 0.1.0 wheel output.

@pytest.mark.parametrize("cmd", ["gpu", "translate", "profile"])
def test_unified_dispatch_silent_on_legacy_shaped_commands(
    cmd: str, capsys: pytest.CaptureFixture[str],
):
    """`fortranspire gpu/translate/profile --help` must NOT show the
    `agent-*` deprecation notice — that's reserved for the legacy
    console scripts."""
    main([cmd, "--help"])
    err = capsys.readouterr().err
    assert "deprecated" not in err.lower(), (
        f"`fortranspire {cmd}` should not show the legacy "
        f"`agent-{cmd}` deprecation message"
    )


# ── Loki first-import workaround (issue #71) ───────────────────────────────
#
# loki-ifs has a fragile first import under Python 3.12: a re-entrant
# `import logging` in loki/logging.py surfaces as "partially initialized
# module 'logging'". It only reproduces in a clean PyPI install, not under
# the vendored-loki dev environment, so these tests pin the *structure* of
# the workaround rather than the Heisenbug itself.

class TestLokiWarmup:
    def test_main_warms_loki_before_dispatching_a_loki_verb(self, monkeypatch):
        """`graph` imported loki unguarded and crashed; warming it first
        in a clean state we verified works sidesteps the fragile import."""
        import fortranspire.cli as cli

        warmed: list[bool] = []
        monkeypatch.setattr(cli, "_warm_loki", lambda: warmed.append(True))
        # `graph` dispatches into call_graph; stop before real work by
        # pointing it at nothing and swallowing its exit.
        cli.main(["graph", "--help"])
        assert warmed == [True]

    def test_main_does_not_warm_loki_for_mcp(self, monkeypatch):
        """`mcp` never touches loki — no reason to import it (and it would
        pull the parser stack into the server process)."""
        import fortranspire.cli as cli

        warmed: list[bool] = []
        monkeypatch.setattr(cli, "_warm_loki", lambda: warmed.append(True))
        monkeypatch.setitem(cli._DISPATCH, "mcp", ("fortranspire.cli", "_noop_entry"))
        cli._noop_entry = lambda: 0  # type: ignore[attr-defined]
        cli.main(["mcp"])
        assert warmed == []

    def test_warm_loki_tolerates_a_broken_loki(self, monkeypatch):
        """A failure in the warm-up must never be fatal — the verb's own
        code path handles a missing or broken loki."""
        import builtins

        import fortranspire.cli as cli

        real_import = builtins.__import__

        def boom(name, *a, **k):
            if name == "loki":
                raise AttributeError("partially initialized module 'logging'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", boom)
        cli._warm_loki()  # must not raise

    def test_call_graph_guard_catches_more_than_import_error(self):
        """The bug was an AttributeError; the guard only caught ImportError,
        so `graph` crashed while `analyze` (broad guard) survived."""
        import inspect

        from fortranspire.agent import call_graph

        src = inspect.getsource(call_graph._extract_one)
        # The loki import is now guarded by a broad Exception, not ImportError.
        assert "except Exception" in src
        assert "except ImportError:\n        return FileGraph" not in src
