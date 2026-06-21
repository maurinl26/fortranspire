"""Unified ``fortranspire <verb>`` CLI dispatcher — issue #8.

Replaces the fan-out of ``agent-analyze`` / ``agent-doc`` / ``agent-explain``
/ ``agent-format`` / ``agent-gpu`` / ``agent-port-batch`` / ``agent-translate``
/ ``agent-profile`` / ``run-mcp`` console scripts with a single
``fortranspire`` entry point that takes a subcommand. Pattern matches
``cargo``, ``kubectl``, ``git`` — discoverable, scriptable, one line of
shell autocompletion to wire.

The legacy ``agent-*`` scripts still work and forward to the same code
paths, with a one-line deprecation warning on stderr (see
:func:`_deprecation_notice` in :mod:`fortranspire.agent.cli`).
Removal is scheduled for 0.3 — pinned in the deprecation message so
users can plan.

Implementation: each subcommand dispatches to the existing module's
``main()`` function (or its ``run_*`` wrapper for the legacy entries
that don't have a ``main`` yet). ``sys.argv`` is rewritten before the
dispatch so the dispatched argparse sees a clean command line.
``SystemExit`` from inside the dispatched function is caught so the
return code propagates correctly without aborting subsequent logic
(matters for embedding ``fortranspire.cli.main()`` in tests).
"""
from __future__ import annotations

import importlib
import sys

# Subcommand → (module name, callable name). The callable can be either:
#   - a ``main(argv=None) -> int`` (preferred — analyze, doc, explain, …)
#   - a legacy ``run_*()`` that calls ``sys.exit`` internally (gpu / translate /
#     profile). The dispatcher catches the SystemExit and propagates the code.
_DISPATCH: dict[str, tuple[str, str]] = {
    "analyze":    ("fortranspire.agent.analyze",  "main"),
    "doc":        ("fortranspire.agent.document", "main"),
    "explain":    ("fortranspire.agent.explain",  "main"),
    "format":     ("fortranspire.agent.format",   "main"),
    "port-batch": ("fortranspire.agent.batch",    "main"),
    "graph":      ("fortranspire.agent.call_graph", "main"),
    "diff":       ("fortranspire.agent.diff",     "main"),
    "report":     ("fortranspire.agent.report",   "main"),
    "gpu":        ("fortranspire.agent.cli",      "run_translate_gpu"),
    "translate":  ("fortranspire.agent.cli",      "run_translate"),
    "profile":    ("fortranspire.agent.cli",      "run_profile"),
    "mcp":        ("fortranspire.server",         "main"),
}

_HELP = """\
fortranspire — Fortran → GPU/JAX pipeline (Mistral-driven, MCP-exposed)

Usage:
  fortranspire <command> [options...]
  fortranspire --help
  fortranspire <command> --help

Commands:
  analyze     Static analysis (Loki AST, no LLM, CI-friendly)
  doc         Documentation generator (inline !> + Sphinx)
  explain     Pre-flight cost + risk estimate (no LLM, no tokens)
  format      Fortran source formatter (fprettify wrapper)
  graph       Module-level call-graph report (Mermaid flowchart)
  diff        Semantic before/after diff viewer (text or HTML)
  report      HTML audit dashboard for a Phase-1 output directory
  gpu         Phase 1: Fortran → GPU (OpenACC) + Cython wrapper
  port-batch  Parallel Phase 1 port across many files
  translate   Phase 2: Fortran → JAX (experimental)
  profile     Performance benchmarking
  mcp         Run the MCP server (HTTP/SSE)

Legacy aliases (deprecated, removed in 0.3):
  agent-analyze, agent-doc, agent-explain, agent-format, agent-gpu,
  agent-port-batch, agent-translate, agent-profile, run-mcp.

Examples:
  fortranspire analyze src/
  fortranspire explain --output estimate.md src/
  fortranspire gpu src/kernel.f90
  fortranspire port-batch -j 4 -o /tmp/out src/
"""


def _print_help(stream=None) -> None:
    # Default `sys.stdout` resolved at call time so test capture (capsys,
    # contextlib.redirect_stdout, …) sees the output. Don't change to a
    # default param — Python binds default values at function definition,
    # which captures the ORIGINAL stdout from module import.
    print(_HELP, file=stream if stream is not None else sys.stdout)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv`` to the matching subcommand. Returns its exit code."""
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    cmd = args[0]
    if cmd not in _DISPATCH:
        print(f"fortranspire: unknown command {cmd!r}", file=sys.stderr)
        _print_help(stream=sys.stderr)   # resolved at call time
        return 2

    module_name, attr = _DISPATCH[cmd]
    rest = args[1:]
    saved_argv = sys.argv
    sys.argv = [f"fortranspire {cmd}"] + rest

    try:
        module = importlib.import_module(module_name)
        entry = getattr(module, attr)
        result = entry()
        # `main(argv)` returns int; `run_*()` returns None and sys.exits.
        return int(result) if isinstance(result, int) else 0
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
