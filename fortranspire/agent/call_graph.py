"""``fortranspire graph`` — module-level call graph + Mermaid renderer.

Closes issue #15.

Walks the Fortran sources, uses Loki to extract every CALL relationship
inside each routine, and emits a Mermaid ``flowchart`` block per file
(or one merged graph across the codebase). Output is a self-contained
Markdown file ready to drop into a Sphinx site, GitHub README, or
internal wiki — Mermaid is rendered natively by GitHub, GitLab, Notion,
Obsidian and any Sphinx site with ``sphinxcontrib-mermaid``.

Zero LLM call by default. With ``--narrate``, an LLM pass adds a short
"How this module fits in the system" paragraph per file, using the
graph as context — useful for new contributors on a legacy codebase.
"""
from __future__ import annotations

from fortranspire.agent.nodes._common import collect_fortran_files

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from fortranspire.agent.nodes import parser_phase1


@dataclass
class RoutineNode:
    name: str
    file: str               # the source file this routine was extracted from
    calls: list[str] = field(default_factory=list)  # callee names (best-effort)


@dataclass
class FileGraph:
    file: str
    routines: list[RoutineNode] = field(default_factory=list)
    parse_error: str | None = None


# ── Loki extraction ────────────────────────────────────────────────────────

def _extract_one(path: str) -> FileGraph:
    """Run the parser and collect CALL targets per routine."""
    abspath = str(Path(path).resolve())
    state = {
        "fortran_filepath": abspath,
        "fortran_code": Path(abspath).read_text(encoding="utf-8"),
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "executed_agents": [],
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            parser_phase1(state)
        except Exception as exc:
            return FileGraph(file=abspath, parse_error=str(exc))

    # parser_phase1's `kernel_results` carries per-routine intent/dimensions/etc.
    # but not the CALL targets. Re-walk via Loki to get those — fast, in-process.
    try:
        from loki import Sourcefile, FindNodes
        from loki.ir.nodes import CallStatement
    except ImportError:
        return FileGraph(file=abspath, parse_error="loki not installed")

    try:
        source = Sourcefile.from_file(abspath)
    except Exception as exc:
        return FileGraph(file=abspath, parse_error=f"Loki parse: {exc}")

    routines: list[RoutineNode] = []
    all_routines = list(source.routines)
    for module in (source.modules or []):
        for routine in (module.subroutines or []):
            if routine not in all_routines:
                all_routines.append(routine)

    for routine in all_routines:
        calls: list[str] = []
        try:
            for call in FindNodes(CallStatement).visit(routine.body):
                callee = str(getattr(call, "name", "?")).strip()
                if callee and callee not in calls:
                    calls.append(callee)
        except Exception:
            # Some routine bodies in REGEX-frontend mode aren't walkable;
            # leave calls empty rather than crash the whole graph.
            pass
        routines.append(RoutineNode(name=routine.name, file=abspath, calls=calls))

    return FileGraph(file=abspath, routines=routines)


def extract_graphs(paths: Iterable[str]) -> list[FileGraph]:
    """Walk every `.f90` / `.F90` under `paths` and return per-file graphs."""
    files: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            # Shared discovery: fixed-form suffixes too (.F, .f, .for),
            # which are most of the legacy corpus.
            files.extend(collect_fortran_files([p]))
        else:
            files.append(str(p))
    seen: set[str] = set()
    return [_extract_one(f) for f in files if not (f in seen or seen.add(f))]


# ── Mermaid rendering ──────────────────────────────────────────────────────

def render_mermaid(graph: FileGraph) -> str:
    """Return a Mermaid ``flowchart LR`` block describing one file's call graph.

    External callees (names referenced but not defined in this file) get a
    dashed border so the reader can tell what crosses the file boundary.
    """
    if graph.parse_error:
        return f"```text\n(parse error: {graph.parse_error})\n```"
    if not graph.routines:
        return "```text\n(no routines)\n```"

    defined = {r.name.lower() for r in graph.routines}
    lines: list[str] = ["```mermaid", "flowchart LR"]

    # Nodes: internal first, then external (callees not defined in this file).
    external: set[str] = set()
    for r in graph.routines:
        lines.append(f"    {_mermaid_id(r.name)}[\"{r.name}\"]")
    for r in graph.routines:
        for callee in r.calls:
            if callee.lower() not in defined:
                external.add(callee)
    for ext in sorted(external):
        lines.append(f"    {_mermaid_id(ext)}([\"{ext}\"]):::external")

    # Edges
    for r in graph.routines:
        for callee in r.calls:
            lines.append(f"    {_mermaid_id(r.name)} --> {_mermaid_id(callee)}")

    lines.append("    classDef external stroke-dasharray: 4 2, fill:#f5f5f5;")
    lines.append("```")
    return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Sanitize a routine name into a Mermaid-safe identifier."""
    import re
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not s or s[0].isdigit():
        s = "n_" + s
    return s


# ── LLM narration (optional) ───────────────────────────────────────────────

def narrate_graph(graph: FileGraph) -> str:
    """One-paragraph LLM narration of how the routines in `graph` compose.

    Lazy-imports the LLM stack so `--no-narrate` (default) runs on the
    core-only install. Falls back to a templated description if the LLM
    call fails — keeps the report writer-friendly even on a flaky endpoint.
    """
    if graph.parse_error or not graph.routines:
        return ""

    summary = ", ".join(
        f"`{r.name}` calls [{', '.join(r.calls) or '—'}]"
        for r in graph.routines[:20]
    )

    try:
        from fortranspire.llm import get_llm
        from fortranspire.observability import tracer
        from fortranspire.observability.llm_callback import token_callback
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return ""

    system = SystemMessage(content=(
        "You explain how the routines in a Fortran source file fit together. "
        "One paragraph, plain text, no Markdown. Focus on the data-flow "
        "(who feeds whom), the role each routine plays in the overall "
        "computation, and any obvious external dependency. Keep it under "
        "120 words."
    ))
    user = HumanMessage(content=(
        f"File: {Path(graph.file).name}\n"
        f"Routines and their call targets:\n  {summary}\n\n"
        "Write the paragraph."
    ))

    try:
        llm = get_llm("code")
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        with tracer.span(node="call_graph_narrate", model=model_name) as span:
            cfg = {"callbacks": [token_callback(span)]}
            resp = llm.invoke([system, user], config=cfg)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return text.strip()
    except Exception:
        return ""


# ── Markdown report ────────────────────────────────────────────────────────

def render_report(graphs: list[FileGraph], *, narrate: bool = False) -> str:
    """Render the full report — Mermaid block + optional narration per file."""
    lines: list[str] = []
    lines.append("# fortranspire — module call-graph report")
    lines.append("")
    lines.append(f"Scanned **{len(graphs)} file(s)**. "
                 f"**{sum(len(g.routines) for g in graphs)}** routines total.")
    lines.append("")

    for g in graphs:
        lines.append(f"## `{Path(g.file).name}`")
        lines.append("")
        if g.parse_error:
            lines.append(f"> Parse failure: {g.parse_error}")
            lines.append("")
            continue
        if not g.routines:
            lines.append("> (no routines)")
            lines.append("")
            continue

        # Per-file narration first (if requested) so the diagram makes
        # sense when scrolled past.
        if narrate:
            paragraph = narrate_graph(g)
            if paragraph:
                lines.append(paragraph)
                lines.append("")

        lines.append(render_mermaid(g))
        lines.append("")

        # Routine list with their callees as a fallback view when Mermaid
        # rendering isn't available (e.g. plain-text consumers).
        lines.append("**Routines:**")
        for r in g.routines:
            callees = ", ".join(f"`{c}`" for c in r.calls) or "*(no calls)*"
            lines.append(f"- `{r.name}` → {callees}")
        lines.append("")

    lines.append("---")
    lines.append("Generated by **`fortranspire graph`**. Mermaid blocks render "
                 "natively in GitHub, GitLab, Notion, Obsidian, and Sphinx sites "
                 "with the `sphinxcontrib-mermaid` extension.")
    lines.append("")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fortranspire graph",
        description=(
            "Module-level call-graph report for a Fortran codebase. "
            "Emits a Markdown file with one Mermaid `flowchart` block per "
            "source file. Useful for code reviews, onboarding, and "
            "before/after audits of a pipeline run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fortranspire graph src/                     # all files\n"
            "  fortranspire graph src/seismic.f90          # single file\n"
            "  fortranspire graph --narrate src/           # + LLM paragraph per file\n"
            "  fortranspire graph -o graph.md src/         # save to file\n"
        ),
    )
    parser.add_argument("paths", nargs="+",
                        help="Fortran files or directories to graph")
    parser.add_argument("--narrate", action="store_true",
                        help="Add a 1-paragraph LLM narration per file "
                             "(requires the [gpu] extra)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write to a file instead of stdout")
    args = parser.parse_args(argv)

    graphs = extract_graphs(args.paths)
    if not graphs:
        print("fortranspire graph: no .f90 / .F90 file found in the given paths",
              file=sys.stderr)
        return 2

    report = render_report(graphs, narrate=args.narrate)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"fortranspire graph: report saved to {args.output} "
              f"({sum(len(g.routines) for g in graphs)} routine(s), "
              f"{len(graphs)} file(s))")
    else:
        print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
