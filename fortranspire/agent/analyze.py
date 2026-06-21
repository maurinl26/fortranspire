"""Analyze-only mode — runs the deterministic Loki parser stage and emits
findings in human-readable, JSON, or SARIF 2.1.0 format.

No LLM is called. No file is rewritten. Intended as a CI hook and as a
pre-flight check before committing to a full GPU port.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from fortranspire.agent.translation_graph_phase1 import parser_phase1
from fortranspire.agent.toolchain import (
    CompilerInfo,
    best_openacc_compiler,
    detect_compilers,
    summarize as summarize_toolchain,
)

Severity = Literal["error", "warning", "note"]
_SEVERITY_ORDER: dict[Severity, int] = {"note": 0, "warning": 1, "error": 2}


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    message: str
    file: str
    routine: str | None = None
    line: int | None = None
    help_uri: str | None = None


@dataclass
class FileReport:
    file: str
    findings: list[Finding] = field(default_factory=list)
    parse_error: str | None = None


# ── Rule catalog ────────────────────────────────────────────────────────────
# Stable IDs so CI annotations remain comparable across runs.
RULES: dict[str, dict[str, str]] = {
    "FORT001": {
        "name": "IOInKernel",
        "severity": "error",
        "summary": "I/O statements detected inside a kernel candidate — blocks GPU port",
        "help": "Move PRINT/WRITE/READ outside the kernel body before annotating with OpenACC.",
    },
    "FORT002": {
        "name": "SaveAttribute",
        "severity": "warning",
        "summary": "SAVE attribute creates hidden state — must be lifted to INTENT(INOUT)",
        "help": "Promote SAVE variables to explicit arguments so the kernel becomes PURE.",
    },
    "FORT003": {
        "name": "CommonBlock",
        "severity": "warning",
        "summary": "COMMON block detected — incompatible with PURE/ELEMENTAL",
        "help": "Replace COMMON with a MODULE of explicit arguments; the LLM extractor handles this.",
    },
    "FORT004": {
        "name": "LoopCarriedDep",
        "severity": "warning",
        "summary": "Loop-carried dependency suspected — collapse(2) may produce wrong results",
        "help": "Refactor the recurrence or annotate the offending loop as sequential.",
    },
    "FORT005": {
        "name": "PointerAttribute",
        "severity": "warning",
        "summary": "POINTER attribute used — review for GPU data movement",
        "help": "Convert to ALLOCATABLE or to a direct argument; pointers complicate !$acc data clauses.",
    },
    "FORT006": {
        "name": "MissingImplicitNone",
        "severity": "note",
        "summary": "IMPLICIT NONE not enforced — implicit typing is enabled",
        "help": "Add IMPLICIT NONE at the top of every module and subroutine.",
    },
    "FORT007": {
        "name": "ImplicitKind",
        "severity": "note",
        "summary": "REAL/INTEGER declared without an explicit KIND — precision is compiler-dependent",
        "help": "Use REAL(KIND=8) / INTEGER(KIND=4) (or iso_fortran_env kinds) so behavior is portable.",
    },
    "FORT008": {
        "name": "DerivedType",
        "severity": "note",
        "summary": "Derived TYPE detected — Array-of-Structures candidate, review for SoA conversion",
        "help": "AoS layouts hurt GPU coalescing; consider Structure-of-Arrays before annotating.",
    },
    "FORT009": {
        "name": "ParseError",
        "severity": "error",
        "summary": "Loki failed to parse this file — analyzer fell back to regex",
        "help": "Check the file compiles with `gfortran -fsyntax-only` before running the pipeline.",
    },
    "FORT010": {
        "name": "NoFortranCompiler",
        "severity": "warning",
        "summary": "No Fortran compiler found on PATH — generated code cannot be validated",
        "help": "Install gfortran (CPU validation) or the NVIDIA HPC SDK (`nvfortran`, GPU validation).",
    },
    "FORT011": {
        "name": "NoOpenACCCompiler",
        "severity": "warning",
        "summary": "Source uses !$acc directives but no OpenACC-capable compiler is available",
        "help": "Install the NVIDIA HPC SDK (`nvfortran -acc`) or a recent gfortran (≥7, `-fopenacc`).",
    },
}

_HELP_URL = "https://fortranspire.readthedocs.io/en/latest/concepts/fortran-patterns.html#{anchor}"


def _help_uri(rule_id: str) -> str:
    return _HELP_URL.format(anchor=rule_id.lower())


# ── Line-number lookups ─────────────────────────────────────────────────────
# Loki does not return file offsets in the structures parser_phase1 exposes,
# so we re-grep the source for the first occurrence of each pattern. Good
# enough for CI annotations; an exact AST mapping is a later optimisation.

def _line_of(source: str, pattern: str, flags: int = re.IGNORECASE) -> int | None:
    match = re.search(pattern, source, flags | re.MULTILINE)
    if not match:
        return None
    return source.count("\n", 0, match.start()) + 1


def _line_of_routine(source: str, name: str) -> int | None:
    return _line_of(source, rf"^\s*(SUBROUTINE|FUNCTION)\s+{re.escape(name)}\b")


# ── Core ────────────────────────────────────────────────────────────────────

def analyze_file(path: str) -> FileReport:
    """Run parser_phase1 on a single .f90 file and return its findings."""
    abspath = str(Path(path).resolve())
    report = FileReport(file=abspath)

    if not Path(abspath).is_file():
        report.parse_error = f"file not found: {abspath}"
        report.findings.append(_make_finding("FORT009", abspath, report.parse_error))
        return report

    state: dict[str, Any] = {
        "fortran_filepath": abspath,
        "fortran_code": Path(abspath).read_text(encoding="utf-8"),
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "executed_agents": [],
    }

    # parser_phase1 prints status to stdout — capture it so machine output stays clean.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = parser_phase1(state)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover — parser_phase1 already has its own try/except
            report.parse_error = f"{type(exc).__name__}: {exc}"
            report.findings.append(_make_finding("FORT009", abspath, report.parse_error))
            return report

    ast_info = out.get("ast_info", {}) or {}
    kernels = out.get("kernel_results", []) or []

    if ast_info.get("status") == "error":
        msg = ast_info.get("message", "unknown Loki failure")
        report.parse_error = msg
        report.findings.append(_make_finding("FORT009", abspath, msg))
        return report

    src = state["fortran_code"]

    # ── File-level findings ───────────────────────────────────────────
    if not ast_info.get("has_implicit_none", True):
        report.findings.append(_make_finding("FORT006", abspath, None, line=1))

    if ast_info.get("has_implicit_types"):
        line = _line_of(src, r"^\s*(REAL|INTEGER|COMPLEX|DOUBLE\s+PRECISION)\s+\w")
        report.findings.append(_make_finding("FORT007", abspath, None, line=line))

    if ast_info.get("has_pointers"):
        line = _line_of(src, r",\s*POINTER\s*::")
        report.findings.append(_make_finding("FORT005", abspath, None, line=line))

    if ast_info.get("has_derived_types"):
        line = _line_of(src, r"^\s*TYPE\s*::")
        report.findings.append(_make_finding("FORT008", abspath, None, line=line))

    for block in ast_info.get("common_blocks", []) or []:
        name = block.get("name", "?")
        line = _line_of(src, rf"COMMON\s*/\s*{re.escape(name)}\s*/")
        report.findings.append(
            _make_finding("FORT003", abspath, f"COMMON block `/{name}/`", line=line)
        )

    # ── Per-routine findings ──────────────────────────────────────────
    for kernel in kernels:
        routine = kernel.get("routine_name", "?")
        rline = _line_of_routine(src, routine)

        if kernel.get("has_io"):
            report.findings.append(
                _make_finding("FORT001", abspath, f"routine `{routine}` contains I/O",
                              routine=routine, line=rline)
            )
        if kernel.get("has_save"):
            report.findings.append(
                _make_finding("FORT002", abspath, f"routine `{routine}` uses SAVE",
                              routine=routine, line=rline)
            )
        if kernel.get("has_loop_carried_dep"):
            report.findings.append(
                _make_finding("FORT004", abspath, f"routine `{routine}` has a suspected loop-carried dependency",
                              routine=routine, line=rline)
            )

    return report


def analyze_paths(paths: Iterable[str]) -> list[FileReport]:
    """Walk `paths` and analyze every .f90 / .F90 file found."""
    files: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(str(f) for f in p.rglob("*.[fF]90"))
        else:
            files.append(str(p))
    return [analyze_file(f) for f in sorted(set(files))]


_ACC_PRAGMA_RE = re.compile(r"^\s*!\$acc\b", re.IGNORECASE | re.MULTILINE)


def _any_openacc_in(reports: list[FileReport]) -> str | None:
    """Return the first source file that contains an `!$acc` pragma, if any."""
    for r in reports:
        if not Path(r.file).is_file():
            continue
        try:
            blob = Path(r.file).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _ACC_PRAGMA_RE.search(blob):
            return r.file
    return None


def toolchain_findings(
    compilers: list[CompilerInfo], reports: list[FileReport]
) -> list[Finding]:
    """Cross-check what the source needs against what the host provides."""
    findings: list[Finding] = []

    if not compilers:
        findings.append(_make_finding(
            "FORT010", "<toolchain>",
            "no `gfortran`, `nvfortran`, `ifx`, `flang` or `lfortran` on PATH",
        ))
        return findings

    has_acc_source = _any_openacc_in(reports)
    has_acc_compiler = best_openacc_compiler(compilers) is not None
    if has_acc_source and not has_acc_compiler:
        detected = ", ".join(f"{c.name} {c.version or '?'}" for c in compilers)
        findings.append(_make_finding(
            "FORT011", has_acc_source,
            f"detected compilers: {detected}",
        ))

    return findings


def _make_finding(
    rule_id: str,
    file: str,
    detail: str | None,
    *,
    routine: str | None = None,
    line: int | None = None,
) -> Finding:
    rule = RULES[rule_id]
    base = rule["summary"]
    message = f"{base} — {detail}" if detail else base
    return Finding(
        rule_id=rule_id,
        severity=rule["severity"],  # type: ignore[arg-type]
        message=message,
        file=file,
        routine=routine,
        line=line,
        help_uri=_help_uri(rule_id),
    )


# ── Emitters ────────────────────────────────────────────────────────────────

_ANSI = {
    "error":   "\033[31m",  # red
    "warning": "\033[33m",  # yellow
    "note":    "\033[36m",  # cyan
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
}


def render_text(
    reports: list[FileReport],
    use_color: bool = True,
    compilers: list[CompilerInfo] | None = None,
    toolchain_only_findings: list[Finding] | None = None,
) -> str:
    out: list[str] = []
    counters: dict[Severity, int] = {"error": 0, "warning": 0, "note": 0}

    def c(key: str, s: str) -> str:
        return f"{_ANSI[key]}{s}{_ANSI['reset']}" if use_color else s

    for r in reports:
        if not r.findings:
            out.append(c("dim", f"  ok  {_rel(r.file)}"))
            continue
        out.append(c("bold", _rel(r.file)))
        for f in r.findings:
            counters[f.severity] += 1
            loc = f":{f.line}" if f.line else ""
            routine = f" [{f.routine}]" if f.routine else ""
            sev = c(f.severity, f.severity.upper().ljust(7))
            out.append(f"  {sev} {f.rule_id}{routine}  {_rel(f.file)}{loc}  {f.message}")

    if toolchain_only_findings:
        out.append(c("bold", "<toolchain>"))
        for f in toolchain_only_findings:
            counters[f.severity] += 1
            sev = c(f.severity, f.severity.upper().ljust(7))
            out.append(f"  {sev} {f.rule_id}  {f.message}")

    if compilers is not None:
        out.append("")
        out.append(c("dim", summarize_toolchain(compilers)))

    total = sum(counters.values())
    summary = (
        f"\n{counters['error']} error(s), {counters['warning']} warning(s), "
        f"{counters['note']} note(s) — {len(reports)} file(s) analyzed."
    )
    out.append(c("bold", summary) if total else c("dim", summary))
    return "\n".join(out)


def render_json(
    reports: list[FileReport],
    compilers: list[CompilerInfo] | None = None,
    toolchain_only_findings: list[Finding] | None = None,
) -> str:
    payload: dict = {
        "files": [
            {
                "file": r.file,
                "parse_error": r.parse_error,
                "findings": [_finding_to_dict(f) for f in r.findings],
            }
            for r in reports
        ],
    }
    if toolchain_only_findings is not None:
        payload["toolchain_findings"] = [_finding_to_dict(f) for f in toolchain_only_findings]
    if compilers is not None:
        payload["toolchain"] = {
            "compilers": [c.to_dict() for c in compilers],
            "recommended_openacc": (
                best_openacc_compiler(compilers).name
                if best_openacc_compiler(compilers) else None
            ),
        }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _finding_to_dict(f: Finding) -> dict:
    return {
        "rule_id": f.rule_id,
        "severity": f.severity,
        "message": f.message,
        "file": f.file,
        "routine": f.routine,
        "line": f.line,
        "help_uri": f.help_uri,
    }


def render_sarif(
    reports: list[FileReport],
    compilers: list[CompilerInfo] | None = None,
    toolchain_only_findings: list[Finding] | None = None,
) -> str:
    """Emit SARIF 2.1.0 — uploadable to GitHub Code Scanning."""
    extra: list[Finding] = list(toolchain_only_findings or [])
    used_rule_ids = sorted(
        {f.rule_id for r in reports for f in r.findings}
        | {f.rule_id for f in extra}
    )

    rules = [
        {
            "id": rid,
            "name": RULES[rid]["name"],
            "shortDescription": {"text": RULES[rid]["summary"]},
            "fullDescription": {"text": RULES[rid]["help"]},
            "helpUri": _help_uri(rid),
            "defaultConfiguration": {"level": _sarif_level(RULES[rid]["severity"])},
        }
        for rid in used_rule_ids
    ]

    results = []
    for r in reports:
        for f in r.findings:
            results.append(_sarif_result(f))
    for f in extra:
        results.append(_sarif_result(f))

    run: dict = {
        "tool": {
            "driver": {
                "name": "fortranspire",
                "informationUri": "https://github.com/maurinl26/fortranspire",
                "rules": rules,
            }
        },
        "results": results,
    }
    if compilers is not None:
        run["properties"] = {
            "toolchain": [c.to_dict() for c in compilers],
            "recommended_openacc": (
                best_openacc_compiler(compilers).name
                if best_openacc_compiler(compilers) else None
            ),
        }

    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }
    return json.dumps(sarif, indent=2, ensure_ascii=False)


def _sarif_result(f: Finding) -> dict:
    location_uri = _rel(f.file) if Path(f.file).exists() else f.file
    return {
        "ruleId": f.rule_id,
        "level": _sarif_level(f.severity),
        "message": {"text": f.message},
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": location_uri},
                "region": {"startLine": max(f.line or 1, 1)},
            }
        }],
        "properties": {"routine": f.routine} if f.routine else {},
    }


def _sarif_level(severity: str) -> str:
    return {"error": "error", "warning": "warning", "note": "note"}.get(severity, "note")


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


# ── Exit-code policy ────────────────────────────────────────────────────────

def compute_exit_code(
    reports: list[FileReport],
    fail_on: Severity,
    toolchain_only_findings: list[Finding] | None = None,
) -> int:
    threshold = _SEVERITY_ORDER[fail_on]
    for r in reports:
        for f in r.findings:
            if _SEVERITY_ORDER[f.severity] >= threshold:
                return 1
    for f in toolchain_only_findings or []:
        if _SEVERITY_ORDER[f.severity] >= threshold:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="agent-analyze",
        description=(
            "Static analysis of Fortran sources — runs the Loki-based parser "
            "stage only. Zero LLM calls, zero file rewrites. Suitable as a CI "
            "hook or a pre-commit check."
        ),
        epilog=(
            "Examples:\n"
            "  agent-analyze src/kernel.f90\n"
            "  agent-analyze --format sarif src/ > fortranspire.sarif\n"
            "  agent-analyze --fail-on warning src/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Fortran files or directories to analyze")
    parser.add_argument(
        "--format", choices=("text", "json", "sarif"), default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--fail-on", choices=("error", "warning", "note"), default="error",
        help="Lowest severity that produces a non-zero exit code (default: error)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors in text output",
    )
    parser.add_argument(
        "--no-toolchain-check", action="store_true",
        help="Skip Fortran compiler / OpenACC capability detection on the host",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write output to this file instead of stdout",
    )

    args = parser.parse_args(argv)

    reports = analyze_paths(args.paths)

    compilers: list[CompilerInfo] | None = None
    tc_findings: list[Finding] | None = None
    if not args.no_toolchain_check:
        compilers = detect_compilers()
        tc_findings = toolchain_findings(compilers, reports)

    if args.format == "text":
        rendered = render_text(
            reports,
            use_color=not args.no_color and sys.stdout.isatty(),
            compilers=compilers,
            toolchain_only_findings=tc_findings,
        )
    elif args.format == "json":
        rendered = render_json(reports, compilers=compilers, toolchain_only_findings=tc_findings)
    else:
        rendered = render_sarif(reports, compilers=compilers, toolchain_only_findings=tc_findings)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    return compute_exit_code(reports, args.fail_on, tc_findings)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
