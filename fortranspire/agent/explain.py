"""``agent-explain`` — pre-flight cost + risk estimator (no LLM calls).

Answers the question a decision-maker asks before opening their wallet:

  "How much will fortranspire cost to port this codebase, and what's
   going to bite me?"

Runs the deterministic Loki parser, sums per-routine token-budget
estimates against the per-stage model price table, surfaces the
structural risks already detected by ``agent-analyze``, and emits a
Markdown report the operator can forward to a buyer.

Zero LLM calls. Zero token cost. Suitable for sales / RFI / planning.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from fortranspire.agent.nodes import parser_phase1
from fortranspire.observability.pricing import estimate_cost_usd

# Per-routine token estimates — calibrated against typical seismic_CPML
# and AROME-class kernels. Conservative: rounded up so the estimate is
# a soft upper bound on real cost.
_TOKENS_PER_ROUTINE_REASONING_IN  = 1500   # extractor + openacc context
_TOKENS_PER_ROUTINE_REASONING_OUT = 1000   # rewritten Fortran block
_TOKENS_PER_ROUTINE_CODE_IN       = 800    # cython_pyx + cython_header context
_TOKENS_PER_ROUTINE_CODE_OUT      = 600    # .pyx + .h boilerplate

# One-shot extractor call (per file, not per routine).
_TOKENS_EXTRACTOR_IN_BASE  = 3500
_TOKENS_EXTRACTOR_OUT_BASE = 2500

# Model names used by the pipeline (see fortranspire/llm/__init__.py).
_REASONING_MODEL = os.getenv("MISTRAL_MODEL_REASONING", "mistral-large-latest")
_CODE_MODEL      = os.getenv("MISTRAL_MODEL_CODE",      "codestral-latest")

# Risk catalog mirrors agent-analyze rules — kept in sync manually
# rather than imported to keep this module dep-free.
_RISK_LABELS: dict[str, str] = {
    "FORT001": "I/O in kernel candidate (blocks GPU port)",
    "FORT002": "SAVE attribute (hidden state)",
    "FORT003": "COMMON block",
    "FORT004": "Suspected loop-carried dependency",
    "FORT005": "POINTER attribute",
    "FORT006": "IMPLICIT NONE missing",
    "FORT007": "REAL/INTEGER without KIND",
    "FORT008": "Derived TYPE (AoS — consider SoA)",
    "FORT009": "Loki parse failure",
}


@dataclass
class FileEstimate:
    path: str
    n_routines: int
    routines: list[str] = field(default_factory=list)
    risks: list[tuple[str, str, str]] = field(default_factory=list)  # (rule, severity, detail)
    parse_error: str | None = None

    # Token / cost rollup
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_cost_usd: float = 0.0
    code_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.reasoning_cost_usd + self.code_cost_usd


@dataclass
class CodebaseEstimate:
    files: list[FileEstimate] = field(default_factory=list)
    paths_scanned: list[str] = field(default_factory=list)

    @property
    def n_files_ok(self) -> int:
        return sum(1 for f in self.files if not f.parse_error)

    @property
    def n_routines(self) -> int:
        return sum(f.n_routines for f in self.files)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(f.prompt_tokens for f in self.files)

    @property
    def total_completion_tokens(self) -> int:
        return sum(f.completion_tokens for f in self.files)

    @property
    def total_cost_usd(self) -> float:
        return sum(f.total_cost_usd for f in self.files)


# ── Loki integration ────────────────────────────────────────────────────────

def _parse_file(path: str) -> tuple[list[str], dict, str | None]:
    """Run parser_phase1 silently. Returns (routine_names, ast_info, error)."""
    state = {
        "fortran_filepath": str(Path(path).resolve()),
        "fortran_code": Path(path).read_text(encoding="utf-8"),
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "executed_agents": [],
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = parser_phase1(state)
        except Exception as exc:
            return [], {}, str(exc)

    ast_info = out.get("ast_info") or {}
    if ast_info.get("status") == "error":
        return [], {}, ast_info.get("message", "parse error")

    routines = [k["routine_name"] for k in (out.get("kernel_results") or [])]
    return routines, ast_info, None


def _collect_risks(routines: list[str], ast_info: dict, kernels: list[dict]) -> list[tuple[str, str, str]]:
    """Map parser_phase1 output to (rule_id, severity, detail) tuples — same set as agent-analyze."""
    risks: list[tuple[str, str, str]] = []

    if not ast_info.get("has_implicit_none", True):
        risks.append(("FORT006", "note", "module lacks IMPLICIT NONE"))
    if ast_info.get("has_implicit_types"):
        risks.append(("FORT007", "note", "REAL/INTEGER declared without explicit KIND"))
    if ast_info.get("has_pointers"):
        risks.append(("FORT005", "warning", "POINTER attribute present"))
    if ast_info.get("has_derived_types"):
        risks.append(("FORT008", "note", "derived TYPE detected"))
    for block in ast_info.get("common_blocks") or []:
        risks.append(("FORT003", "warning", f"COMMON /{block.get('name', '?')}/"))

    for k in kernels or []:
        rname = k.get("routine_name", "?")
        if k.get("has_io"):
            risks.append(("FORT001", "error", f"routine `{rname}` contains I/O"))
        if k.get("has_save"):
            risks.append(("FORT002", "warning", f"routine `{rname}` uses SAVE"))
        if k.get("has_loop_carried_dep"):
            risks.append(("FORT004", "warning", f"routine `{rname}` has suspected loop-carried dep"))

    return risks


def estimate_file(path: str) -> FileEstimate:
    """Heuristic cost + risk estimate for a single .f90 file. Zero LLM calls."""
    abspath = str(Path(path).resolve())

    # Re-run the parser so we get both routine names and the full kernel dict
    # (for risk surfacing — _parse_file only returns the names).
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
            out = parser_phase1(state)
        except Exception as exc:
            return FileEstimate(path=abspath, n_routines=0, parse_error=str(exc))

    ast_info = out.get("ast_info") or {}
    if ast_info.get("status") == "error":
        return FileEstimate(path=abspath, n_routines=0,
                            parse_error=ast_info.get("message", "parse error"))

    kernels = out.get("kernel_results") or []
    routine_names = [k["routine_name"] for k in kernels]
    risks = _collect_risks(routine_names, ast_info, kernels)

    n = len(kernels)

    # Token budget per file:
    #   - 1 extractor call per file (reasoning model)
    #   - 1 openacc kernel call per routine (reasoning)
    #   - 1 openacc driver call per file (reasoning)
    #   - 1 cython pyx call + 1 cython header call per file (code model)
    #   - 1 doc_routine call per routine (code), assumed in scope when called
    #     by agent-doc — counted under code model here as well.
    reasoning_in = (
        _TOKENS_EXTRACTOR_IN_BASE
        + n * _TOKENS_PER_ROUTINE_REASONING_IN          # per-kernel openacc
        + _TOKENS_PER_ROUTINE_REASONING_IN              # one-shot driver
    )
    reasoning_out = (
        _TOKENS_EXTRACTOR_OUT_BASE
        + n * _TOKENS_PER_ROUTINE_REASONING_OUT
        + _TOKENS_PER_ROUTINE_REASONING_OUT
    )
    code_in  = 2 * _TOKENS_PER_ROUTINE_CODE_IN          # pyx + header
    code_out = 2 * _TOKENS_PER_ROUTINE_CODE_OUT

    reasoning_cost = estimate_cost_usd(_REASONING_MODEL, reasoning_in, reasoning_out)
    code_cost      = estimate_cost_usd(_CODE_MODEL, code_in, code_out)

    return FileEstimate(
        path=abspath,
        n_routines=n,
        routines=routine_names,
        risks=risks,
        prompt_tokens=reasoning_in + code_in,
        completion_tokens=reasoning_out + code_out,
        reasoning_cost_usd=reasoning_cost,
        code_cost_usd=code_cost,
    )


def estimate_paths(paths: Iterable[str]) -> CodebaseEstimate:
    """Walk paths, parse each Fortran file, sum the estimate."""
    files: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.rglob("*.[fF]90")))
        else:
            files.append(str(p))
    seen: set[str] = set()
    out_files = [f for f in files if not (f in seen or seen.add(f))]

    estimate = CodebaseEstimate(paths_scanned=list(paths))
    for f in out_files:
        estimate.files.append(estimate_file(f))
    return estimate


# ── Markdown rendering ──────────────────────────────────────────────────────

def render_markdown(est: CodebaseEstimate) -> str:
    """Render the estimate as a stakeholder-ready Markdown report."""
    lines: list[str] = []
    lines.append("# fortranspire — port-cost estimate")
    lines.append("")
    lines.append(f"Scanned **{len(est.files)} file(s)** under: "
                 f"`{', '.join(est.paths_scanned)}`")
    lines.append("")

    # Summary table at the top so a quick read gives the answer.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Files parsed successfully | **{est.n_files_ok}** / {len(est.files)} |")
    lines.append(f"| Routines detected | **{est.n_routines}** |")
    lines.append(f"| Estimated prompt tokens | {est.total_prompt_tokens:,} |")
    lines.append(f"| Estimated completion tokens | {est.total_completion_tokens:,} |")
    lines.append(f"| **Estimated LLM cost (one full port pass)** | **${est.total_cost_usd:.2f} USD** |")
    lines.append(f"| Reasoning model | `{_REASONING_MODEL}` |")
    lines.append(f"| Code-gen model | `{_CODE_MODEL}` |")
    lines.append("")
    lines.append("> Estimate is a **soft upper bound** calibrated on seismic-CPML and AROME-class "
                 "kernels. Real runs typically land 10-30% below this figure. Does **not** include "
                 "GPU validation time, optional `agent-doc` documentation pass, or human review.")
    lines.append("")

    # Per-file breakdown so the reader can target the expensive ones.
    if any(not f.parse_error for f in est.files):
        lines.append("## Per-file breakdown")
        lines.append("")
        lines.append("| File | Routines | Tokens (in+out) | Cost (USD) | Risks |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for f in est.files:
            if f.parse_error:
                continue
            tokens = f.prompt_tokens + f.completion_tokens
            risk_count = len(f.risks)
            risk_str = f"{risk_count} ({_risk_pill(f.risks)})" if risk_count else "—"
            rel = _rel(f.path)
            lines.append(f"| `{rel}` | {f.n_routines} | {tokens:,} | "
                         f"${f.total_cost_usd:.3f} | {risk_str} |")
        lines.append("")

    # Risk roll-up — what's going to bite the operator.
    all_risks = [(f, r) for f in est.files for r in f.risks]
    if all_risks:
        lines.append("## Risks (structural — surfaced by Loki, no LLM)")
        lines.append("")
        by_rule: dict[str, list[tuple[str, str, str]]] = {}
        for f, (rule, severity, detail) in all_risks:
            by_rule.setdefault(rule, []).append((_rel(f.path), severity, detail))
        for rule in sorted(by_rule):
            label = _RISK_LABELS.get(rule, rule)
            entries = by_rule[rule]
            top_sev = max(entries, key=lambda e: {"error": 3, "warning": 2, "note": 1}[e[1]])[1]
            lines.append(f"### `{rule}` — {label}  *({top_sev}, {len(entries)} occurrence{'s' if len(entries) != 1 else ''})*")
            lines.append("")
            for file_rel, severity, detail in entries[:20]:
                lines.append(f"- **{file_rel}**: {detail}")
            if len(entries) > 20:
                lines.append(f"- … {len(entries) - 20} more")
            lines.append("")

    # Parse failures get their own section so they don't get lost.
    failed = [f for f in est.files if f.parse_error]
    if failed:
        lines.append("## Parse failures")
        lines.append("")
        for f in failed:
            lines.append(f"- `{_rel(f.path)}` — {f.parse_error}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Report generated by **`agent-explain`** — no LLM was called, "
                 "no tokens were consumed during this estimate.")
    lines.append("")
    return "\n".join(lines)


def _risk_pill(risks: list[tuple[str, str, str]]) -> str:
    n_err = sum(1 for _, sev, _ in risks if sev == "error")
    n_warn = sum(1 for _, sev, _ in risks if sev == "warning")
    n_note = sum(1 for _, sev, _ in risks if sev == "note")
    parts = []
    if n_err:  parts.append(f"{n_err}E")
    if n_warn: parts.append(f"{n_warn}W")
    if n_note: parts.append(f"{n_note}N")
    return "/".join(parts) or "0"


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-explain",
        description=(
            "Pre-flight cost + risk estimate for a Fortran codebase. "
            "Runs the deterministic Loki parser only — zero LLM calls, zero tokens. "
            "Produces a Markdown report suitable for forwarding to a stakeholder "
            "before authorising a full porting run."
        ),
        epilog=(
            "Examples:\n"
            "  agent-explain src/                           # whole codebase\n"
            "  agent-explain src/seismic.f90                # single file\n"
            "  agent-explain --output estimate.md src/      # save report\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+",
                        help="Fortran files or directories to estimate")
    parser.add_argument("-o", "--output", default=None,
                        help="Write the Markdown report to this file (default: stdout)")
    args = parser.parse_args(argv)

    estimate = estimate_paths(args.paths)
    if not estimate.files:
        print("agent-explain: no .f90 / .F90 file found in the given paths",
              file=sys.stderr)
        return 2

    report = render_markdown(estimate)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"agent-explain: report saved to {args.output}")
        print(f"  {estimate.n_routines} routine(s), "
              f"estimated cost ${estimate.total_cost_usd:.2f} USD")
    else:
        print(report)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
