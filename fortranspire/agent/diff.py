"""``fortranspire diff`` — semantic before/after diff viewer.

Closes issue #19.

Generic `diff` shows you every changed line. After a Phase-1 GPU port,
most of those changes are predictable transformations (added `!$acc`
pragma, stripped `PURE`/`ELEMENTAL`, moved a `COMMON` block to an
INTENT argument). Reviewing them line-by-line wastes time. This viewer
classifies each change into a small category vocabulary so reviewers
can focus on the surprising parts:

  [pragma]   added/removed OpenACC or OpenMP directive
  [purity]   added/removed PURE / ELEMENTAL keyword
  [refactor] subroutine signature change (arg list, INTENT, name)
  [type]     KIND / type-spec change (REAL → real(dp), etc.)
  [common]   COMMON block removed
  [save]     SAVE attribute removed
  [other]    everything else — these are the lines that need a human

Outputs colored terminal text by default; `--html` writes a single-file
self-contained HTML page suitable for emailing or attaching to a PR
review.
"""
from __future__ import annotations

import argparse
import difflib
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Category vocabulary — ordered so earlier patterns take precedence when
# multiple match (e.g. a `!$acc` line wins over `[other]`).
_CATEGORIES = ("pragma", "purity", "refactor", "type", "common", "save", "other")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pragma",   re.compile(r"^\s*!\$(acc|omp)\b",                    re.IGNORECASE)),
    ("purity",   re.compile(r"\b(PURE|ELEMENTAL)\b",                  re.IGNORECASE)),
    ("type",     re.compile(r"\b(real\s*\(\s*\w+\s*\)|integer\s*\(\s*\w+\s*\)|"
                            r"selected_real_kind|kind\s*=)",          re.IGNORECASE)),
    ("common",   re.compile(r"\bCOMMON\s*/",                          re.IGNORECASE)),
    ("save",     re.compile(r",\s*SAVE\s*::|^\s*SAVE\s",              re.IGNORECASE)),
    ("refactor", re.compile(r"^\s*(SUBROUTINE|FUNCTION)\b|INTENT\s*\(", re.IGNORECASE)),
]


def classify_line(line: str) -> str:
    """Return one of the category strings for a single source line."""
    for label, pattern in _PATTERNS:
        if pattern.search(line):
            return label
    return "other"


# ── Diff data model ─────────────────────────────────────────────────────────

@dataclass
class DiffEntry:
    kind: str            # "+", "-", " " (unchanged), or "?" (intra-line)
    line: str
    category: str        # one of _CATEGORIES (only meaningful when kind in "+-")
    a_lineno: int | None = None
    b_lineno: int | None = None


@dataclass
class DiffReport:
    before_path: str
    after_path: str
    entries: list[DiffEntry] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)   # category → count

    @property
    def total_changes(self) -> int:
        return sum(1 for e in self.entries if e.kind in ("+", "-"))


def compute_diff(before: str, after: str, *, a_path: str, b_path: str) -> DiffReport:
    """Build a category-annotated unified diff between two source strings."""
    a_lines = before.splitlines()
    b_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)

    report = DiffReport(before_path=a_path, after_path=b_path)
    counts: dict[str, int] = {c: 0 for c in _CATEGORIES}

    a_i = b_i = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                report.entries.append(DiffEntry(
                    kind=" ", line=a_lines[i1 + k], category="other",
                    a_lineno=i1 + k + 1, b_lineno=j1 + k + 1,
                ))
        else:
            for k in range(i2 - i1):
                cat = classify_line(a_lines[i1 + k])
                counts[cat] += 1
                report.entries.append(DiffEntry(
                    kind="-", line=a_lines[i1 + k], category=cat,
                    a_lineno=i1 + k + 1,
                ))
            for k in range(j2 - j1):
                cat = classify_line(b_lines[j1 + k])
                counts[cat] += 1
                report.entries.append(DiffEntry(
                    kind="+", line=b_lines[j1 + k], category=cat,
                    b_lineno=j1 + k + 1,
                ))
    report.counts = counts
    return report


# ── Rendering ──────────────────────────────────────────────────────────────

_ANSI = {
    "reset":    "\033[0m",
    "dim":      "\033[2m",
    "bold":     "\033[1m",
    "add":      "\033[32m",   # green
    "del":      "\033[31m",   # red
    "pragma":   "\033[36m",   # cyan
    "purity":   "\033[35m",   # magenta
    "refactor": "\033[33m",   # yellow
    "type":     "\033[34m",   # blue
    "common":   "\033[91m",   # bright red
    "save":     "\033[93m",   # bright yellow
    "other":    "",
}


def render_text(report: DiffReport, *, use_color: bool = True) -> str:
    """Render `report` as ANSI-colored unified diff with [category] tags."""
    def c(key: str, s: str) -> str:
        return f"{_ANSI[key]}{s}{_ANSI['reset']}" if use_color and _ANSI.get(key) else s

    lines: list[str] = []
    lines.append(c("bold", f"--- {report.before_path}"))
    lines.append(c("bold", f"+++ {report.after_path}"))
    lines.append("")
    lines.append(c("dim", _summary_line(report)))
    lines.append("")

    last_was_eq = False
    eq_skipped = 0
    for entry in report.entries:
        if entry.kind == " ":
            last_was_eq = True
            eq_skipped += 1
            continue
        if last_was_eq and eq_skipped > 0:
            lines.append(c("dim", f"  … {eq_skipped} unchanged line(s) …"))
            eq_skipped = 0
            last_was_eq = False

        tag = f"[{entry.category}]"
        sign = "+" if entry.kind == "+" else "-"
        coloured_tag = c(entry.category, tag) if entry.category != "other" else c("dim", tag)
        coloured_line = c("add" if sign == "+" else "del", f"{sign} {entry.line}")
        lines.append(f"  {coloured_tag:<14} {coloured_line}")

    if eq_skipped > 0:
        lines.append(c("dim", f"  … {eq_skipped} unchanged line(s) …"))
    return "\n".join(lines)


def _summary_line(report: DiffReport) -> str:
    pairs = ", ".join(
        f"{cat}={count}" for cat, count in report.counts.items() if count
    ) or "no changes"
    return f"# {report.total_changes} changed line(s)  ({pairs})"


def render_html(report: DiffReport) -> str:
    """Render `report` as a self-contained single-file HTML page.

    No external assets — inlined `<style>` with the same colour palette
    as the terminal output. Safe to email, attach to a PR, or open from
    a USB stick on an air-gapped reviewer machine.
    """
    rows: list[str] = []
    last_was_eq = False
    eq_count = 0
    for entry in report.entries:
        if entry.kind == " ":
            last_was_eq = True
            eq_count += 1
            continue
        if last_was_eq and eq_count > 0:
            rows.append(f'<tr class="skip"><td colspan="3">'
                        f'… {eq_count} unchanged line(s) …</td></tr>')
            eq_count = 0
            last_was_eq = False
        sign_cls = "add" if entry.kind == "+" else "del"
        rows.append(
            f'<tr class="row {sign_cls}">'
            f'<td class="tag {entry.category}">{entry.category}</td>'
            f'<td class="sign">{entry.kind}</td>'
            f'<td class="code">{html.escape(entry.line)}</td>'
            f'</tr>'
        )
    if eq_count > 0:
        rows.append(f'<tr class="skip"><td colspan="3">'
                    f'… {eq_count} unchanged line(s) …</td></tr>')

    summary = _summary_line(report)
    body = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fortranspire diff — {html.escape(Path(report.after_path).name)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  .summary {{ background: #f4f4f4; padding: 0.6em 1em; border-radius: 4px;
              font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.95em; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1em;
           font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.85em; }}
  td {{ padding: 2px 8px; vertical-align: top; }}
  .tag {{ width: 90px; text-align: center; color: #fff; border-radius: 3px;
          font-size: 0.75em; padding: 1px 4px; }}
  .sign {{ width: 1em; text-align: center; }}
  .code {{ white-space: pre-wrap; word-break: break-all; }}
  tr.add {{ background: #eaffea; }}
  tr.del {{ background: #ffeaea; }}
  tr.skip td {{ color: #888; padding: 0.4em 0; text-align: center; font-style: italic; }}
  .pragma   {{ background: #007aa3; }}
  .purity   {{ background: #8e44ad; }}
  .refactor {{ background: #c08400; }}
  .type     {{ background: #2c3e90; }}
  .common   {{ background: #c0392b; }}
  .save     {{ background: #b59100; }}
  .other    {{ background: #888; }}
</style>
</head>
<body>
<h1>fortranspire diff</h1>
<p><code>--- {html.escape(report.before_path)}</code><br>
   <code>+++ {html.escape(report.after_path)}</code></p>
<p class="summary">{html.escape(summary)}</p>
<table>
{body}
</table>
</body>
</html>
"""


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fortranspire diff",
        description=(
            "Semantic before/after diff for Fortran sources. Each changed "
            "line is tagged with a category — [pragma], [purity], [type], "
            "[refactor], [common], [save], [other] — so reviewers can "
            "skim the predictable transformations and focus on the "
            "lines that need a human."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fortranspire diff src/kernel.f90 output/fortran_gpu/kernel_gpu.f90\n"
            "  fortranspire diff --html -o review.html before.f90 after.f90\n"
        ),
    )
    parser.add_argument("before", help="Path to the original source")
    parser.add_argument("after",  help="Path to the transformed source")
    parser.add_argument("--html", action="store_true",
                        help="Emit a self-contained HTML page instead of terminal text")
    parser.add_argument("-o", "--output", default=None,
                        help="Write to a file instead of stdout")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors in text output")
    args = parser.parse_args(argv)

    try:
        before = Path(args.before).read_text(encoding="utf-8")
        after  = Path(args.after).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"fortranspire diff: {exc}", file=sys.stderr)
        return 2

    report = compute_diff(before, after, a_path=args.before, b_path=args.after)

    if args.html:
        rendered = render_html(report)
    else:
        use_color = not args.no_color and sys.stdout.isatty()
        rendered = render_text(report, use_color=use_color)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"fortranspire diff: {report.total_changes} changed line(s), "
              f"saved to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
