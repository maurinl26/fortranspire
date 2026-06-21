"""``fortranspire report`` — single-file HTML audit dashboard per kernel.

Closes issue #20.

After a Phase-1 port writes its artifacts under ``output/<kernel>/``,
clients want one thing they can open, scroll through, and sign off on.
This subcommand collects every artifact in the directory and stitches
them into a single self-contained HTML page:

  * Original Fortran source (syntax-highlighted via Pygments if
    available, else plain ``<pre>``)
  * Refactored GPU source with `!$acc` pragmas
  * Cython wrapper (`.pyx`) and the C header (`kernel_c.h`)
  * Validation log (gfortran + nvfortran outputs)
  * Equivalence harness (from #11) when present
  * Per-section collapsible via plain `<details>` — no JavaScript

No CDN, no external assets, no dependency beyond stdlib (Pygments is
optional). Drop the file anywhere, open in any browser, including
air-gapped environments — typical TotalEnergies / CEA classified
review.
"""
from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import CythonLexer, FortranLexer
    _PYGMENTS_OK = True
except ImportError:
    _PYGMENTS_OK = False


# Files to look for, in display order. Each entry: (slug, label, filename,
# lexer-name-for-Pygments-or-None).
_SECTIONS: list[tuple[str, str, str, str | None]] = [
    ("source",       "Original Fortran source",          "kernel.f90",                  "fortran"),
    ("module_pure",  "Extracted MODULE (PURE/ELEMENTAL)", "fortran_gpu/kernel_pure.f90", "fortran"),
    ("module_gpu",   "MODULE annotated with OpenACC",    "fortran_gpu/module_kernels_gpu.f90", "fortran"),
    ("driver_gpu",   "Driver with !$acc data region",    "fortran_gpu/driver_gpu.f90",  "fortran"),
    ("kernel_gpu",   "Full GPU source (driver + module)", "fortran_gpu/kernel_gpu.f90", "fortran"),
    ("cython_pyx",   "Cython wrapper (.pyx)",            "cython/*.pyx",                "cython"),
    ("cython_h",     "C header (iso_c_binding)",         "cython/kernel_c.h",           None),
    ("setup_py",     "scikit-build setup",               "setup.py",                    None),
    ("validation",   "Validation log",                   "fortran_gpu/validation.log",  None),
    ("equivalence",  "Equivalence test harness",         "tests/test_*_equivalence.py", None),
]


@dataclass
class Section:
    slug: str
    label: str
    path: str       # display path relative to the output root
    lexer: str | None
    content: str


@dataclass
class Report:
    output_root: str
    kernel_name: str
    sections: list[Section] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # labels of sections not found


# ── Collection ──────────────────────────────────────────────────────────────

def collect_sections(output_root: Path) -> Report:
    """Scan `output_root` and gather every artifact section that exists."""
    kernel_name = output_root.name or "kernel"
    report = Report(output_root=str(output_root), kernel_name=kernel_name)

    for slug, label, pattern, lexer in _SECTIONS:
        candidates = list(output_root.glob(pattern))
        if not candidates:
            report.missing.append(label)
            continue
        # If multiple matches (e.g. several .pyx), keep them all as separate sections.
        for path in sorted(candidates):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(path.relative_to(output_root)) if path.is_relative_to(output_root) else str(path)
            display_slug = slug if len(candidates) == 1 else f"{slug}_{path.stem}"
            report.sections.append(Section(
                slug=display_slug,
                label=label if len(candidates) == 1 else f"{label} — {path.name}",
                path=rel,
                lexer=lexer,
                content=content,
            ))
    return report


# ── Rendering ──────────────────────────────────────────────────────────────

def _highlight(content: str, lexer: str | None) -> str:
    """Syntax-highlight `content` if Pygments is available, else escape into <pre>."""
    if not _PYGMENTS_OK or not lexer:
        return f"<pre><code>{html.escape(content)}</code></pre>"
    lex = FortranLexer() if lexer == "fortran" else CythonLexer()
    formatter = HtmlFormatter(noclasses=True, nobackground=True, style="default")
    return highlight(content, lex, formatter)


def render_html(report: Report) -> str:
    """Render `report` as a single-file self-contained HTML page."""
    n_present = len(report.sections)
    n_total = len(_SECTIONS)
    title = f"fortranspire — audit report: {report.kernel_name}"

    body_parts: list[str] = []
    for section in report.sections:
        highlighted = _highlight(section.content, section.lexer)
        body_parts.append(
            f'<details open>'
            f'<summary><span class="slug">{html.escape(section.slug)}</span>'
            f' &mdash; {html.escape(section.label)} '
            f'<code class="path">{html.escape(section.path)}</code></summary>'
            f'<div class="content">{highlighted}</div>'
            f'</details>'
        )

    if report.missing:
        body_parts.append('<details><summary>Missing artifacts</summary><ul>')
        for label in report.missing:
            body_parts.append(f"<li>{html.escape(label)}</li>")
        body_parts.append("</ul></details>")

    body = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1   {{ font-size: 1.4em; margin-bottom: 0.1em; }}
  .header-meta {{ color: #555; font-size: 0.9em; margin-bottom: 1.5em; }}
  details {{ border: 1px solid #ccc; border-radius: 4px;
             margin: 0.6em 0; padding: 0; background: #fafafa; }}
  details[open] {{ background: #fff; }}
  summary {{ cursor: pointer; padding: 0.6em 1em; font-weight: 600;
             border-bottom: 1px solid #eee; background: #f4f4f4; }}
  summary:hover {{ background: #ececec; }}
  .slug    {{ display: inline-block; font-family: ui-monospace, monospace;
              background: #e0e0e0; padding: 0 0.4em; border-radius: 3px;
              font-size: 0.85em; margin-right: 0.4em; }}
  .path    {{ color: #888; font-size: 0.85em; margin-left: 0.6em;
              font-family: ui-monospace, monospace; }}
  .content {{ padding: 0.6em 1em; overflow-x: auto;
              font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.85em; }}
  pre, code {{ margin: 0; }}
  pre {{ white-space: pre-wrap; word-break: break-word; }}
  ul {{ margin: 0.4em 0 0.8em 1.4em; color: #777; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="header-meta">{n_present} / {n_total} artifact section(s) present.
Generated by <code>fortranspire report</code> — single-file, no external
assets, safe to email and to open from an air-gapped reviewer machine.</p>
{body}
</body>
</html>
"""


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fortranspire report",
        description=(
            "Generate a single-file self-contained HTML audit dashboard for "
            "a Phase-1 port output directory. Sections (original source, "
            "refactored MODULE, OpenACC, Cython wrapper, validation log, "
            "equivalence harness) are collected when present and embedded "
            "with syntax highlighting; missing sections are listed at the "
            "end so a reviewer knows what's not there."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fortranspire report output/kernel/                 # → stdout\n"
            "  fortranspire report -o review.html output/kernel/  # → file\n"
        ),
    )
    parser.add_argument("output_root",
                        help="Phase-1 output directory to audit "
                             "(e.g. output/ or output/<kernel_stem>/)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write the HTML report to this file (default: stdout)")
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    if not root.is_dir():
        print(f"fortranspire report: not a directory: {root}", file=sys.stderr)
        return 2

    report = collect_sections(root)
    if not report.sections:
        print(f"fortranspire report: no artifact files found under {root}",
              file=sys.stderr)
        return 2

    rendered = render_html(report)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"fortranspire report: {len(report.sections)} section(s) → {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
