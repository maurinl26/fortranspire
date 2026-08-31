"""C-preprocessor pass for Fortran sources that ask for one.

By long-standing Fortran convention an **uppercase** suffix — ``.F``,
``.F90``, ``.FOR`` — means "run this through cpp first"; the lowercase
forms mean "do not". Build systems honour that distinction, and legacy
sources rely on it heavily: 525 of CMAQ's Fortran files carry the
uppercase suffix and 199 of those contain live ``#ifdef`` blocks.

Without the pass, Loki reads the raw text, the ``#ifdef`` lines are not
Fortran, and the frontend returns **zero routines** — silently. The file
looks unparseable when it is merely unpreprocessed, and the analyzer
reports a parse failure rather than the actual cause.

Line numbers are preserved on purpose. Findings carry a line, SARIF turns
that into an annotation on GitHub, and the annotation points at the
*original* file. A plain ``cpp -P`` deletes the inactive branches and
shifts every subsequent line, so every annotation past the first
``#ifdef`` would land on the wrong statement. Here the removed lines are
blanked instead of deleted, so line *n* of the output is line *n* of the
input.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Uppercase suffixes are the ones that request preprocessing.
PREPROCESSED_SUFFIXES = frozenset({".F", ".F90", ".F95", ".F03", ".F08", ".FOR", ".FPP"})

# `# <line> "<file>"` — cpp's line marker, our anchor for realignment.
_LINE_MARKER = re.compile(r'^#\s+(\d+)\s+"([^"]*)"')

_TIMEOUT_SECONDS = 30


def needs_preprocessing(filepath: str | os.PathLike) -> bool:
    """Whether the suffix asks for cpp. Case-sensitive, deliberately."""
    return Path(filepath).suffix in PREPROCESSED_SUFFIXES


def _align_to_source(cpp_output: str, source_name: str, line_count: int) -> str:
    """Rebuild the output so line *n* is still line *n* of the input.

    cpp emits ``# <line> "<file>"`` markers whenever it jumps. Walking them
    tells us which original line each surviving line came from, so each one
    goes back to its own index and everything cpp removed stays blank.

    Content pulled in from *other* files is dropped: we are analysing one
    file, and inlining a header would break the line correspondence that is
    the whole point here.
    """
    lines = [""] * line_count
    current: int | None = None
    in_source = True

    for raw in cpp_output.splitlines():
        marker = _LINE_MARKER.match(raw)
        if marker:
            current = int(marker.group(1))
            in_source = Path(marker.group(2)).name == source_name
            continue
        if current is None:
            continue
        if in_source and 1 <= current <= line_count:
            lines[current - 1] = raw
        current += 1

    return "\n".join(lines)


def preprocess(filepath: str | os.PathLike, content: str) -> tuple[str, str | None]:
    """Run cpp over `content`, preserving line numbers.

    Returns ``(text, note)``. ``note`` is None on success, or a short
    explanation when the original content is returned unchanged — the
    caller surfaces it rather than letting a silent no-op look like a
    successful pass.
    """
    path = Path(filepath)
    if not needs_preprocessing(path):
        return content, None

    cpp = shutil.which("cpp") or shutil.which("gcc")
    if cpp is None:
        return content, (
            f"{path.name} has an uppercase suffix, so it expects the C "
            "preprocessor, but neither `cpp` nor `gcc` is on PATH. Any "
            "#ifdef block will reach the Fortran frontend verbatim."
        )

    # `-traditional-cpp` is not optional: without it cpp treats an
    # apostrophe in a comment ("don't") as an unterminated string literal
    # and mangles the file. `-fdirectives-only` is not usable here because
    # it leaves the directives in place, which is exactly the problem.
    argv = [cpp, "-E", "-traditional-cpp", str(path)] if cpp.endswith("gcc") \
        else [cpp, "-traditional-cpp", str(path)]

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return content, f"preprocessing {path.name} failed ({type(exc).__name__}); using raw source."

    if result.returncode != 0:
        # A missing `#include` is the usual cause and is not fatal: the
        # directives we care about are the conditionals in this file.
        detail = (result.stderr or "").strip().splitlines()
        first = detail[0][:160] if detail else f"exit {result.returncode}"
        return content, f"preprocessing {path.name} reported: {first}"

    aligned = _align_to_source(result.stdout, path.name, content.count("\n") + 1)
    if not aligned.strip():
        return content, f"preprocessing {path.name} produced an empty file; using raw source."
    return aligned, None
