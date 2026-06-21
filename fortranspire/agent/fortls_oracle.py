"""`fortls` (Fortran Language Server) thin wrapper — used as an oracle for
symbol lookup, signature retrieval, and "does this name exist?" questions
that the LLM should not answer by hallucination.

The integration is deliberately minimal: we spawn `fortls` per query rather
than holding a long-lived LSP session. That keeps the calling code free of
asyncio plumbing at the price of ~150 ms per query — fine for the
documenter (which queries once per routine) and the analyzer extensions.

`fortls` lives in the ``[gpu]`` extra. Falls back to a no-op when missing
so analyze-only environments keep working.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str       # "module" | "subroutine" | "function" | "variable" | ...
    file: str
    line: int


def is_available() -> bool:
    """Return True when `fortls` is on PATH."""
    return shutil.which("fortls") is not None


def list_symbols(workspace: str | Path) -> list[Symbol]:
    """Index `workspace` and return every Fortran symbol fortls knows about.

    Uses `fortls --debug_workspace` which dumps a JSON representation of
    the indexed symbols. Returns an empty list if fortls is missing or
    the workspace contains no Fortran files.
    """
    if not is_available():
        return []
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        workspace = workspace.parent

    result = subprocess.run(
        ["fortls", "--debug_workspace", "--source_dirs", str(workspace)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if result.returncode != 0:
        return []

    # `--debug_workspace` output mixes status lines with a JSON blob.
    # We pull the JSON section conservatively (first `[` to last `]`).
    blob = result.stdout
    start = blob.find("[")
    end = blob.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        entries = json.loads(blob[start:end + 1])
    except json.JSONDecodeError:
        return []

    symbols: list[Symbol] = []
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        symbols.append(Symbol(
            name=name,
            kind=str(entry.get("kind", "?")),
            file=str(entry.get("path", "")),
            line=int(entry.get("line", 0)),
        ))
    return symbols


def symbol_index_by_name(workspace: str | Path) -> dict[str, list[Symbol]]:
    """Convenience: returns a `name -> [Symbol, ...]` map (one per hit)."""
    out: dict[str, list[Symbol]] = {}
    for sym in list_symbols(workspace):
        out.setdefault(sym.name.lower(), []).append(sym)
    return out


def is_known_symbol(workspace: str | Path, name: str) -> bool:
    """Quick lookup — `True` if `name` resolves anywhere in `workspace`."""
    return name.lower() in symbol_index_by_name(workspace)


def summarize_context(workspace: str | Path, routine_name: str) -> str:
    """Return a short text snippet describing every symbol declared in the
    same file or module as ``routine_name``. Useful as extra context fed
    to the LLM during documentation generation.
    """
    if not is_available():
        return ""
    syms = list_symbols(workspace)
    target = next((s for s in syms if s.name.lower() == routine_name.lower()), None)
    if target is None:
        return ""
    neighbors = [s for s in syms
                 if s.file == target.file and s.name.lower() != target.name.lower()]
    if not neighbors:
        return ""
    lines = [f"Symbols co-located with `{routine_name}` (via fortls):"]
    for s in neighbors[:50]:   # cap to keep prompts bounded
        lines.append(f"  - {s.kind} `{s.name}` (line {s.line})")
    return "\n".join(lines)
