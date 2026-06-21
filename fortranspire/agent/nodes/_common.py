"""Tiny utilities shared by several nodes.

Keep this module dependency-free (stdlib only) so it can be imported
without dragging in Loki / LangChain / langgraph.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

SEP  = "─" * 64
SEP2 = "═" * 64


def _out(category: str = "fortran_gpu") -> Path:
    """Resolve and create the per-category output directory."""
    p = Path("output") / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(path: Path, content: str) -> None:
    """Write `content` to `path` and log the save."""
    path.write_text(content, encoding="utf-8")
    print(f"  Saved → {path}")


def _strip_markdown(code: str) -> str:
    """Pop the first ```lang block out of an LLM response."""
    match = re.search(r"```[a-zA-Z0-9]*\n?(.*?)\n?```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def _gpu_compiler() -> str | None:
    """Return the first available GPU-capable Fortran compiler, if any."""
    for compiler in ["nvfortran", "pgfortran"]:
        if shutil.which(compiler):
            return compiler
    env_fc = os.getenv("FC")
    if env_fc and shutil.which(env_fc):
        return env_fc
    return None
