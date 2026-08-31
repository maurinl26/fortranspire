"""Tiny utilities shared by several nodes.

Keep this module dependency-free (stdlib only) so it can be imported
without dragging in Loki / LangChain / langgraph.
"""
from __future__ import annotations

import os
import re
import shutil
from contextvars import ContextVar
from pathlib import Path

SEP  = "─" * 64
SEP2 = "═" * 64

# Per-thread / per-task output root, set by `agent-port-batch` so concurrent
# kernel ports don't clobber each other's `output/fortran_gpu/` and
# `output/cython/` directories. `None` = use the default `Path("output")`
# (legacy single-port behavior).
#
# ContextVar is thread-local in Python: each worker thread sees `None` until
# it explicitly calls `set_output_root()` at the top of its task function.
_OUTPUT_ROOT: ContextVar[Path | None] = ContextVar("_OUTPUT_ROOT", default=None)


def set_output_root(root: Path | str | None) -> None:
    """Override the output root for the calling thread / async task.

    Used by ``fortranspire.agent.batch`` to isolate per-file outputs when
    porting many kernels in parallel. Pass ``None`` to revert to default.
    """
    _OUTPUT_ROOT.set(Path(root) if root is not None else None)


def get_output_root() -> Path:
    """Resolve the active output root (contextvar > env > default `output/`)."""
    root = _OUTPUT_ROOT.get()
    if root is not None:
        return root
    return Path(os.getenv("FORTRANSPIRE_OUTPUT_ROOT", "output"))


def _out(category: str = "fortran_gpu") -> Path:
    """Resolve and create the per-category output directory.

    Output root resolution (first hit wins):
      1. ``set_output_root(...)`` call active in the current thread/task
      2. ``FORTRANSPIRE_OUTPUT_ROOT`` env var
      3. literal ``output/`` (legacy default)
    """
    p = get_output_root() / category
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


# ── Fortran source discovery ───────────────────────────────────────────────
# Every verb that walks a directory used to spell its own `*.[fF]90` glob,
# in six places. All six missed the fixed-form suffixes, which is most of
# the legacy corpus this project exists for: 525 of CMAQ's Fortran files
# are `.F`, and `fortranspire explain` on that tree reported "no .f90 /
# .F90 file found".
#
# Case matters. An uppercase suffix additionally means "run cpp first"
# (see `nodes/_preprocess.py`), so the two sets are listed separately
# rather than globbed case-insensitively.

FREE_FORM_SUFFIXES = (".f90", ".F90", ".f95", ".F95", ".f03", ".F03", ".f08", ".F08")
FIXED_FORM_SUFFIXES = (".f", ".F", ".for", ".FOR", ".ftn", ".FTN", ".fpp", ".FPP")
FORTRAN_SUFFIXES = FREE_FORM_SUFFIXES + FIXED_FORM_SUFFIXES


def collect_fortran_files(paths) -> list[str]:
    """Expand `paths` into a sorted, de-duplicated list of Fortran sources.

    A path that is a file is taken as-is whatever its suffix — an explicit
    argument is an explicit choice. A directory is walked for every known
    Fortran suffix.
    """
    seen: dict[str, None] = {}

    for raw in paths:
        path = Path(raw)
        if path.is_file():
            seen[str(path)] = None
        elif path.is_dir():
            for suffix in FORTRAN_SUFFIXES:
                for found in path.rglob(f"*{suffix}"):
                    if found.is_file():
                        seen[str(found)] = None

    return sorted(seen)
