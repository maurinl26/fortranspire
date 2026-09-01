"""Cross-module symbol resolution — Stage 0 of the semantic-resolution issue (#99).

`dataflow.free_symbols` (#5) finds the module state a routine reads; `#4` types
it syntactically and stops at an honest `needs_fixture` boundary when an integer
*index table* (`IRM2`) has no shape locally. The shapes live in the module that
declares the symbol — for CMAQ `RBFEVAL` that is `rbdata_mod.F` sitting right
next to it. This resolves them **on Loki**, no new frontend: parse the files
that define the `USE`d modules, read their declarations, and hand back each
symbol's dtype, rank and declared dimensions.

Scope is deliberately whole-program-lite: a routine's own directory is searched
by default (which resolves siblings like `RBDATA`); extra directories — a chosen
mechanism's `RXNS_DATA_MODULE.F90`, say — are passed in. When Loki's own
directory Scheduler is wired later this becomes its provider; for now it is a
direct, cached parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fortranspire.agent.nodes._common import collect_fortran_files

_MODULE_RE = re.compile(r"^\s*MODULE\s+(\w+)", re.IGNORECASE | re.MULTILINE)


@dataclass
class ResolvedSymbol:
    name: str
    dtype: str = "unknown"          # 'integer' | 'real' | 'logical' | 'complex'
    rank: int = 0
    dims: List[str] = field(default_factory=list)
    is_parameter: bool = False
    module: str = ""


@lru_cache(maxsize=32)
def _index_modules(search_dirs: tuple[str, ...]) -> Dict[str, str]:
    """Map module name (lower) → the file that defines it. Regex-cheap, no parse."""
    index: Dict[str, str] = {}
    for d in search_dirs:
        p = Path(d)
        files = collect_fortran_files([p]) if p.is_dir() else [str(p)]
        for f in files:
            try:
                text = Path(f).read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for m in _MODULE_RE.finditer(text):
                index.setdefault(m.group(1).lower(), f)
    return index


@lru_cache(maxsize=64)
def _module_symbols(path: str) -> Dict[str, ResolvedSymbol]:
    """Parse one file and return {symbol_lower: ResolvedSymbol} for its modules."""
    import tempfile, os

    from fortranspire.agent.nodes._preprocess import needs_preprocessing, preprocess
    from fortranspire.agent.dataflow import _basic_dtype

    try:
        from loki import Sourcefile, FindNodes
        from loki.ir.nodes import VariableDeclaration
    except Exception:  # noqa: BLE001
        return {}

    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    if needs_preprocessing(path):
        content, _ = preprocess(path, content)

    fd, tmp = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        source = Sourcefile.from_file(tmp)
    except Exception:  # noqa: BLE001
        return {}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    out: Dict[str, ResolvedSymbol] = {}
    for mod in (source.modules or []):
        for decl in FindNodes(VariableDeclaration).visit(mod.spec):
            for v in decl.symbols:
                dims = [str(d) for d in (getattr(v, "dimensions", None) or [])]
                out[v.name.lower()] = ResolvedSymbol(
                    name=v.name,
                    dtype=_basic_dtype(v.type),
                    rank=len(dims),
                    dims=dims,
                    is_parameter=bool(getattr(v.type, "parameter", False)),
                    module=mod.name,
                )
    return out


def resolve_modules(module_names: List[str], search_dirs: tuple[str, ...]) -> Dict[str, ResolvedSymbol]:
    """Resolve every symbol of the named modules found under ``search_dirs``."""
    index = _index_modules(tuple(search_dirs))
    resolved: Dict[str, ResolvedSymbol] = {}
    for name in module_names:
        path = index.get(name.lower())
        if path:
            resolved.update(_module_symbols(path))
    return resolved


def used_modules(routine) -> List[str]:
    """The module names a Loki routine imports via ``USE``."""
    from loki import FindNodes
    from loki.ir.nodes import Import

    names: List[str] = []
    for imp in FindNodes(Import).visit(routine.spec):
        mod = getattr(imp, "module", None)
        if mod:
            names.append(str(mod))
    return names


def resolve_for_routine(routine, search_dirs: tuple[str, ...]) -> Dict[str, ResolvedSymbol]:
    """Resolve the symbols a routine imports, from its ``USE`` modules."""
    return resolve_modules(used_modules(routine), tuple(search_dirs))


def default_search_dirs(routine_file: str, extra: Optional[List[str]] = None) -> tuple[str, ...]:
    """The routine's own directory, plus any explicit module directories."""
    dirs = [str(Path(routine_file).resolve().parent)]
    if extra:
        dirs.extend(extra)
    # de-dup, keep order
    seen: set[str] = set()
    return tuple(d for d in dirs if not (d in seen or seen.add(d)))
