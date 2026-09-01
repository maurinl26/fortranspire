"""Static lint for emitted JAX — catch a Python branch on traced data.

The classic emission bug (surfaced on CMAQ RBFEVAL once shapes resolved): the
model reads a scalar out of an array and branches on it with a Python `if` —

    nreact = NREACT[nrk]
    if nreact == 1: ...

`nreact` is a *traced* value under `jit`, so this raises a
`TracerBoolConversionError` — a cryptic trace deep in JAX. This lint finds it in
the source and reports the exact line and name, so the fix ("use `jnp.where` /
`lax.switch`") is obvious. It is deterministic and needs no execution.

The signal is precise: a name assigned from an **array subscript** (data), then
used in an `if` test. Indexing an array's *shape* / *size* is trace-time and is
excluded, so a legitimate `if x.shape[0] > 1` does not trip it.
"""
from __future__ import annotations

import ast
from typing import Dict, List

# Attribute accesses that yield a trace-time (concrete) value, not array data.
_STATIC_ATTRS = {"shape", "size", "ndim", "at"}


def _is_data_subscript(node: ast.AST) -> bool:
    """A Subscript that reads array *data* (not `.shape`/`.size`/`.at[...]`)."""
    if not isinstance(node, ast.Subscript):
        return False
    val = node.value
    if isinstance(val, ast.Attribute) and val.attr in _STATIC_ATTRS:
        return False
    return True


def _contains_data_subscript(node: ast.AST) -> bool:
    return any(_is_data_subscript(n) for n in ast.walk(node))


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _target_names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def data_dependent_branches(code: str) -> List[Dict]:
    """Return the Python `if` statements that branch on array-derived data.

    Each hit: ``{"line": int, "names": [str], "snippet": str}``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    hits: List[Dict] = []
    seen: set[tuple] = set()
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        tainted = _tainted_names(func)
        for node in ast.walk(func):
            if isinstance(node, ast.If):
                hit = _names(node.test) & tainted
                if hit and (node.lineno, tuple(sorted(hit))) not in seen:
                    seen.add((node.lineno, tuple(sorted(hit))))
                    try:
                        snippet = ast.unparse(node.test)
                    except Exception:  # noqa: BLE001 - py<3.9 or odd node
                        snippet = ""
                    hits.append({"line": node.lineno, "names": sorted(hit),
                                 "snippet": snippet})
    return hits


def _tainted_names(func: ast.FunctionDef) -> set[str]:
    """Names in ``func`` that carry array data (seed: array subscripts; then
    propagate through assignments to a fixed point)."""
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                rhs = node.value
                if _contains_data_subscript(rhs) or (_names(rhs) & tainted):
                    for tgt in node.targets:
                        for name in _target_names(tgt):
                            if name not in tainted:
                                tainted.add(name)
                                changed = True
    return tainted


# Modules the emitted kernel may import. A Fortran `USE` copied as a Python
# import (`from RXNS_FUNCTION import ...`) is a defect — the module does not
# exist and the file raises ModuleNotFoundError at load.
_ALLOWED_IMPORTS = {
    "jax", "jax.numpy", "jax.lax", "jax.scipy", "numpy", "math", "functools",
    "typing", "fortranspire.jax_smooth", "__future__",
}


def disallowed_imports(code: str) -> List[Dict]:
    """Imports of modules the emitted kernel must not pull in."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    hits: List[Dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in {m.split(".")[0] for m in _ALLOWED_IMPORTS}:
                    hits.append({"line": node.lineno, "module": alias.name})
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod and mod not in _ALLOWED_IMPORTS and mod.split(".")[0] not in {
                m.split(".")[0] for m in _ALLOWED_IMPORTS}:
                hits.append({"line": node.lineno, "module": mod})
    return hits


def data_dependent_loops(code: str) -> List[Dict]:
    """Python `for x in range(<array-derived value>)` — a data-dependent loop
    bound, which cannot be a concrete trip count under `jit`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    hits: List[Dict] = []
    seen: set[tuple] = set()
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        tainted = _tainted_names(func)
        for node in ast.walk(func):
            if isinstance(node, ast.For):
                it = node.iter
                # `range(...)` whose argument is array-derived (tainted or a
                # direct array subscript like `range(NUSERAT[NCSP])`).
                if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                        and it.func.id == "range"):
                    if any(_contains_data_subscript(a) or (_names(a) & tainted)
                           for a in it.args):
                        try:
                            snippet = ast.unparse(it)
                        except Exception:  # noqa: BLE001
                            snippet = "range(...)"
                        key = (node.lineno, snippet)
                        if key not in seen:
                            seen.add(key)
                            hits.append({"line": node.lineno, "snippet": snippet})
    return hits
