"""Deterministic JAX skeleton for the cases we can *derive* — no LLM.

The robustness ceiling is the LLM emission (Mistral has no larger model). The
answer is to shrink what the LLM must do: derive the JAX structure from the loop
and expression trees, and only fall back to the model for what cannot be lowered
deterministically yet.

This first pass handles the most common — and, for a small model, the most
error-prone — shape: an **element-wise** loop nest. Every array subscript is a
bare loop index, so the whole nest is one whole-array `jnp` expression
(`c(i) = 2*a(i) + b(i)` → `c = 2.0*a + b`). No `jit` pitfalls, differentiable by
construction, and no token spent. Anything else — a stencil shift, a recurrence
(scan), a gather/scatter, an unknown intrinsic, a boundary assignment — returns
``None`` and the LLM path takes over. The target is JAX (`jnp`), not XLA: JAX
lowers to XLA under `jit` while keeping `jax.grad`, which is the whole point.
"""
from __future__ import annotations

from typing import Optional, Set

# Fortran intrinsic → JAX, binary/unary only (variadic MAX/MIN handled below).
_INTRINSIC = {
    "abs": "jnp.abs", "sqrt": "jnp.sqrt", "exp": "jnp.exp", "log": "jnp.log",
    "log10": "jnp.log10", "sin": "jnp.sin", "cos": "jnp.cos", "tan": "jnp.tan",
    "sinh": "jnp.sinh", "cosh": "jnp.cosh", "tanh": "jnp.tanh",
    "asin": "jnp.arcsin", "acos": "jnp.arccos", "atan": "jnp.arctan",
    "max": "jnp.maximum", "min": "jnp.minimum", "mod": "jnp.mod",
    "sign": "jnp.sign", "real": "", "dble": "",  # casts: identity in x64 JAX
}


def _fortran_float(text: str) -> str:
    """`2.0d0` / `1.5D-3` / `0.5_dp` → a Python float literal."""
    for suffix in ("_dp", "_sp", "_8", "_4"):    # strip KIND before the d→e swap
        text = text.replace(suffix, "")
    return text.replace("d", "e").replace("D", "e")


def _lower(expr, loopvars: Set[str]) -> Optional[str]:
    """Lower one Loki/pymbolic expression to a JAX (jnp) whole-array string."""
    import pymbolic.primitives as p
    from loki.expression import symbols as sym

    if isinstance(expr, sym.Array):
        for d in (expr.dimensions or ()):
            # Element-wise only: every subscript must be a bare loop index.
            if not (isinstance(d, sym.Scalar) and d.name.lower() in loopvars):
                return None
        return expr.name
    if isinstance(expr, sym.Scalar):
        return expr.name
    if isinstance(expr, sym.FloatLiteral):
        return _fortran_float(str(expr.value))
    if isinstance(expr, sym.IntLiteral):
        return str(expr.value)
    if isinstance(expr, sym.LogicLiteral):
        return "True" if str(expr.value).lower() in (".true.", "true") else "False"

    if isinstance(expr, p.Sum):
        parts = [_lower(c, loopvars) for c in expr.children]
        return None if any(x is None for x in parts) else "(" + " + ".join(parts) + ")"
    if isinstance(expr, p.Product):
        parts = [_lower(c, loopvars) for c in expr.children]
        return None if any(x is None for x in parts) else "(" + " * ".join(parts) + ")"
    if isinstance(expr, p.Quotient):
        n, d = _lower(expr.numerator, loopvars), _lower(expr.denominator, loopvars)
        return None if n is None or d is None else f"({n} / {d})"
    if isinstance(expr, p.Power):
        b, e = _lower(expr.base, loopvars), _lower(expr.exponent, loopvars)
        return None if b is None or e is None else f"({b} ** {e})"

    if isinstance(expr, sym.InlineCall):
        fn = str(expr.function).lower()
        args = [_lower(a, loopvars) for a in (expr.parameters or ())]
        if any(x is None for x in args):
            return None
        if fn in ("max", "min") and len(args) > 2:
            op = _INTRINSIC[fn]
            acc = args[0]
            for a in args[1:]:
                acc = f"{op}({acc}, {a})"
            return acc
        jax_fn = _INTRINSIC.get(fn)
        if jax_fn is None:
            return None
        if jax_fn == "":                        # identity cast (real/dble)
            return args[0] if len(args) == 1 else None
        return f"{jax_fn}({', '.join(args)})"

    return None  # unknown node → let the LLM handle it


def _parse(fortran_code: str):
    import os
    import tempfile

    try:
        from loki import Sourcefile
    except Exception:  # noqa: BLE001
        return None
    fd, tmp = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, fortran_code.encode())
        os.close(fd)
        src = Sourcefile.from_file(tmp)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    routines = list(src.routines) + [r for m in (src.modules or []) for r in (m.subroutines or [])]
    return routines[0] if routines else None


def lower_kernel(kernel: dict) -> Optional[str]:
    """A whole-array JAX function for an element-wise kernel, else ``None``."""
    # Promoted module state / index topology / recurrences are out of scope here.
    if kernel.get("free_reads") or kernel.get("free_writes"):
        return None
    if kernel.get("needs_scan") or kernel.get("has_loop_carried_dep"):
        return None

    routine = _parse(kernel.get("fortran_code") or "")
    if routine is None:
        return None

    from loki import FindNodes
    from loki.ir.nodes import Assignment, Loop
    from loki.expression import symbols as sym

    loops = FindNodes(Loop).visit(routine.body)
    if not loops:
        return None
    loopvars = {str(lp.variable).lower() for lp in loops if lp.variable is not None}

    outputs = [o for o in kernel.get("outputs", [])]
    out_lower = {o.lower() for o in outputs}

    body: list[str] = []
    produced: set[str] = set()
    for asn in FindNodes(Assignment).visit(routine.body):
        lhs = asn.lhs
        if not isinstance(lhs, sym.Array):
            return None                              # scalar / boundary write → LLM
        for d in (lhs.dimensions or ()):
            if not (isinstance(d, sym.Scalar) and d.name.lower() in loopvars):
                return None
        rhs = _lower(asn.rhs, loopvars)
        if rhs is None:
            return None
        body.append(f"    {lhs.name} = {rhs}")
        produced.add(lhs.name.lower())

    if not body or (out_lower - produced):
        return None

    args = ", ".join(kernel.get("inputs", []))
    lines = ["import jax.numpy as jnp", "", f"def {kernel['routine_name']}({args}):"]
    lines += body
    if len(outputs) == 1:
        lines.append(f"    return {outputs[0]}")
    else:
        lines.append(f"    return ({', '.join(outputs)})")
    return "\n".join(lines) + "\n"
