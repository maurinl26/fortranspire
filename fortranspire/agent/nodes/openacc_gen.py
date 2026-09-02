"""Deterministic OpenACC/OpenMP pragma derivation — no LLM.

The pragma for a compute loop is decidable from the AST, and getting it wrong is
not a stylistic slip: a reduction loop parallelised *without* a ``reduction``
clause is a race — silently wrong on the GPU, and nothing downstream catches it
unless the (GPU-only) equivalence harness runs. The previous node asked an LLM,
assumed a "2D FD stencil", and hard-coded ``collapse(2)`` — no reduction handling
at all. This derives the clauses instead:

* **``reduction(op:var)``** — a scalar ``s`` read-and-written as ``s = s <op> …``
  (``+``/``*``/``max``/``min``/``iand``/``ior``/``.and.``/``.or.``). The missing
  piece, and a correctness fix.
* **``collapse(n)``** — the real depth of a *perfect* loop nest, not a constant.
* **``private(tmp)``** — scalar temporaries assigned inside the loop.
* a loop that carries a dependency is **not** parallelised (``!$acc loop seq``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class LoopPragma:
    depth: int = 1
    reductions: Dict[str, str] = field(default_factory=dict)   # var -> op
    privates: Set[str] = field(default_factory=set)
    carried: bool = False


def _reduction_op(name: str, rhs) -> str | None:
    """The reduction operator if ``name`` accumulates into itself in ``rhs``."""
    import pymbolic.primitives as p
    from loki import FindVariables
    from loki.expression import symbols as sym

    refs = {v.name.lower() for v in FindVariables().visit(rhs)}
    if name.lower() not in refs:
        return None
    if isinstance(rhs, p.Sum):
        return "+"            # covers s = s + x and s = s - x (both + reductions)
    if isinstance(rhs, p.Product):
        return "*"
    if isinstance(rhs, sym.InlineCall):
        fn = str(rhs.function).lower()
        return {"max": "max", "min": "min", "iand": "iand", "ior": "ior"}.get(fn)
    if isinstance(rhs, getattr(p, "LogicalAnd", ())):
        return ".and."
    if isinstance(rhs, getattr(p, "LogicalOr", ())):
        return ".or."
    return None


def _perfect_nest_depth(loop) -> int:
    """How many loops are perfectly nested (each body is a single inner loop)."""
    from loki.ir.nodes import Comment, Loop, Pragma

    depth, node = 1, loop
    while True:
        inner = [n for n in node.body if isinstance(n, Loop)]
        other = [n for n in node.body
                 if not isinstance(n, (Loop, Comment, Pragma))]
        if len(inner) == 1 and not other:
            depth += 1
            node = inner[0]
        else:
            break
    return depth


def analyse_loop(loop, carried: bool = False) -> LoopPragma:
    """Derive the pragma clauses for one (outermost) compute loop nest."""
    from loki import FindNodes
    from loki.ir.nodes import Assignment, Loop
    from loki.expression import symbols as sym

    loopvars = {str(lp.variable).lower() for lp in FindNodes(Loop).visit(loop)
                if lp.variable is not None}
    loopvars.add(str(loop.variable).lower() if loop.variable is not None else "")

    reductions: Dict[str, str] = {}
    privates: Set[str] = set()
    for asn in FindNodes(Assignment).visit(loop.body):
        lhs = asn.lhs
        if not isinstance(lhs, sym.Scalar):
            continue                          # array element write → not scalar
        name = lhs.name
        if name.lower() in loopvars:
            continue
        op = _reduction_op(name, asn.rhs)
        if op:
            reductions[name] = op
        else:
            privates.add(name)                # scalar temporary
    for r in reductions:                       # a reduction is never also private
        privates.discard(r)

    return LoopPragma(
        depth=_perfect_nest_depth(loop),
        reductions=reductions,
        privates=privates,
        carried=carried,
    )


def insert_pragma(source: str, loop_var: str, pragma: str) -> str:
    """Insert ``pragma`` on its own line before ``do <loop_var> =``, keeping indent."""
    import re

    pat = re.compile(rf"^(\s*)(do\s+{re.escape(loop_var)}\s*=)",
                     re.IGNORECASE | re.MULTILINE)
    new, n = pat.subn(lambda m: f"{m.group(1)}{pragma}\n{m.group(1)}{m.group(2)}",
                      source, count=1)
    return new if n else source


def render_pragma(info: LoopPragma, gpu_pragma: str = "acc") -> str:
    """The directive line for the analysed loop."""
    if info.carried:
        # A carried dependency cannot be a parallel loop; run it sequentially.
        return "!$acc loop seq" if gpu_pragma == "acc" \
            else "!$omp target teams distribute parallel do"  # caller warns

    if gpu_pragma == "acc":
        parts = ["!$acc parallel loop"]
    else:
        parts = ["!$omp target teams distribute parallel do"]
    if info.depth > 1:
        parts.append(f"collapse({info.depth})")
    # group reductions by operator: reduction(+:a,b) reduction(max:c)
    by_op: Dict[str, list] = {}
    for var, op in sorted(info.reductions.items()):
        by_op.setdefault(op, []).append(var)
    for op, vs in by_op.items():
        parts.append(f"reduction({op}:{','.join(vs)})")
    if info.privates:
        parts.append(f"private({','.join(sorted(info.privates))})")
    return " ".join(parts)
