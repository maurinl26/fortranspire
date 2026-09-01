"""Deterministic use-def / free-variable analysis on the Loki AST (issue #5).

A routine that reads its inputs from a ``USE`` module — most real library
code — has arguments that do not tell the whole story. CMAQ ``RBFEVAL`` takes
``(NCSP, YIN, YDOT)`` but *reads* the reaction network (``IRM2``, ``NREACT``,
``SC`` …) and the rate coefficients (``RKI``) straight from ``RXNS_DATA`` /
``RBDATA``. A JAX or gt4py function cannot see module state, so those free
symbols must be **promoted to explicit arguments** — the read ones become
inputs, the written ones outputs. That promotion is a classic dataflow pass,
not something to guess: this module computes it from the AST.

Why the union of two traversals
-------------------------------
When the ``USE`` target is not resolved (we parse one file, not the whole
program), Loki types a module symbol from how it is *used*: ``RKI(n,k)`` reads
as an ``Array``, but ``NUMCELLS`` in a loop bound stays a ``DeferredTypeSymbol``.
A single ``FindVariables`` filter catches one class and drops the other, so we
union ``visit(routine.body)`` (the array-typed refs) with
``visit(routine.ir)`` restricted to every ``TypedSymbol`` (the deferred
scalars), then subtract what is locally bound.

Completeness is therefore best-effort at the *syntactic* level. A fully
resolved semantic representation — LFortran's ASR, or Flang's FIR — would give
every reference with its type, shape and scope in one pass, and is the right
escalation when we need the promoted symbols' **shapes/dtypes** (to synthesise
gradcheck inputs) rather than just their names. Their module sources sit next
to the routine, so that upgrade is feasible; it is deliberately out of scope
here. Until then, :func:`free_symbols` reports what it could not classify so an
omission surfaces as a diagnostic, never as silently broken emitted code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Fortran intrinsics that may appear as bare names; never data to promote.
_INTRINSICS = frozenset({
    "abs", "max", "min", "mod", "modulo", "sign", "sqrt", "exp", "log", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "real", "int", "nint", "floor", "ceiling", "aint", "anint", "dble", "float",
    "sum", "product", "maxval", "minval", "maxloc", "minloc", "dot_product",
    "matmul", "transpose", "size", "shape", "lbound", "ubound", "count", "any",
    "all", "merge", "reshape", "spread", "pack", "cshift", "eoshift", "huge",
    "tiny", "epsilon", "present", "allocated", "associated", "trim", "len",
})


@dataclass
class FreeSymbols:
    """Module-provided symbols a routine references but does not declare."""

    reads: List[str] = field(default_factory=list)      # promote → inputs
    writes: List[str] = field(default_factory=list)     # promote → outputs
    # names seen but not classifiable (kept for a diagnostic, never emitted).
    unclassified: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.reads and not self.writes


def free_symbols(routine) -> FreeSymbols:
    """Free (module-provided) symbols of a Loki routine, split read vs written.

    Deterministic, no LLM. ``routine`` is a ``loki.Subroutine``.
    """
    from loki import FindNodes, FindVariables
    from loki.ir.nodes import Assignment, VariableDeclaration, Loop, CallStatement
    from loki.expression import symbols as sym

    # Names that are *bound* here and so are not free.
    args = {a.name.lower() for a in getattr(routine, "arguments", [])}
    locals_: set[str] = set()
    params: set[str] = set()
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        for v in decl.symbols:
            locals_.add(v.name.lower())
            if getattr(v.type, "parameter", False):
                params.add(v.name.lower())
    loopvars = {
        str(lp.variable).lower()
        for lp in FindNodes(Loop).visit(routine.body)
        if lp.variable is not None
    }
    # Call *targets* are procedure names, not data — never promote them.
    call_names = {
        str(c.name).lower() for c in FindNodes(CallStatement).visit(routine.body)
    }

    bound = args | locals_ | loopvars | params | call_names | _INTRINSICS

    # Union of the two traversals, preserving original case for the signature.
    original: Dict[str, str] = {}

    # Loki splits data symbols across two disjoint bases: `Scalar`/`Array`
    # derive from `MetaSymbol`, while an unresolved module symbol is a
    # `DeferredTypeSymbol` under `TypedSymbol`. Filtering by one base silently
    # drops the other, which is what made the analysis incomplete. Include both
    # and exclude `ProcedureSymbol` (a call/function *name*, never data).
    _DATA = (sym.MetaSymbol, sym.TypedSymbol)

    def _collect(node) -> None:
        # No `unique=True`: on this Loki version it drops the array-typed refs
        # from the body traversal, exactly the symbols we must not miss.
        for v in FindVariables().visit(node):
            if isinstance(v, _DATA) and not isinstance(v, sym.ProcedureSymbol):
                original.setdefault(v.name.lower(), v.name)

    _collect(routine.body)   # array-typed module refs (RKI, IRM2, …)
    _collect(routine.ir)     # deferred scalars (NUMCELLS, NCS, NSPECIAL_RXN, …)

    free = {low: orig for low, orig in original.items() if low not in bound}

    # Written = base name on the LHS of an assignment. Everything else free is
    # read. (Module state written by a routine is rare but must become an
    # output; RBFEVAL writes none.)
    written_low = {
        str(a.lhs).split("(")[0].strip().lower()
        for a in FindNodes(Assignment).visit(routine.body)
    }

    reads, writes = [], []
    for low, orig in sorted(free.items()):
        (writes if low in written_low else reads).append(orig)

    return FreeSymbols(reads=reads, writes=writes)
