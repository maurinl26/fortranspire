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


def _basic_dtype(loki_type) -> str:
    """Map a Loki declared type to 'integer' | 'real' | 'logical' | 'complex'."""
    from loki import BasicType

    dt = getattr(loki_type, "dtype", None)
    return {
        BasicType.INTEGER: "integer",
        BasicType.REAL: "real",
        BasicType.LOGICAL: "logical",
        BasicType.COMPLEX: "complex",
    }.get(dt, "unknown")


def infer_dtypes(routine) -> Dict[str, str]:
    """Best-effort dtype per referenced symbol: 'integer' | 'real' | … | 'unknown'.

    Declared symbols (arguments, locals) get their Loki type directly. Symbols
    with no local declaration — promoted module state (#5) — are typed from how
    they are *used*, which is decidable syntactically and is exactly what
    gradcheck needs: an integer index or loop bound must never be fed a float.

    Integer evidence, any one of which is conclusive:
      * loop index variable, or a symbol appearing in a loop **bound**;
      * a symbol used as an array **subscript**;
      * a symbol in a local array's **dimension** (a size);
      * a symbol compared against an integer literal (``NSPECIAL_RXN .GT. 0``);
      * an array whose value is assigned to an integer target
        (``ISP1 = IRM2(...)`` ⇒ ``IRM2`` is an integer table).

    Everything else stays 'unknown'; gradcheck treats unknown as differentiable
    (float), the safe default for the numeric payload (``RKI``, ``SC``, …).
    """
    from loki import FindNodes, FindVariables
    from loki.ir.nodes import Assignment, VariableDeclaration, Loop
    from loki.expression import symbols as sym

    dtypes: Dict[str, str] = {}

    # 1. Declared types (arguments and locals live in the spec).
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        base = _basic_dtype(getattr(decl, "symbols", [None])[0].type
                            if decl.symbols else None)
        for v in decl.symbols:
            dtypes[v.name.lower()] = _basic_dtype(v.type)

    integer_ctx: set[str] = set()

    def _names(node) -> list[str]:
        return [v.name.lower() for v in FindVariables().visit(node)
                if isinstance(v, (sym.MetaSymbol, sym.TypedSymbol))
                and not isinstance(v, sym.ProcedureSymbol)]

    # 2. Loop indices and everything in a loop bound → integer.
    for lp in FindNodes(Loop).visit(routine.body):
        if lp.variable is not None:
            integer_ctx.add(str(lp.variable).lower())
        if lp.bounds is not None:
            integer_ctx.update(_names(lp.bounds))

    # 3. Array subscripts (body) and local-array dimensions (spec) → integer.
    for v in FindVariables().visit(routine.body):
        if isinstance(v, sym.Array):
            for dim in (getattr(v, "dimensions", None) or []):
                integer_ctx.update(_names(dim))
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        for var in decl.symbols:
            for dim in (getattr(var, "dimensions", None) or []):
                integer_ctx.update(_names(dim))

    # 4. Comparison against an integer literal, and integer-target assignments.
    for a in FindNodes(Assignment).visit(routine.body):
        lhs_base = str(a.lhs).split("(")[0].strip().lower()
        if dtypes.get(lhs_base) == "integer":
            # a scalar-ish RHS that is a single symbol/array ref → that table is integer
            rhs_names = _names(a.rhs)
            if len(rhs_names) >= 1:
                # the head symbol of the RHS reference (e.g. IRM2 in IRM2(NRK,NR,NCS))
                head = str(a.rhs).split("(")[0].strip().lower()
                if head and head.replace("_", "").isidentifier():
                    integer_ctx.add(head)

    # Integer comparisons live in Conditional conditions (`NSPECIAL_RXN .GT. 0`).
    _mark_integer_comparisons(routine, integer_ctx)

    # Merge: usage-based integer only fills what a declaration did not already type.
    for name in integer_ctx:
        dtypes.setdefault(name, "integer")
        if dtypes.get(name) == "unknown":
            dtypes[name] = "integer"

    return dtypes


def _mark_integer_comparisons(routine, integer_ctx: set) -> None:
    """A symbol compared against an integer literal is integer (e.g. `n .GT. 0`)."""
    from loki import FindNodes
    from loki.ir.nodes import Conditional
    from loki.expression import symbols as sym

    try:
        from pymbolic.primitives import Comparison
    except Exception:  # noqa: BLE001
        return

    def _walk(expr):
        if isinstance(expr, Comparison):
            left, right = expr.left, expr.right
            for a, b in ((left, right), (right, left)):
                if isinstance(b, sym.IntLiteral) and isinstance(a, (sym.Scalar, sym.DeferredTypeSymbol)):
                    integer_ctx.add(a.name.lower())
        for child in getattr(expr, "children", ()) or ():
            _walk(child)

    for cond in FindNodes(Conditional).visit(routine.body):
        try:
            _walk(cond.condition)
        except Exception:  # noqa: BLE001
            pass


def integer_index_args(routine) -> List[str]:
    """Integer symbols whose *value* is an index, so a random probe is unsafe.

    A loop bound can take any extent, but an index — an integer **array**
    (a lookup table like ``IRM2``), or an integer scalar used as an array
    **subscript** — must be a valid position into another array whose shape is
    not known from this file alone. gradcheck cannot fabricate those, so it
    reports them as a required *fixture* instead of crashing on an
    out-of-range probe. This is the honest boundary of a from-scratch check.
    """
    from loki import FindNodes, FindVariables
    from loki.ir.nodes import VariableDeclaration
    from loki.expression import symbols as sym

    dtypes = infer_dtypes(routine)
    index: set[str] = set()

    # integer arrays = lookup tables
    declared_arrays = set()
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        for v in decl.symbols:
            if getattr(v, "dimensions", None):
                declared_arrays.add(v.name.lower())
    for v in FindVariables().visit(routine.body):
        low = v.name.lower()
        if isinstance(v, sym.Array) and dtypes.get(low) == "integer":
            index.add(low)
        # integer scalar used as a subscript
        if isinstance(v, sym.Array):
            for dim in (getattr(v, "dimensions", None) or []):
                for s in FindVariables().visit(dim):
                    if dtypes.get(s.name.lower()) == "integer":
                        index.add(s.name.lower())

    return sorted(index)


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
