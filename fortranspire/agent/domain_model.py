"""Typed domain model — scan Fortran, propose the typed Python equivalent.

Both functional targets need type and shape information the operator body
does not carry: gt4py.next needs named `Dimension`s, `Field` types,
`FieldOffset`s and a domain; JAX needs array shapes and dtypes. Today each
emitter lets the LLM re-derive that from the source. This module derives it
**once, deterministically, from the Loki AST**, target-agnostic, so both
emitters — and the GT4Py driver / halo work (issue #82) — read one typed
model instead of three guesses.

What Loki gives, per variable, that this turns into a typed spec:

* ``dtype`` (REAL/INTEGER/LOGICAL) + ``kind`` (8/4) → a Python dtype;
* ``dimensions`` → the rank and the extent expression of each axis;
* loop bounds (``do k = 2, nlev-1``) → which axis a loop iterates and its
  range;
* array subscripts (``t(k+1)``, ``t(k-1)``) relative to a loop variable →
  the constant offsets on that axis — the stencil shape, which feeds
  `FieldOffset`s and the halo thickness.

Axis *roles* (which axis is vertical vs horizontal) cannot be read off the
declarations — Fortran gives dimensions positionally — so they are inferred
by heuristic (naming, and which axis a recurrence runs over) and every
inferred role is flagged as such. A wrong role is a naming choice, not a
correctness bug: it changes `Dims[K]` to `Dims[Cell]`, which a human
confirms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ── dtype mapping ──────────────────────────────────────────────────────────
# (Fortran basic type, KIND) → the Python/gt4py/JAX dtype. KIND 8 is double,
# 4 is single; a bare REAL with no KIND is compiler-dependent (the FORT007
# finding) and defaults to float64 with the ambiguity flagged.
def _python_dtype(basic: str, kind: Optional[str]) -> str:
    basic = (basic or "").upper().replace("BASICTYPE.", "")
    k = str(kind) if kind is not None else None
    if basic == "REAL":
        if k in ("4", "sp", "SP", "real32"):
            return "float32"
        return "float64"
    if basic == "INTEGER":
        return "int32" if k in ("4", "int32") else "int64"
    if basic == "LOGICAL":
        return "bool"
    return "float64"  # deferred / unknown → the common case, flagged elsewhere


# field(table(idx, ...)) — a field indexed through another array (a
# neighbour table). The outer name is the field, the inner is the table.
_CONN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\(")

# Names that conventionally denote the vertical axis in NWP codes.
_VERTICAL_NAMES = re.compile(r"^(n?lev|k?lev|nz|klev|nflevg|klon_k|k)$", re.IGNORECASE)
# Names that conventionally denote a horizontal / column-packing axis.
_HORIZONTAL_NAMES = re.compile(r"^(n?ij|klon|nproma|ngptot|ncol|npoints|n|nx|ny)$", re.IGNORECASE)


@dataclass
class AxisSpec:
    """One dimension of a field."""

    extent: str                 # the Fortran extent expression, e.g. "nlev"
    role: str = "unknown"       # "vertical" | "horizontal" | "unknown"
    role_inferred: bool = True  # roles are always heuristic; never trust blindly
    offsets: frozenset = field(default_factory=frozenset)  # {-1, 0, 1} shifts seen

    @property
    def halo(self) -> int:
        """Halo thickness this axis needs: the largest |offset| read."""
        return max((abs(o) for o in self.offsets), default=0)


@dataclass
class FieldSpec:
    """A typed field (or scalar) argument."""

    name: str
    intent: str                 # IN | OUT | INOUT | ""
    dtype: str                  # float64 | float32 | int32 | bool
    axes: list[AxisSpec] = field(default_factory=list)
    dtype_ambiguous: bool = False   # REAL with no KIND (FORT007)

    @property
    def is_scalar(self) -> bool:
        return not self.axes

    @property
    def rank(self) -> int:
        return len(self.axes)


@dataclass
class ConnectivitySpec:
    """An unstructured neighbour access — the mesh model (FVM, icon4py).

    Detected from ``field(conn(n, c))``-style indirection. `table` is the
    Fortran neighbour table name (``e2c``, ``v2e``, a node-neighbour list);
    `arity` is the number of neighbours if it could be read from a bound,
    else None. This maps to a gt4py.next ``FieldOffset`` over a LOCAL
    dimension with an ``as_connectivity`` provider and a ``neighbor_sum``.
    """

    accessed_field: str         # the field read through the table, e.g. `cellval`
    table: str                  # the connectivity table, e.g. `e2c`
    arity: Optional[int] = None
    reduced: bool = False       # summed/reduced over the neighbours?


@dataclass
class DomainModel:
    """The typed domain of one routine — target-agnostic."""

    routine: str
    fields: list[FieldSpec] = field(default_factory=list)
    # Loop variable → axis extent it iterates, e.g. {"k": "nlev"}.
    loop_axes: dict[str, str] = field(default_factory=dict)
    # Unstructured neighbour accesses — the FVM / icon4py mesh model.
    connectivities: list["ConnectivitySpec"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_unstructured(self) -> bool:
        return bool(self.connectivities)

    def field(self, name: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def arrays(self) -> list[FieldSpec]:
        return [f for f in self.fields if not f.is_scalar]

    @property
    def max_halo(self) -> int:
        return max((a.halo for f in self.arrays for a in f.axes), default=0)


def _infer_role(extent: str, is_recurrence_axis: bool) -> str:
    """Heuristic axis role. Always paired with role_inferred=True."""
    stem = re.sub(r"^\d+\s*:\s*", "", extent).strip()  # drop a lower bound
    stem = re.split(r"[-+*/ ]", stem)[0]               # first identifier
    if is_recurrence_axis:
        return "vertical"          # a loop-carried dependency runs down a column
    if _VERTICAL_NAMES.match(stem):
        return "vertical"
    if _HORIZONTAL_NAMES.match(stem):
        return "horizontal"
    return "unknown"


def _offsets_for(source: str, array: str, loop_var: str, axis_index: int, rank: int) -> frozenset:
    """Constant offsets read on `array`'s `axis_index`, relative to `loop_var`.

    Scans subscripts like ``a(k+1, j)`` and records the ± integer on the
    axis whose index expression involves the loop variable.
    """
    offsets: set[int] = {0}
    # array( sub0, sub1, ... ) — capture the full subscript list, no nesting.
    pattern = re.compile(rf"\b{re.escape(array)}\s*\(([^()]*)\)")
    for match in pattern.finditer(source):
        subs = [s.strip() for s in match.group(1).split(",")]
        if len(subs) != rank or axis_index >= len(subs):
            continue
        sub = subs[axis_index]
        # Does this axis's subscript use the loop variable?
        m = re.match(rf"^{re.escape(loop_var)}\s*([+-]\s*\d+)?$", sub.replace(" ", ""))
        if m:
            shift = m.group(1)
            offsets.add(int(shift.replace(" ", "")) if shift else 0)
    return frozenset(offsets)


def build_domain_model(routine_source: str, *, routine_name: str = "") -> DomainModel:
    """Extract the typed domain model from one routine's Fortran source.

    Deterministic, Loki-based. Degrades to an empty model (with a note) if
    Loki cannot parse the snippet — the caller still gets a valid object.
    """
    model = DomainModel(routine=routine_name)

    try:
        import tempfile
        from pathlib import Path

        from loki import FindNodes, Sourcefile
        from loki.ir.nodes import Loop, VariableDeclaration
    except Exception as exc:  # noqa: BLE001 - loki missing/broken
        model.notes.append(f"loki unavailable ({type(exc).__name__}); no typed model")
        return model

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "routine.f90"
        path.write_text(routine_source, encoding="utf-8")
        try:
            sf = Sourcefile.from_file(path)
            routines = list(sf.all_subroutines)
        except Exception as exc:  # noqa: BLE001
            model.notes.append(f"loki parse failed ({type(exc).__name__}); no typed model")
            return model

    if not routines:
        model.notes.append("no routine parsed; no typed model")
        return model

    routine = routines[0]
    model.routine = model.routine or routine.name

    # Loop variable → the extent it iterates (from the loop bounds' upper).
    recurrence_axes: set[str] = set()
    for loop in FindNodes(Loop).visit(routine.body):
        var = str(getattr(loop, "variable", "")).strip()
        bounds = str(getattr(loop, "bounds", "")).strip()
        upper = bounds.split(":")[-1].strip() if ":" in bounds else bounds
        stem = re.split(r"[-+*/ ]", upper)[0].strip()
        if var:
            model.loop_axes[var] = stem

    # intents
    intents: dict[str, str] = {}
    if hasattr(routine, "arguments"):
        for v in routine.arguments:
            intent = getattr(v.type, "intent", None)
            if intent:
                intents[v.name] = intent.upper()

    # Per-variable typed spec.
    seen: set[str] = set()
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        for v in decl.symbols:
            if v.name in seen:
                continue
            seen.add(v.name)
            t = v.type
            basic = str(getattr(t, "dtype", ""))
            kind = getattr(t, "kind", None)
            dims = [str(d) for d in v.dimensions] if getattr(v, "dimensions", None) else []
            ambiguous = "REAL" in basic.upper() and kind is None

            axes: list[AxisSpec] = []
            for i, extent in enumerate(dims):
                # Which loop variable indexes this axis?
                loop_var = next(
                    (lv for lv, ext in model.loop_axes.items()
                     if _same_extent(ext, extent)),
                    None,
                )
                offsets = frozenset({0})
                if loop_var:
                    offsets = _offsets_for(routine.to_fortran(), v.name, loop_var, i, len(dims))
                is_recurrence = loop_var in recurrence_axes
                axes.append(AxisSpec(
                    extent=extent,
                    role=_infer_role(extent, is_recurrence),
                    offsets=offsets,
                ))

            # Only arguments carry an intent worth modelling; keep locals too
            # but mark intent "".
            model.fields.append(FieldSpec(
                name=v.name,
                intent=intents.get(v.name, ""),
                dtype=_python_dtype(basic, kind),
                axes=axes,
                dtype_ambiguous=ambiguous,
            ))

    # Unstructured neighbour accesses: field(table(node, c)) — the mesh model.
    body = routine.to_fortran()
    for m in _CONN_RE.finditer(body):
        outer, table = m.group(1), m.group(2)
        if outer == table or _same_extent(outer, table):
            continue
        # A reduction (sum over neighbours) if the access sits under SUM(...)
        # or accumulates in a loop — a light signal, refined by the emitter.
        reduced = bool(re.search(rf"sum\s*\([^)]*{re.escape(outer)}\s*\(",
                                 body, re.IGNORECASE))
        if not any(c.accessed_field == outer and c.table == table
                   for c in model.connectivities):
            model.connectivities.append(
                ConnectivitySpec(accessed_field=outer, table=table, reduced=reduced)
            )
    if model.connectivities:
        names = ", ".join(sorted({c.table for c in model.connectivities}))
        model.notes.append(
            f"unstructured connectivity access ({names}) — the mesh model "
            "(FVM / icon4py); maps to neighbor_sum over an as_connectivity table"
        )

    if any(a.role_inferred and a.role != "unknown" for f in model.arrays for a in f.axes):
        model.notes.append("axis roles are heuristic — confirm vertical vs horizontal")
    return model


def _same_extent(a: str, b: str) -> bool:
    """Loose match between a loop's upper-bound stem and a dimension extent."""
    norm = lambda s: re.sub(r"\s", "", re.sub(r"^\d+:", "", s)).lower()
    na, nb = norm(a), norm(b)
    return na == nb or na == re.split(r"[-+]", nb)[0] or nb == re.split(r"[-+]", na)[0]


# ── Target renderers — one typed model, two DSLs ───────────────────────────

_ROLE_TO_DIM = {"vertical": "K", "horizontal": "Cell"}


def to_gt4py_hints(model: "DomainModel") -> list[str]:
    """Deterministic gt4py.next hints: dimensions, field types, offsets, halo.

    These are facts, not guesses — the emitter should follow them rather
    than infer types from the source a second time.
    """
    if not model.arrays:
        return ["Scalars only — no fields to dimension."]

    hints: list[str] = []

    # Unstructured mesh access comes first — it is the primary model (FVM,
    # icon4py), not a Cartesian shift.
    if model.connectivities:
        for c in model.connectivities:
            hints.append(
                f"`{c.accessed_field}` is read through the connectivity "
                f"`{c.table}` — this is an UNSTRUCTURED mesh access (the FVM / "
                f"icon4py model). Declare a LOCAL neighbour dimension and a "
                f"`FieldOffset` over `{c.table.upper()}`, and reduce with "
                f"`neighbor_sum({c.accessed_field}({c.table.upper()}), "
                f"axis=<Local>Dim)`. Do NOT treat it as a Cartesian shift."
            )

    # Named dimensions to declare, from the axis roles.
    dims: dict[str, str] = {}
    for f in model.arrays:
        for a in f.axes:
            dim = _ROLE_TO_DIM.get(a.role, "Cell")
            dims[a.extent] = dim
    decl = ", ".join(f"{d} (from Fortran `{ext}`)" for ext, d in dims.items())
    hints.append(f"Declare these Dimensions: {decl}. Roles are heuristic — "
                 "flip vertical/horizontal if the physics says otherwise.")

    # Field type per array.
    for f in model.arrays:
        gdims = ", ".join(_ROLE_TO_DIM.get(a.role, "Cell") for a in f.axes)
        amb = " (REAL had no KIND — assumed float64)" if f.dtype_ambiguous else ""
        hints.append(f"`{f.name}`: gtx.Field[Dims[{gdims}], {f.dtype}]{amb}")

    # Offsets → FieldOffsets, and the halo they imply.
    for f in model.arrays:
        for a in f.axes:
            if a.halo > 0:
                dim = _ROLE_TO_DIM.get(a.role, "Cell")
                shifts = ", ".join(str(o) for o in sorted(a.offsets) if o != 0)
                hints.append(
                    f"`{f.name}` is shifted on {dim} by {{{shifts}}} -> declare "
                    f"`{dim}off = gtx.FieldOffset('{dim}off', source={dim}, "
                    f"target=({dim},))` and read `{f.name}({dim}off[n])`. "
                    f"Halo on {dim}: {a.halo} (driver concern, issue #82)."
                )
    return hints


def to_jax_hints(model: "DomainModel") -> list[str]:
    """Deterministic JAX hints: shapes and dtypes."""
    if not model.arrays:
        return ["Scalars only — no arrays to shape."]
    hints: list[str] = []
    for f in model.arrays:
        shape = ", ".join(a.extent for a in f.axes)
        amb = " (REAL had no KIND — assumed float64)" if f.dtype_ambiguous else ""
        hints.append(f"`{f.name}`: shape ({shape}), dtype {f.dtype}{amb}")
    if model.max_halo:
        hints.append(
            "Stencil shifts detected — a shifted read near a boundary needs "
            "slicing or padding; keep the array shape, do not flatten."
        )
    return hints


def domain_model_agent(state) -> dict:
    """Attach the typed domain model to each kernel. Deterministic, no LLM.

    Sits after `functionalize` and before either emitter, so both the JAX
    and gt4py.next emission read one typed model rather than re-deriving
    types from the source.
    """
    from fortranspire.agent.nodes._common import SEP

    print(f"\n{SEP}")
    print("  [Domain model] scanning types, dimensions and offsets — no LLM")
    print(SEP)

    updated = []
    for kernel in state.get("kernel_results", []):
        name = kernel.get("routine_name", "?")
        model = build_domain_model(kernel.get("fortran_code", ""), routine_name=name)
        arrays = len(model.arrays)
        if arrays:
            print(f"  {name:<26} {arrays} field(s), max halo {model.max_halo}")
        updated.append({**kernel, "domain_model": model})

    return {
        "kernel_results": updated,
        "executed_agents": state.get("executed_agents", []) + ["domain_model"],
    }
