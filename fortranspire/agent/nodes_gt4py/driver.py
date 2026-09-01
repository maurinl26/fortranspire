"""GT4Py driver generation + static domain/halo validation (issue #82).

A field operator says *what* to compute at a point; it says nothing about
the geometric **domain** — the range each dimension is written over, the
halo a shifted read needs, and the offset providers. That lives in the
driver, and it is where the real bugs are: an operator that reads
``a(Koff[1])`` at the top layer reads out of bounds unless the domain is
restricted to the interior, or the field carries a ghost point.

None of that is visible to the frontend type check (`type_check`), so it is
handled here, and — crucially — **deterministically from the typed domain
model** (issue #84). The model already carries, per axis, the stencil
offsets and the halo they imply, so the driver's domain restriction and the
provider wiring are generated, not guessed.

On *executing* the driver: a Cartesian-offset operator does not run on the
embedded backend (it wants a neighbour table), and the compiled ``gtfn``
backend needs a full C++ toolchain. So this module **generates and
statically validates** the driver; running it end to end is a follow-up
that depends on a gtfn toolchain being present. The static checks catch the
domain/halo mistakes without needing to run anything.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fortranspire.agent.domain_model import DomainModel, FieldSpec

_ROLE_TO_DIM = {"vertical": "K", "horizontal": "Cell"}


@dataclass
class DomainReport:
    """The outcome of the static domain/halo check on one operator."""

    ok: bool = True
    halo: dict[str, int] = field(default_factory=dict)   # dim -> halo thickness
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _axis_dim(axis) -> str:
    return _ROLE_TO_DIM.get(axis.role, "Cell")


def interior_domain(model: "DomainModel") -> dict[str, tuple[str, str]]:
    """The domain each dimension must be restricted to, from the offsets.

    An axis whose reads span offsets in ``[-lo, +hi]`` can only be computed
    on ``[lo, n - hi)`` without reading out of bounds (0-based, half-open).
    Returns ``{dim: (lower, upper_expr)}`` with symbolic extents.
    """
    per_dim_lo: dict[str, int] = {}
    per_dim_hi: dict[str, int] = {}
    per_dim_extent: dict[str, str] = {}

    for f in model.arrays:
        for axis in f.axes:
            dim = _axis_dim(axis)
            lo = max((-o for o in axis.offsets if o < 0), default=0)
            hi = max((o for o in axis.offsets if o > 0), default=0)
            per_dim_lo[dim] = max(per_dim_lo.get(dim, 0), lo)
            per_dim_hi[dim] = max(per_dim_hi.get(dim, 0), hi)
            per_dim_extent.setdefault(dim, axis.extent)

    domain: dict[str, tuple[str, str]] = {}
    for dim, extent in per_dim_extent.items():
        lo = per_dim_lo.get(dim, 0)
        hi = per_dim_hi.get(dim, 0)
        upper = f"n_{dim}" if hi == 0 else f"n_{dim} - {hi}"
        domain[dim] = (str(lo), upper)
    return domain


def field_offsets_used(model: "DomainModel") -> dict[str, int]:
    """The FieldOffsets an operator over this model needs: {dim: halo}."""
    halo: dict[str, int] = {}
    for f in model.arrays:
        for axis in f.axes:
            if axis.halo > 0:
                dim = _axis_dim(axis)
                halo[dim] = max(halo.get(dim, 0), axis.halo)
    return halo


def build_driver(model: "DomainModel", operator_name: str) -> str:
    """Generate a gt4py.next driver (`program`-style) from the typed model.

    Deterministic: the dimensions, dtypes, field allocations, offset
    providers, and the interior domain all come from the domain model.
    """
    dims: dict[str, str] = {}
    for f in model.arrays:
        for axis in f.axes:
            dims[_axis_dim(axis)] = axis.extent

    halo = field_offsets_used(model)
    domain = interior_domain(model)

    lines: list[str] = []
    lines.append("import gt4py.next as gtx")
    lines.append("from gt4py.next import Dims  # noqa: F401")
    lines.append("import numpy as np")
    lines.append(f"from {operator_name}_module import {operator_name}")
    lines.append("")
    lines.append("# Dimensions (roles are heuristic — confirm vertical vs horizontal).")
    for dim, extent in dims.items():
        kind = "gtx.DimensionKind.VERTICAL" if dim == "K" else "gtx.DimensionKind.HORIZONTAL"
        lines.append(f'{dim} = gtx.Dimension("{dim}", kind={kind})   # Fortran `{extent}`')
    for dim in halo:
        lines.append(
            f'{dim}off = gtx.FieldOffset("{dim}off", source={dim}, target=({dim},))'
        )
    lines.append("")

    # Driver function.
    inputs = [f for f in model.arrays if "IN" in f.intent] + \
             [f for f in model.fields if f.is_scalar and "IN" in f.intent]
    outputs = [f for f in model.arrays if "OUT" in f.intent]
    size_args = ", ".join(f"n_{d}" for d in dims)

    arg_list = ", ".join(f.name for f in inputs + outputs)
    lines.append(f"def run_{operator_name}({arg_list}, *, {size_args}):")
    lines.append('    """Allocate, wire the offset providers, restrict the domain, run."""')
    # offset provider
    if halo:
        provs = ", ".join(f'"{d}off": gtx.CartesianConnectivity({d})' for d in halo)
        lines.append(f"    offset_provider = {{{provs}}}")
        lines.append("    # A Cartesian shift executes on the gtfn backend, not embedded (#82).")
    else:
        lines.append("    offset_provider = {}")
    # domain
    if domain:
        dom = ", ".join(f"{d}: ({lo}, {hi})" for d, (lo, hi) in domain.items())
        lines.append(f"    domain = {{{dom}}}")
        lines.append("    # Restricted to the interior so a shifted read stays in bounds.")
    out_name = outputs[0].name if outputs else "out"
    call_args = ", ".join(f.name for f in inputs)
    domain_kw = ", domain=domain" if domain else ""
    lines.append(
        f"    {operator_name}({call_args}, out={out_name}, "
        f"offset_provider=offset_provider{domain_kw})"
    )
    lines.append(f"    return {out_name}")
    return "\n".join(lines) + "\n"


def validate_domain(model: "DomainModel", operator_source: str = "") -> DomainReport:
    """Static domain/halo check — no gt4py execution needed.

    Verifies the halo is consistent and, when the operator source is given,
    that every shift it performs has a declared `FieldOffset`. Catches the
    "reads out of bounds because the domain was not restricted" class before
    anything runs.
    """
    report = DomainReport(halo=field_offsets_used(model))

    if not model.arrays:
        report.notes.append("scalars only — no domain to validate")
        return report

    # Every axis with a halo needs the domain restricted; record it.
    domain = interior_domain(model)
    for dim, (lo, hi) in domain.items():
        if lo != "0" or "-" in hi:
            report.notes.append(
                f"{dim}: compute on [{lo}, {hi}) — the boundary layers are read "
                f"by the stencil and cannot be written (halo {report.halo.get(dim, 0)})."
            )

    # If we have the emitted operator, check each FieldOffset it uses is one
    # the model expects — a shift with no matching halo is a red flag.
    if operator_source:
        import re

        used = set(re.findall(r"\b([A-Za-z]\w*)off\s*\[", operator_source))
        expected = set(report.halo)
        stray = used - {d for d in expected}
        if stray:
            report.ok = False
            report.problems.append(
                f"operator shifts by {sorted(stray)}off but the domain model "
                f"found no such offset — the halo/domain would be wrong."
            )

    return report


# ICON/FVM connectivity name letters → gt4py.next mesh dimensions.
_MESH_LETTER = {"c": "Cell", "e": "Edge", "v": "Vertex", "n": "Vertex"}


def _decode_connectivity(table: str) -> tuple[str, str, str]:
    """Decode an `X2Y` table name into (target_dim, source_dim, local_dim).

    `e2c` (edge-to-cell): each edge indexes its neighbouring cells, so the
    field lives on `Cell` (source) and is gathered onto `Edge` (target)
    through a LOCAL `E2C` dimension. A non-`X2Y` name falls back to a
    generic node/neighbour pair.
    """
    m = re.match(r"^([a-z])2([a-z])\d*$", table, re.IGNORECASE)
    if m:
        target = _MESH_LETTER.get(m.group(1).lower(), "Vertex")
        source = _MESH_LETTER.get(m.group(2).lower(), "Vertex")
        return target, source, table.upper()
    # Generic neighbour list: node -> its neighbours.
    return "Vertex", "Vertex", f"{table.upper()}Nbh"


def build_unstructured_driver(model: "DomainModel", operator_name: str) -> str:
    """Generate a gt4py.next **unstructured** driver — the mesh model (#82).

    This is the FVM / icon4py path: a `FieldOffset` over a LOCAL dimension,
    an `as_connectivity` neighbour table passed at call time, and a
    `neighbor_sum` reduction. The connectivity table is a runtime input to
    the driver (Atlas provides it in FVM), not something we synthesise.
    """
    lines: list[str] = []
    lines.append("import gt4py.next as gtx")
    lines.append("from gt4py.next import Dims, neighbor_sum  # noqa: F401")
    lines.append("import numpy as np")
    lines.append(f"from {operator_name}_module import {operator_name}")
    lines.append("")
    lines.append("# Node-based unstructured mesh (FVM / icon4py). Dimensions and")
    lines.append("# LOCAL neighbour axes decoded from the Fortran connectivity names.")

    dims: set[str] = set()
    offsets: list[tuple[str, str, str, str]] = []  # (name, target, source, local)
    for conn in model.connectivities:
        target, source, local = _decode_connectivity(conn.table)
        name = conn.table.upper()
        dims.update({target, source})
        offsets.append((name, target, source, local))

    for dim in sorted(dims):
        lines.append(f'{dim} = gtx.Dimension("{dim}")')
    for name, target, source, local in offsets:
        lines.append(f'{local}Dim = gtx.Dimension("{local}", kind=gtx.DimensionKind.LOCAL)')
        lines.append(
            f'{name} = gtx.FieldOffset("{name}", source={source}, '
            f'target=({target}, {local}Dim))'
        )
    lines.append("")

    # The connectivity tables are integer neighbour lists, not data fields —
    # they come in as `<table>_table` arguments, so exclude them from the
    # data inputs.
    table_names = {c.table for c in model.connectivities}
    inputs = [f for f in model.arrays if "IN" in f.intent and f.name not in table_names]
    outputs = [f for f in model.arrays if "OUT" in f.intent and f.name not in table_names]
    table_args = ", ".join(f"{t}_table" for t in sorted(table_names))
    out_name = outputs[0].name if outputs else "out"
    in_names = ", ".join(f.name for f in inputs)
    field_args = ", ".join(f.name for f in inputs + outputs)

    lines.append(f"def run_{operator_name}({field_args}, *, {table_args}):")
    lines.append('    """Build the connectivity providers (from Atlas in FVM) and run."""')
    prov_parts = []
    for name, target, source, local in offsets:
        table_var = next(c.table for c in model.connectivities if c.table.upper() == name)
        lines.append(
            f"    {name}_conn = gtx.as_connectivity([{target}, {local}Dim], "
            f"codomain={source}, data={table_var}_table, skip_value=-1)"
        )
        prov_parts.append(f'"{name}": {name}_conn')
    lines.append(f"    offset_provider = {{{', '.join(prov_parts)}}}")
    lines.append(
        f"    {operator_name}({in_names}, out={out_name}, "
        f"offset_provider=offset_provider)"
    )
    lines.append(f"    return {out_name}")
    lines.append("")
    accessed = model.connectivities[0].accessed_field if model.connectivities else "field"
    off_name = offsets[0][0] if offsets else "OFF"
    off_local = offsets[0][3] if offsets else "Local"
    lines.append("# In the operator body, a neighbour reduction is:")
    lines.append(f"#     neighbor_sum({accessed}({off_name}), axis={off_local}Dim)")
    return "\n".join(lines) + "\n"


def domain_check_agent(state) -> dict:
    """Generate the driver and statically validate the domain/halos (#82).

    Runs after `type_check` in the GT4Py graph. For each emitted operator it
    builds the driver from the typed domain model (already attached by the
    `domain_model` node) and runs the static domain check — no gt4py
    execution, so it needs no toolchain.
    """
    from fortranspire.agent.nodes._common import SEP, _out, _save

    print(f"\n{SEP}")
    print("  [Domain/halo] generating the driver + static domain check (no exec)")
    print(SEP)

    updated = []
    problems: list[str] = []
    out_dir = _out("gt4py")

    for kernel in state.get("kernel_results", []):
        name = kernel.get("routine_name", "?")
        model = kernel.get("domain_model")
        if model is None or not getattr(model, "arrays", None) or not kernel.get("gt4py_code"):
            updated.append(kernel)
            continue

        if getattr(model, "is_unstructured", False):
            driver = build_unstructured_driver(model, name)
        else:
            driver = build_driver(model, name)
        report = validate_domain(model, kernel.get("gt4py_code", ""))
        _save(out_dir / f"{name}_driver.py", driver)

        halo = report.halo or {}
        if report.ok:
            print(f"  ✓ {name:<26} driver written, halo {halo or '{}'}")
        else:
            for p in report.problems:
                problems.append(f"{name}: {p}")
            print(f"  ✗ {name:<26} domain problem")
        updated.append({**kernel, "gt4py_driver": driver, "domain_report": report})

    return {
        "kernel_results": updated,
        "domain_ok": not problems,
        "domain_problems": problems,
        "executed_agents": state.get("executed_agents", []) + ["domain_check"],
    }
