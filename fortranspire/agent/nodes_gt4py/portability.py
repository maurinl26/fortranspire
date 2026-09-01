"""GT4Py (gt4py.next) portability scoring — deterministic, no LLM.

The `FORT032` finding: given a parsed routine, how good a gt4py.next field
operator would it make? This lets a user triage which kernels are worth a
GT4Py port before any LLM is called, exactly as `FORT030` does for JAX and
the port-cost estimate does for the cost.

The rubric is the one written in `docs/concepts/gt4py-next-patterns.md`
(§9), and it is grounded in the same facts every other target reads:

* the **purity verdict** from the Phase 2 `functionalize` node is the
  floor — a routine that cannot be a pure function cannot be a field
  operator either, so `blocked` caps the score at 0 and `threaded`
  (hidden SAVE state, threadable but not pure) caps it at 3;
* a **loop-carried dependency** (already computed for `FORT004`) is a
  vertical recurrence: it maps, but to a `scan_operator`, not a plain
  field operator — score 3;
* **indirect indexing** (`a(e2c(e))`) is an **unstructured connectivity**
  access — the mature, high-value gt4py.next model (icon4py, Pace), where a
  cell's neighbours come from a connectivity table and the reduction is a
  `neighbor_sum` over a `FieldOffset`. It maps to a construct, so it scores
  3, the same tier as `where` and `scan_operator` — not a penalty. (The
  older Cartesian horizontal-grid model — `Ioff`/`Joff` shifts — is more
  procedural and less clearly worth it; the vertical `Koff` shift below is
  universal and stays first-class, since every mesh has a column.);
* a **data-dependent branch** (`IF … THEN`) maps to `where()`, which is a
  construct beyond a point-wise operator — score 3;
* everything else — point-wise, or a constant-offset stencil — is a clean
  field operator, score 5.

Scores are deliberately coarse (0/1/3/5). A finer number would imply a
precision the static signals do not have.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# Comment tails would otherwise make `! uses idx(i)` look like indirection.
_COMMENT_RE = re.compile(r"!.*$", re.MULTILINE)

# `name(other(...))` — an array indexed by another array reference. Rough,
# but it is the signature of a connectivity / permutation access, which is
# the boundary between a Cartesian stencil (portable) and unstructured or
# unportable indexing.
_INDIRECT_RE = re.compile(r"\b[A-Za-z_]\w*\s*\(\s*[A-Za-z_]\w*\s*\(")
# ICON-style connectivity names: X2Y (edge-to-cell e2c, cell-to-edge c2e, …),
# a strong signal that an indirection is a mesh connectivity, not an
# arbitrary permutation.
_MESH_CONN_RE = re.compile(r"\b[a-z]2[a-z]\b", re.IGNORECASE)

# A data-dependent branch. `IF (...) THEN` (block) or a one-line
# `IF (...) x = ...`. Either becomes a `where()`.
_BRANCH_RE = re.compile(r"^\s*IF\s*\(", re.IGNORECASE | re.MULTILINE)


class Gt4PyVerdict(NamedTuple):
    score: int          # 0 | 1 | 3 | 5
    label: str          # short human label
    reason: str         # one line, names the mapping or the blocker


def _purity_floor(kernel: dict) -> tuple[int, str] | None:
    """Return a cap from the shared purity verdict, or None if pure.

    Imported from the Phase 2 node rather than restated: "can this be a
    pure function?" must have exactly one answer across targets, or a
    GT4Py score would promise a port the pipeline then refuses.
    """
    from fortranspire.agent.nodes_jax.functionalize import _split_by_intent, _verdict

    _, outputs, _ = _split_by_intent(kernel.get("intent_map") or {})
    verdict, reason = _verdict(kernel, outputs)
    if verdict == "blocked":
        return 0, reason
    if verdict == "threaded":
        return 3, reason
    return None  # pure — no floor imposed


def score_routine(kernel: dict, source: str = "") -> Gt4PyVerdict:
    """Score one parsed routine for a gt4py.next port. Deterministic.

    `source` is the routine's Fortran text; when empty, only the parser
    flags are used (the source-scan signals are simply not raised).
    """
    floor = _purity_floor(kernel)
    if floor is not None and floor[0] == 0:
        return Gt4PyVerdict(0, "does not map",
                            f"cannot be a field operator — {floor[1]}")

    stripped = _COMMENT_RE.sub("", source) if source else ""

    # Indirect indexing is an unstructured connectivity access — the mature,
    # high-value gt4py.next model (icon4py / Pace). It maps to a construct
    # (neighbor_sum over a FieldOffset on a LOCAL dimension), so it is a
    # first-class target at the same tier as where/scan, not a penalty.
    if stripped and _INDIRECT_RE.search(stripped):
        mesh = " (mesh connectivity, e.g. `e2c`)" if _MESH_CONN_RE.search(stripped) else ""
        return Gt4PyVerdict(
            3, "unstructured connectivity",
            f"connectivity access{mesh} → `neighbor_sum(a(OFF), axis=LocalDim)` "
            "over an `as_connectivity` table — the unstructured model",
        )

    # A vertical recurrence maps to scan_operator, not a plain operator.
    if kernel.get("has_loop_carried_dep"):
        return Gt4PyVerdict(
            3, "needs a construct",
            "loop-carried dependency → a vertical recurrence, maps to "
            "`@scan_operator` (not a plain field operator)",
        )

    # SAVE state: threadable but not pure — capped at 3 by the floor.
    if floor is not None:  # threaded
        return Gt4PyVerdict(
            3, "needs a construct",
            f"{floor[1]} — the state must be passed and returned explicitly",
        )

    # A data-dependent branch maps to where().
    if stripped and _BRANCH_RE.search(stripped):
        return Gt4PyVerdict(
            3, "needs a construct",
            "data-dependent branch → `where(cond, a, b)` (no `if` in a "
            "field operator)",
        )

    return Gt4PyVerdict(
        5, "field operator",
        "point-wise or constant-offset stencil — a clean field operator",
    )
