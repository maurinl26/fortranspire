"""Shared TypedDicts for the Phase 2 (Fortran → JAX) pipeline state.

Phase 2 is **not** Phase 1 followed by a translation: it targets JAX
directly, with a functional refactoring in the middle instead of pragma
insertion. The state therefore carries a purity verdict and a functional
signature that have no Phase 1 equivalent.

Mirrors the layout of :mod:`fortranspire.agent.nodes._state` so the two
pipelines stay legible side by side.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

# What the functionalize node concluded about a routine.
#
#   pure      — no side effect, no hidden state: becomes a JAX function as-is
#   threaded  — hidden state exists but can be passed explicitly as an extra
#               argument and returned alongside the outputs
#   blocked   — cannot become a pure function without changing what the code
#               *does* (Fortran I/O, or no observable output at all)
Purity = Literal["pure", "threaded", "blocked"]


class JaxKernelInfo(TypedDict):
    routine_name: str
    fortran_code: str
    intent_map: Dict[str, str]        # {arg: "IN"|"OUT"|"INOUT"}
    dimensions: Dict[str, Any]

    # ── Functional interface, derived deterministically from intent_map ──
    # An imperative `SUBROUTINE k(a, b)` with INTENT(IN) a / INTENT(OUT) b
    # becomes `def k(a) -> b`. Mutation of arguments is not expressible in
    # JAX, so every written argument has to come back as a return value.
    inputs: List[str]                 # INTENT(IN) + INTENT(INOUT)
    outputs: List[str]                # INTENT(OUT) + INTENT(INOUT)
    carried: List[str]                # INTENT(INOUT) — read *and* written
    jax_signature: str                # rendered `def name(...) -> tuple[...]`

    # ── Purity verdict ──────────────────────────────────────────────────
    purity: Purity
    purity_reason: str

    # ── JAX construct hints, also deterministic ─────────────────────────
    needs_scan: bool                  # loop-carried dependency → lax.scan
    hints: List[str]                  # human-readable notes fed to the prompt

    # ── Emission + validation ───────────────────────────────────────────
    jax_code: str
    gradcheck: Dict[str, Any]         # {"status", "max_abs_err", "tol", ...}
    status: str                       # "pending" | "success" | "error" | "skipped"
    error_log: str


class Phase2State(TypedDict):
    fortran_filepath: str
    fortran_code: str
    ast_info: Dict[str, Any]
    kernel_results: List[JaxKernelInfo]
    is_program: bool

    # Extractor outputs, reused from Phase 1 (COMMON / SAVE promoted to
    # explicit arguments) — the first half of functionalisation.
    module_fortran: str
    driver_fortran: str
    kernel_names: List[str]

    # How far emission may go to obtain gradients: "none" | "guarded" |
    # "smooth". Only "smooth" changes what the kernel computes, which is
    # why it is a mode rather than a default.
    smoothing: str

    # Phase 2 outputs
    functionalized: bool
    jax_module: str                   # consolidated importable module
    gradcheck_passed: bool
    gradcheck_log: str

    executed_agents: List[str]
