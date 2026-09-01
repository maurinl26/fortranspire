"""Phase 2, node 3 — derive the functional interface. No LLM.

This is the step Phase 2 was missing (issue #73). Going straight from
imperative Fortran to JAX asks the model to invent the function signature
*and* the body in one jump, and the signature is exactly the part that
must not be invented: it is pure data-flow, already computed by Loki.

What is decided here, deterministically:

* **The functional signature.** A subroutine mutates its arguments; a JAX
  function cannot. `SUBROUTINE k(a, b)` with `INTENT(IN) a`,
  `INTENT(OUT) b` becomes `def k(a) -> b`. Every written argument has to
  come back as a return value, and `INTENT(INOUT)` appears on both sides.

* **The purity verdict.** `PURE`/`ELEMENTAL` is Fortran's own word for
  functional purity — no side effects, no persistent state — and the
  parser already establishes the two facts that decide it (``has_io``,
  ``has_save``). A routine that cannot be `PURE` cannot become a JAX
  function without threading its state explicitly. That is a free,
  deterministic gate, and Phase 2 was not using it.

* **Which JAX construct the loops need.** A loop-carried dependency is a
  recurrence: it maps to ``lax.scan``, not to a vectorised expression.
  ``has_loop_carried_dep`` is already computed for FORT004.

The LLM node that follows emits the *body* against this fixed interface,
rather than choosing it.
"""
from __future__ import annotations

from typing import List

from fortranspire.agent.nodes._common import SEP
from fortranspire.agent.nodes_jax._state import JaxKernelInfo, Phase2State, Purity


def _split_by_intent(intent_map: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Return ``(inputs, outputs, carried)`` from the INTENT map.

    An argument with no recorded INTENT is treated as an input *and* an
    output. That is deliberately pessimistic: an unknown INTENT is the
    Fortran default (effectively INOUT), and assuming read-only would
    silently drop a mutation from the returned tuple.
    """
    inputs: list[str] = []
    outputs: list[str] = []
    carried: list[str] = []

    for arg, intent in intent_map.items():
        norm = (intent or "").strip().upper()
        if norm == "IN":
            inputs.append(arg)
        elif norm == "OUT":
            outputs.append(arg)
        else:  # INOUT, or unknown → pessimistic
            inputs.append(arg)
            outputs.append(arg)
            carried.append(arg)

    return inputs, outputs, carried


def _promote_free_state(
    kernel: dict,
    inputs: List[str],
    outputs: List[str],
    carried: List[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Promote module state to explicit arguments (issue #5).

    The parser's free-variable analysis lists the module-provided symbols a
    routine reads (``free_reads``) or writes (``free_writes``) — state a pure
    function cannot see. Read state becomes an extra **input**; written state
    is read-and-written, so it is threaded as **INOUT** (input *and* output).
    Order: the declared arguments first, promoted state appended, so the
    original call signature is preserved as a prefix.
    """
    reads = list(kernel.get("free_reads") or [])
    writes = list(kernel.get("free_writes") or [])
    promoted: list[str] = []

    in_lower = {x.lower() for x in inputs}
    out_lower = {x.lower() for x in outputs}

    for r in reads:
        if r.lower() not in in_lower:
            inputs.append(r)
            in_lower.add(r.lower())
            promoted.append(r)
    for w in writes:
        if w.lower() not in in_lower:
            inputs.append(w)
            in_lower.add(w.lower())
        if w.lower() not in out_lower:
            outputs.append(w)
            out_lower.add(w.lower())
            carried.append(w)
        promoted.append(w)

    return inputs, outputs, carried, promoted


def _render_signature(name: str, inputs: List[str], outputs: List[str]) -> str:
    """Render the Python signature the emitted kernel must match.

    The return is documented in a trailing comment, **not** a `->`
    annotation: the outputs are argument *names* (`avg`), and
    ``def k(a) -> avg`` is invalid Python — `avg` is undefined at
    definition time, so the emitted file raises ``NameError`` when imported
    on its own. (It slipped past ``gradcheck`` only because that module's
    ``from __future__ import annotations`` leaks into its ``exec``.) A
    comment carries the same information to the emitter while keeping the
    line importable.
    """
    args = ", ".join(inputs) if inputs else ""
    if not outputs:
        returns = "None"
    elif len(outputs) == 1:
        returns = outputs[0]
    else:
        returns = f"({', '.join(outputs)})"
    return f"def {name}({args}):  # returns: {returns}"


def _verdict(kernel: dict, outputs: List[str]) -> tuple[Purity, str]:
    """Decide whether the routine can become a pure JAX function."""
    if kernel.get("has_io"):
        return (
            "blocked",
            "Fortran I/O in the routine body — a side effect that cannot be "
            "expressed as a return value. Extract the I/O into the caller.",
        )
    if not outputs:
        return (
            "blocked",
            "no INTENT(OUT) or INTENT(INOUT) argument — the routine's effect is "
            "invisible from its signature, so it writes global state. Promote "
            "that state to an explicit argument first.",
        )
    if kernel.get("has_save"):
        return (
            "threaded",
            "SAVE state persists between calls. It has to be passed in and "
            "returned as an extra carry argument; JAX cannot hold it.",
        )
    return ("pure", "no I/O, no SAVE, and every effect is visible in the signature.")


def _hints(kernel: dict, carried: List[str], needs_scan: bool) -> List[str]:
    """Deterministic notes handed to the emission prompt."""
    hints: List[str] = []

    if needs_scan:
        hints.append(
            "A loop-carried dependency was detected: the loop is a recurrence. "
            "Emit `jax.lax.scan` over the sequential axis — a vectorised "
            "expression would silently compute the wrong answer."
        )
    elif kernel.get("loops"):
        hints.append(
            "Loops carry no detected dependency: they vectorise. Prefer whole-array "
            "`jnp` expressions over an explicit Python loop."
        )

    if carried:
        hints.append(
            f"{', '.join(carried)} are read and written (INTENT(INOUT)). In JAX they "
            "must be taken as arguments and returned as new values — never mutated "
            "in place. Use `.at[...].set(...)` for indexed updates."
        )

    if kernel.get("has_save"):
        hints.append(
            "SAVE state must appear as an explicit extra argument and be returned "
            "alongside the outputs."
        )

    dims = kernel.get("dimensions") or {}
    arrays = [a for a in kernel.get("intent_map", {}) if dims.get(a)]
    if arrays:
        hints.append(
            f"Array arguments: {', '.join(arrays)}. Keep their declared rank; do not "
            "flatten or transpose them, the caller relies on the Fortran layout."
        )

    return hints


def functionalize_agent(state: Phase2State) -> dict:
    """Derive the functional interface for every kernel. Deterministic."""
    print(f"\n{SEP}")
    print("  [Functionalize] Deriving the functional interface — no LLM")
    print(SEP)

    updated: List[JaxKernelInfo] = []
    blocked = 0

    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]
        intent_map = kernel.get("intent_map") or {}

        inputs, outputs, carried = _split_by_intent(intent_map)
        # Promote module state (USE globals) to explicit arguments before the
        # verdict: a routine that "writes global state" is only blocked if that
        # state cannot be made visible — promotion is exactly making it visible.
        inputs, outputs, carried, promoted = _promote_free_state(
            kernel, inputs, outputs, carried)
        purity, reason = _verdict(kernel, outputs)
        needs_scan = bool(kernel.get("has_loop_carried_dep"))
        signature = _render_signature(name, inputs, outputs)

        hints = _hints(kernel, carried, needs_scan)
        if promoted:
            hints.insert(0,
                f"Promoted from module state (USE globals): {', '.join(promoted)}. "
                "In the Fortran these are read/written as module globals; a pure "
                "function cannot see them, so they are explicit arguments now. "
                "Reference them by these names — never read a global or an "
                "undefined symbol. Integer topology (index/count arrays) stays "
                "integer; do not differentiate through it."
            )

        entry: JaxKernelInfo = {
            **kernel,  # carry the parser's fields through
            "inputs": inputs,
            "outputs": outputs,
            "carried": carried,
            "jax_signature": signature,
            "purity": purity,
            "purity_reason": reason,
            "needs_scan": needs_scan,
            "hints": hints,
            "jax_code": kernel.get("jax_code", ""),
            "gradcheck": kernel.get("gradcheck", {}),
            "status": kernel.get("status", "pending"),
            "error_log": kernel.get("error_log", ""),
        }

        if purity == "blocked":
            blocked += 1
            entry["status"] = "skipped"
            print(f"  ✗ {name:<28} blocked — {reason.splitlines()[0]}")
        else:
            marker = "~" if purity == "threaded" else "✓"
            print(f"  {marker} {name:<28} {signature}")
            if needs_scan:
                print(f"    ↳ recurrence detected → lax.scan")

        updated.append(entry)

    total = len(updated)
    print(f"\n  {total - blocked}/{total} routine(s) can become JAX functions.")
    if blocked:
        print(f"  {blocked} blocked — reported, not silently ported.")

    return {
        "kernel_results": updated,
        "functionalized": True,
        "executed_agents": state.get("executed_agents", []) + ["functionalize"],
    }
