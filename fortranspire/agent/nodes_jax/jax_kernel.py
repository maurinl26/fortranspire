"""Phase 2, node 4 — emit the JAX body against the derived interface.

The signature is *not* chosen here: :mod:`.functionalize` derived it from
the INTENT map, deterministically. This node fills in the body. That split
is the same one Phase 1 makes between ``pure_elemental`` (rules) and
``openacc`` (LLM), and it exists for the same reason — the part that can
be computed should not be guessed.

Prompts live under ``prompts/jax_kernel/<lang>/<version>.md`` so they are
versioned, reviewable and translatable, like every other prompt family
(issue #3). They were inline in the Phase 2 monolith before this node.
"""
from __future__ import annotations

import os
from typing import List

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown
from fortranspire.agent.nodes_jax._state import JaxKernelInfo, Phase2State

_PROMPT_VERSION = os.getenv("FORTRANSPIRE_JAX_PROMPT_VERSION", "v1")
_PROMPT_LANG = os.getenv("FORTRANSPIRE_PROMPT_LANG", "en")

# How many targeted retries the emitter may spend fixing a deterministic defect
# (a syntax error, a branch on traced array data). Bounded and only spent when a
# defect is actually present — the point is to tolerate a smaller model.
_MAX_REPAIRS = int(os.getenv("FORTRANSPIRE_JAX_MAX_REPAIRS", "2"))

# How far the emitted kernel may go to become differentiable.
#
#   none      faithful translation. Gradients may be zero or undefined
#             where the Fortran was non-smooth. The default, because a
#             translation that silently changes the model is worse than
#             one that is honestly non-differentiable.
#   guarded   guards only — `safe_sqrt`, `safe_divide`, guarded `where`.
#             Forward values are unchanged wherever the original was
#             defined. Repairs a translation; decides nothing.
#   smooth    relaxations too — softmax for MAX, pseudo-Huber for ABS.
#             **Forward values change.** That is a modelling decision, so
#             it is opt-in and every applied relaxation is reported.
SMOOTHING_MODES = ("none", "guarded", "smooth")

_INSTRUCTIONS = {
    "none": (
        "Translate faithfully. Do **not** substitute a smooth approximation for a "
        "non-smooth construct: keep `MAX` as `jnp.maximum`, `ABS` as `jnp.abs`. "
        "The gradient may be one-sided or zero — that is the correct answer for a "
        "faithful translation, and the caller was told."
    ),
    "guarded": (
        "Apply **guards only**. A guard keeps the forward values identical wherever "
        "the original was defined and stops a NaN or an infinite derivative appearing "
        "in the untaken branch of a `where`. Use `safe_sqrt`, `safe_divide`, "
        "`safe_log`. Do **not** relax `MAX`, `MIN`, `ABS` or a threshold — those "
        "change what the code computes."
    ),
    "smooth": (
        "Apply guards **and** relaxations. A relaxation changes what the code "
        "computes, so use the library forms with their explicit parameter and name "
        "each one you applied in a comment on the line, e.g. "
        "`# relaxed: MAX -> smooth_max(beta=50)`. The caller has asked for a "
        "differentiable model, not a faithful one."
    ),
}


def _render_hints(kernel: JaxKernelInfo) -> str:
    """functionalize hints, prefixed with the deterministic typed shapes."""
    hints = list(kernel.get("hints") or [])

    # The typed domain model (shapes, dtypes) is derived deterministically
    # from Loki — the emitter follows it rather than inferring types anew.
    model = kernel.get("domain_model")
    if model is not None:
        from fortranspire.agent.domain_model import to_jax_hints

        hints = to_jax_hints(model) + hints

    if not hints:
        return "- Nothing unusual: independent loops, explicit INTENT, no hidden state."
    return "\n".join(f"- {h}" for h in hints)


def jax_kernel_agent(state: Phase2State) -> dict:
    """LLM: translate each functionalizable routine into a JAX function."""
    from fortranspire.jax_smooth import catalogue_for_prompt
    from fortranspire.llm import get_llm
    from fortranspire.prompts.loader import load_prompt
    from fortranspire.agent.nodes_jax.repair import emit_with_repair
    from langchain_core.messages import HumanMessage, SystemMessage

    smoothing = (state.get("smoothing") or "none").lower()
    if smoothing not in SMOOTHING_MODES:
        print(f"  WARNING: unknown smoothing={smoothing!r}, falling back to 'none'")
        smoothing = "none"

    print(f"\n{SEP}")
    print(f"  [JAX kernel] Emitting against the derived signature — smoothing: {smoothing}")
    print(SEP)
    if smoothing == "smooth":
        print("  Relaxations change the forward values. Applied ones are named in "
              "the emitted code.")

    # Reasoning stage: choosing `scan` over a vectorised expression, or
    # guarding an unsafe branch, is a semantic decision — a wrong call is
    # silently wrong, which is exactly what the cheaper model gets wrong.
    # Built lazily: a run whose routines are all blocked emits nothing and
    # must not require an API key.
    llm = None

    updated: List[JaxKernelInfo] = []
    emitted = 0

    from fortranspire.agent.nodes_jax.jax_skeleton import lower_kernel

    derived = 0
    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]

        if kernel.get("purity") == "blocked":
            print(f"  ⏭ {name:<28} blocked by functionalize — not sent to the LLM")
            updated.append({**kernel, "status": "skipped"})
            continue

        # Deterministic skeleton first: an element-wise kernel is derived from
        # the loop/expression trees, no LLM, no emission-ceiling risk. Only what
        # cannot be lowered yet (stencils, scan, gather) reaches the model.
        skeleton = lower_kernel(kernel)
        if skeleton is not None:
            derived += 1
            print(f"  ✓ {name:<28} {len(skeleton)} chars · derived (no LLM)")
            updated.append({**kernel, "jax_code": skeleton, "status": "pending"})
            continue

        if llm is None:
            llm = get_llm("reasoning")

        system = SystemMessage(
            content=load_prompt(
                "jax_kernel",
                version=_PROMPT_VERSION,
                lang=_PROMPT_LANG,
                signature=kernel["jax_signature"],
                hints=_render_hints(kernel),
                fortran_code=kernel["fortran_code"],
                smoothing_mode=smoothing,
                smoothing_instruction=_INSTRUCTIONS[smoothing],
                smoothing_catalogue=catalogue_for_prompt(),
            )
        )

        # Semantic feedback: run gradcheck on the emitted code and, on failure,
        # hand the exact gradient error back to the model for another try. This
        # is what gives a clean-but-wrong emission (a bad lax.switch) a second,
        # informed attempt instead of just blocking at the gradcheck node.
        def _verify(candidate: str, _kernel=kernel, _name=name) -> list[str]:
            from fortranspire.agent.nodes_jax.gradcheck import check_kernel
            ns: dict = {}
            try:
                exec(candidate, ns)  # noqa: S102 - our own generated code
                fn = ns[_name]
            except Exception as exc:  # noqa: BLE001
                return [f"the module fails to load: {type(exc).__name__}: {exc}"]
            try:
                report = check_kernel(fn, _kernel)
            except Exception as exc:  # noqa: BLE001
                return [f"gradcheck raised: {type(exc).__name__}: {exc}"]
            if report.get("status") == "fail":
                d = (report.get("failures") or [{}])[0]
                return [f"the gradient check failed ({d.get('kind')}): "
                        f"{str(d.get('detail', ''))[:220]} — fix the function so "
                        "jax.grad agrees with finite differences under jit."]
            return []   # pass / skipped / needs_fixture → nothing to re-emit for

        try:
            code, remaining, repairs = emit_with_repair(
                llm, system, strip=_strip_markdown,
                max_repairs=_MAX_REPAIRS, log=print, verify=_verify,
            )
        except Exception as exc:  # noqa: BLE001 - one bad kernel must not kill the run
            print(f"  ✗ {name:<28} LLM error: {type(exc).__name__}: {exc}")
            updated.append({**kernel, "status": "error",
                            "error_log": f"{type(exc).__name__}: {exc}"})
            continue

        emitted += 1
        tag = f"{len(code)} chars"
        if repairs:
            tag += f" · repaired ×{repairs}" + ("" if not remaining else " (defects remain)")
        print(f"  ✓ {name:<28} {tag}")
        updated.append({**kernel, "jax_code": code, "status": "pending"})

    # One file per kernel, plus a consolidated module for the caller.
    out_dir = _out("jax")
    module_parts: List[str] = []
    for kernel in updated:
        if kernel.get("jax_code"):
            _save(out_dir / f"{kernel['routine_name']}.py", kernel["jax_code"])
            module_parts.append(kernel["jax_code"])

    print(f"\n  {derived} derived (no LLM) + {emitted} emitted (LLM) → {out_dir}")

    return {
        "kernel_results": updated,
        "jax_module": "\n\n".join(module_parts),
        "executed_agents": state.get("executed_agents", []) + ["jax_kernel"],
    }
