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


def _render_hints(kernel: JaxKernelInfo) -> str:
    hints = kernel.get("hints") or []
    if not hints:
        return "- Nothing unusual: independent loops, explicit INTENT, no hidden state."
    return "\n".join(f"- {h}" for h in hints)


def jax_kernel_agent(state: Phase2State) -> dict:
    """LLM: translate each functionalizable routine into a JAX function."""
    from fortranspire.llm import get_llm
    from fortranspire.prompts.loader import load_prompt
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{SEP}")
    print("  [JAX kernel] Emitting against the derived signature")
    print(SEP)

    # Reasoning stage: choosing `scan` over a vectorised expression, or
    # guarding an unsafe branch, is a semantic decision — a wrong call is
    # silently wrong, which is exactly what the cheaper model gets wrong.
    llm = get_llm("reasoning")

    updated: List[JaxKernelInfo] = []
    emitted = 0

    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]

        if kernel.get("purity") == "blocked":
            print(f"  ⏭ {name:<28} blocked by functionalize — not sent to the LLM")
            updated.append({**kernel, "status": "skipped"})
            continue

        system = SystemMessage(
            content=load_prompt(
                "jax_kernel",
                version=_PROMPT_VERSION,
                lang=_PROMPT_LANG,
                signature=kernel["jax_signature"],
                hints=_render_hints(kernel),
                fortran_code=kernel["fortran_code"],
            )
        )

        try:
            response = llm.invoke([system, HumanMessage(content="Emit the module.")])
            code = _strip_markdown(getattr(response, "content", str(response)))
        except Exception as exc:  # noqa: BLE001 - one bad kernel must not kill the run
            print(f"  ✗ {name:<28} LLM error: {type(exc).__name__}: {exc}")
            updated.append({**kernel, "status": "error",
                            "error_log": f"{type(exc).__name__}: {exc}"})
            continue

        emitted += 1
        print(f"  ✓ {name:<28} {len(code)} chars")
        updated.append({**kernel, "jax_code": code, "status": "pending"})

    # One file per kernel, plus a consolidated module for the caller.
    out_dir = _out("jax")
    module_parts: List[str] = []
    for kernel in updated:
        if kernel.get("jax_code"):
            _save(out_dir / f"{kernel['routine_name']}.py", kernel["jax_code"])
            module_parts.append(kernel["jax_code"])

    print(f"\n  {emitted} kernel(s) emitted → {out_dir}")

    return {
        "kernel_results": updated,
        "jax_module": "\n\n".join(module_parts),
        "executed_agents": state.get("executed_agents", []) + ["jax_kernel"],
    }
