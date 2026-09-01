"""Emit a gt4py.next field operator against the derived interface (#42).

Mirrors the Phase 2 `jax_kernel` node exactly — the signature and the
purity verdict are already derived by the shared `functionalize` node, so
this node fills in the operator body against a fixed interface. The
correspondences it instructs the model on are the ones specified in
`docs/concepts/gt4py-next-patterns.md`, and the prompt pins the confirmed
gt4py.next ≥ 1.2 API surface.
"""
from __future__ import annotations

import os
from typing import List

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown

_PROMPT_VERSION = os.getenv("FORTRANSPIRE_GT4PY_PROMPT_VERSION", "v1")
_PROMPT_LANG = os.getenv("FORTRANSPIRE_PROMPT_LANG", "en")


def _render_hints(kernel: dict) -> str:
    """Reuse the functionalize hints, plus GT4Py-specific dimension notes."""
    hints = list(kernel.get("hints") or [])

    dims = kernel.get("dimensions") or {}
    arrays = [a for a in (kernel.get("intent_map") or {}) if dims.get(a)]
    if arrays:
        ranks = {a: len(dims[a]) for a in arrays}
        hints.append(
            "Field dimensions to use: "
            + ", ".join(f"{a} ({ranks[a]}-D)" for a in arrays)
            + ". A 1-D field is usually a column `Dims[K]`, 2-D a horizontal "
            "`Dims[I, J]` or `Dims[Cell]`, 3-D `Dims[Cell, K]`. Pick named "
            "dimensions consistently across the operator."
        )

    from fortranspire.agent.nodes_gt4py.portability import score_routine

    verdict = score_routine(kernel, kernel.get("fortran_code", ""))
    if verdict.score <= 1:
        hints.append(
            f"GT4Py portability is low ({verdict.score}/5): {verdict.reason}. "
            "Emit the closest faithful operator and flag what does not map."
        )
    elif verdict.score == 3:
        hints.append(f"This routine needs a construct: {verdict.reason}.")

    return "\n".join(f"- {h}" for h in hints) if hints else \
        "- Point-wise stencil, explicit dimensions, no hidden state."


def gt4py_kernel_agent(state) -> dict:
    """LLM: translate each functionalizable routine into a gt4py.next operator."""
    from fortranspire.llm import get_llm
    from fortranspire.prompts.loader import load_prompt
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{SEP}")
    print("  [GT4Py kernel] emitting field operators against the derived signature")
    print(SEP)

    # The LLM is built lazily, on the first routine that actually needs
    # emitting. A run whose routines are all blocked (I/O, hidden state)
    # emits nothing and must not require an API key — and the CI exercises
    # exactly that path.
    llm = None
    updated: List[dict] = []
    emitted = 0

    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]

        if kernel.get("purity") == "blocked":
            print(f"  ⏭ {name:<28} blocked by functionalize — not a field operator")
            updated.append({**kernel, "status": "skipped"})
            continue

        if llm is None:
            llm = get_llm("reasoning")

        system = SystemMessage(
            content=load_prompt(
                "gt4py_kernel",
                version=_PROMPT_VERSION,
                lang=_PROMPT_LANG,
                signature=kernel.get("jax_signature", f"def {name}(...)"),
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
        updated.append({**kernel, "gt4py_code": code, "status": "pending"})

    out_dir = _out("gt4py")
    module_parts: List[str] = []
    for kernel in updated:
        if kernel.get("gt4py_code"):
            _save(out_dir / f"{kernel['routine_name']}.py", kernel["gt4py_code"])
            module_parts.append(kernel["gt4py_code"])

    print(f"\n  {emitted} operator(s) emitted → {out_dir}")

    return {
        "kernel_results": updated,
        "gt4py_module": "\n\n".join(module_parts),
        "executed_agents": state.get("executed_agents", []) + ["gt4py_kernel"],
    }
