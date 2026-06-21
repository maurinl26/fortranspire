"""Node 3 — annotate kernels with PURE / ELEMENTAL where the rules allow.

Deterministic, no LLM. Built on the `has_io` / `has_save` / `loops` flags
gathered by the parser.
"""
from __future__ import annotations

import re
from typing import List

from fortranspire.agent.nodes._common import SEP, _out, _save
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def _annotate_purity(kernel: "KernelInfo") -> tuple[str, bool, bool]:
    """Détermine et applique l'annotation PURE/ELEMENTAL par règles AST — sans LLM.

    Loki a déjà détecté has_io, has_save, intent_map, loops.
    Règles :
      - has_io=True  → non éligible (WRITE/PRINT/READ incompatibles avec PURE)
      - has_save=True → non éligible (état persistant incompatible avec PURE)
      - Sinon : PURE si des boucles sont présentes (FD stencil)
      - ELEMENTAL si aucune boucle interne (fonction scalaire point-à-point)
    PURE est un hint sémantique intermédiaire — l'étape openacc le retirera
    avant d'ajouter !$acc parallel loop (Fortran standard l'exige).
    """
    src = kernel["fortran_code"]
    if kernel["has_io"] or kernel["has_save"]:
        return src, False, False

    has_loops    = bool(kernel.get("loops"))
    is_elemental = not has_loops  # scalaire point-à-point sans boucle interne

    if is_elemental:
        annotated = re.sub(
            r"^(\s*)(function\b)",
            r"\1ELEMENTAL \2", src, count=1, flags=re.IGNORECASE | re.MULTILINE,
        )
        return annotated, False, True

    # FD stencil avec boucles → PURE (hint intermédiaire, sera retiré par openacc)
    annotated = re.sub(
        r"^(\s*)(subroutine\b)",
        r"\1PURE \2", src, count=1, flags=re.IGNORECASE | re.MULTILINE,
    )
    return annotated, True, False


def pure_elemental_agent(state: Phase1State) -> dict:
    """Annote les kernels PURE/ELEMENTAL par règles AST — zéro appel LLM.

    Loki a déjà collecté has_io, has_save, loops, intent_map au stade parser.
    La décision est donc déterministe et reproductible.
    """
    print(f"\n{SEP}")
    print("  [PURE/ELEMENTAL] Annotating compute kernels (deterministic — no LLM)")
    print(SEP)

    updated: List[KernelInfo] = []
    for kernel in state.get("kernel_results", []):
        if kernel["has_io"] or kernel["has_save"]:
            label = "I/O" if kernel["has_io"] else "SAVE"
            print(f"  ⏭ Skip {kernel['routine_name']} ({label} — not eligible)")
            updated.append({**kernel, "pure_elemental_code": kernel["fortran_code"]})
            continue

        annotated, is_pure, is_elemental = _annotate_purity(kernel)
        tag = "ELEMENTAL" if is_elemental else ("PURE" if is_pure else "plain")
        print(f"  ✨ {kernel['routine_name']} → {tag}")
        updated.append({
            **kernel,
            "pure_elemental_code": annotated,
            "is_pure": is_pure,
            "is_elemental": is_elemental,
        })

    combined = "\n\n".join(k["pure_elemental_code"] for k in updated)
    _save(_out("fortran_gpu") / "kernel_pure.f90", combined)

    return {
        "kernel_results": updated,
        "pure_elemental_fortran": combined,
        "executed_agents": list(state.get("executed_agents", [])) + ["pure_elemental"],
    }
