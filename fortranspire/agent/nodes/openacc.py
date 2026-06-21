"""Node 4 — insert OpenACC pragmas around the 2D loop nests and the time loop.

LLM-driven (`reasoning` stage — wrong `!$acc data` placement = silent GPU
corruption, so it deserves Mistral-Large rather than Codestral).
"""
from __future__ import annotations

import re
from typing import List

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def openacc_insert_agent(state: Phase1State) -> dict:
    """LLM : insère les pragmas OpenACC dans les kernels ET la région data du driver.

    Deux cibles :
      1. Chaque subroutine kernel → !$acc parallel loop collapse(2) sur les boucles 2D
      2. Le driver PROGRAM → !$acc data ... région autour du time loop +
         !$acc update host(...) avant chaque bloc I/O périodique
    """
    print(f"\n{SEP}")
    print("  [OpenACC] Inserting OpenACC pragmas (kernels + driver data region)")
    print(SEP)

    # Reasoning stage: data-flow analysis to place !$acc data copyin/copy clauses
    # correctly around the time loop. Wrong placement = silent GPU corruption.
    from fortranspire.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_llm("reasoning")

    # ── 4a : Kernel subroutines ───────────────────────────────────────────────
    kernel_system = SystemMessage(content=(
        "You are an OpenACC GPU expert for scientific Fortran.\n"
        "Add OpenACC directives to parallelize this subroutine on NVIDIA A100 GPUs.\n"
        "Compiler: nvfortran -acc -gpu=cc80 (Ampere)\n\n"
        "CRITICAL — PURE/ELEMENTAL compatibility:\n"
        "  The Fortran standard forbids OpenACC compute directives (!$acc parallel, !$acc kernels)\n"
        "  inside PURE or ELEMENTAL procedures. If the subroutine has PURE or ELEMENTAL in its\n"
        "  declaration, REMOVE that keyword. The functional purity is preserved as a semantic\n"
        "  property (all INTENT are explicit, no I/O, no SAVE) — but the Fortran keyword must\n"
        "  be absent when !$acc parallel loop is present.\n\n"
        "Guidelines for finite-difference stencil subroutines:\n"
        "  - Remove PURE / ELEMENTAL from the subroutine statement\n"
        "  - Add !$acc parallel loop collapse(2) before the outermost 2D loop nest\n"
        "  - Add private(...) clause for scalar temporaries computed inside the loop\n"
        "  - Do NOT add data movement clauses here — handled by the driver !$acc data region\n"
        "  - !$acc end parallel after end of loop nest\n"
        "  - The subroutine does NOT need !$acc routine — it's called from host, not device\n"
        "Return ONLY the modified Fortran subroutine, no prose."
    ))

    updated: List[KernelInfo] = []
    for kernel in state.get("kernel_results", []):
        src = kernel.get("pure_elemental_code") or kernel["fortran_code"]
        name = kernel["routine_name"]

        if kernel["has_io"]:
            print(f"  ⏭ Skip {name} (has I/O)")
            updated.append({**kernel, "openacc_code": src})
            continue

        # G1 — ELEMENTAL → !$acc routine seq (no !$acc parallel inside ELEMENTAL)
        if kernel.get("is_elemental"):
            annotated = re.sub(
                r"^(\s*)(PURE\s+|ELEMENTAL\s+)+(SUBROUTINE|FUNCTION)\b",
                r"\1\3", src, flags=re.IGNORECASE | re.MULTILINE,
            )
            annotated = re.sub(
                r"(^\s*(?:subroutine|function)\s+\w+[^\n]*\n)",
                r"\1  !$acc routine seq\n",
                annotated, count=1, flags=re.IGNORECASE | re.MULTILINE,
            )
            print(f"  ⚡ {name} (ELEMENTAL) → !$acc routine seq")
            updated.append({**kernel, "openacc_code": annotated})
            continue

        # G2 — Loop-carried dependency → skip collapse, warn
        if kernel.get("has_loop_carried_dep"):
            annotated = re.sub(
                r"^(\s*)(PURE\s+|ELEMENTAL\s+)+(SUBROUTINE\b)",
                r"\1\3", src, flags=re.IGNORECASE | re.MULTILINE,
            )
            # Inject a warning comment before first do loop
            annotated = re.sub(
                r"(^\s*do\s+\w+\s*=)",
                r"  ! ⚠ loop-carried dependency detected — cannot use !$acc parallel loop collapse\n\1",
                annotated, count=1, flags=re.IGNORECASE | re.MULTILINE,
            )
            print(f"  ⚠ {name} — loop-carried dependency, skipping !$acc parallel")
            updated.append({**kernel, "openacc_code": annotated})
            continue

        # FD stencil → !$acc parallel loop collapse(2) via LLM
        prompt = HumanMessage(content=(
            f"This is a 2D FD stencil subroutine (NOT ELEMENTAL — accesses neighbours).\n"
            f"Add !$acc parallel loop collapse(2) inside the subroutine body.\n"
            f"INTENT(IN):    {[n for n,i in kernel['intent_map'].items() if i=='IN']}\n"
            f"INTENT(INOUT): {[n for n,i in kernel['intent_map'].items() if i=='INOUT']}\n\n"
            f"```fortran\n{src}\n```"
        ))
        try:
            resp = llm.invoke([kernel_system, prompt])
            annotated = _strip_markdown(resp.content)
            # Safety net: strip any remaining PURE/ELEMENTAL the LLM left
            annotated = re.sub(
                r"^(\s*)(PURE\s+|ELEMENTAL\s+)+(SUBROUTINE\b)",
                r"\1\3", annotated, flags=re.IGNORECASE | re.MULTILINE,
            )
            print(f"  🚀 {name} → !$acc parallel loop collapse(2)")
            updated.append({**kernel, "openacc_code": annotated})
        except Exception as e:
            print(f"  ❌ LLM failed for {name}: {e}")
            updated.append({**kernel, "openacc_code": src, "error_log": str(e)})

    # Extension: .F90 when CPP feature flags are active
    feature_flags = state.get("ast_info", {}).get("feature_flags", {})
    out_ext = ".F90" if feature_flags else ".f90"

    # ── 4b : Driver data region ───────────────────────────────────────────────
    driver_src = state.get("driver_fortran", "")
    driver_with_acc = ""

    if driver_src:
        driver_system = SystemMessage(content=(
            "You are an OpenACC GPU expert.\n"
            "Add an !$acc data region around the time loop in this Fortran PROGRAM driver.\n"
            "The subroutines inside the loop are already annotated with !$acc parallel loop.\n\n"
            "Guidelines:\n"
            "  - !$acc data copyin(lambda,mu,rho,b_x,b_x_half,b_y,b_y_half,a_x,...) "
            "copy(vx,vy,sigma_xx,sigma_yy,sigma_xy,memory_dvx_dx,...) before the time loop\n"
            "  - INTENT(IN) arrays  → copyin(...)\n"
            "  - INTENT(INOUT) arrays (field + memory arrays) → copy(...)\n"
            "  - Just before each periodic I/O block (if mod(it,IT_DISPLAY)==0): "
            "add !$acc update host(vx,vy) to transfer velocity fields for PRINT/image output\n"
            "  - !$acc end data after end of time loop\n"
            "  - Keep ALL existing code and I/O intact — only add !$acc directives\n"
            "Return ONLY the modified Fortran PROGRAM."
        ))
        driver_prompt = HumanMessage(content=(
            f"Add !$acc data region around the time loop.\n"
            f"Kernel subroutines called inside the loop: {state.get('kernel_names', [])}\n\n"
            f"```fortran\n{driver_src}\n```"
        ))
        try:
            resp = llm.invoke([driver_system, driver_prompt])
            driver_with_acc = _strip_markdown(resp.content)
            _save(_out("fortran_gpu") / f"driver_gpu{out_ext}", driver_with_acc)
            print(f"  driver → !$acc data region inserted")
        except Exception as e:
            driver_with_acc = driver_src
            print(f"  LLM failed for driver data region: {e}")
    else:
        print("  No driver.f90 found — skipping driver data region")

    # ── Save annotated MODULE ────────────────────────────────────────────────
    module_combined = "\n\n".join(k["openacc_code"] for k in updated)
    _save(_out("fortran_gpu") / f"module_kernels_gpu{out_ext}", module_combined)

    # The fallback "kernel_gpu" target for validation is the full GPU source
    full_gpu = module_combined + ("\n\n" + driver_with_acc if driver_with_acc else "")
    _save(_out("fortran_gpu") / f"kernel_gpu{out_ext}", full_gpu)

    return {
        "kernel_results": updated,
        "openacc_fortran": full_gpu,
        "driver_fortran":  driver_with_acc or driver_src,
        "executed_agents": list(state.get("executed_agents", [])) + ["openacc"],
    }
