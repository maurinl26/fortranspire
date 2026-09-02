"""Node 4 — insert OpenACC/OpenMP pragmas on the compute loops and the time loop.

The **kernel** pragmas are now derived deterministically from the AST (see
:mod:`.openacc_gen`): ``collapse(n)`` from the real nest depth, ``reduction``
for scalar accumulators — the correctness fix; a missing reduction clause is a
silent GPU race — and ``private`` for temporaries. The **driver** data region
(where wrong ``!$acc data`` placement is silent corruption) is still LLM-driven.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def _parse_kernel(source: str) -> Optional[object]:
    """Parse one kernel's Fortran with Loki; None if it cannot be parsed."""
    try:
        from loki import Sourcefile
    except Exception:  # noqa: BLE001
        return None
    fd, tmp = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, source.encode())
        os.close(fd)
        src = Sourcefile.from_file(tmp)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    routines = list(src.routines) + \
        [r for m in (src.modules or []) for r in (m.subroutines or [])]
    return routines[0] if routines else None


def openacc_insert_agent(state: Phase1State) -> dict:
    """LLM : insère les pragmas GPU (OpenACC ou OpenMP target) dans les kernels
    et la région data du driver.

    Le type de pragma est choisi par ``state["gpu_pragma"]`` :
      - ``"acc"`` (défaut, rétrocompatible) → !$acc parallel loop / !$acc data
      - ``"omp"`` (issue #18)               → !$omp target teams distribute
                                              parallel do / !$omp target data

    Deux cibles dans tous les cas :
      1. Chaque subroutine kernel → pragmas autour des boucles 2D
      2. Le driver PROGRAM → région data autour du time loop +
         update host(...) avant chaque bloc I/O périodique
    """
    gpu_pragma = (state.get("gpu_pragma") or "acc").lower()
    if gpu_pragma not in ("acc", "omp"):
        print(f"  WARNING: unknown gpu_pragma={gpu_pragma!r}, falling back to 'acc'")
        gpu_pragma = "acc"

    pragma_label   = "OpenACC" if gpu_pragma == "acc" else "OpenMP target"

    print(f"\n{SEP}")
    print(f"  [GPU pragmas: {pragma_label}] Inserting kernel + driver data region "
          "(deterministic, no LLM)")
    print(SEP)

    # ── 4a : Kernel subroutines ───────────────────────────────────────────────
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

        # Compute loop → parallel-loop pragma, DERIVED from the AST (no LLM).
        # collapse(n) from the real nest depth, reduction(op:var) for scalar
        # accumulators (the correctness fix), private(tmp) for scalar temporaries.
        try:
            from fortranspire.agent.nodes.openacc_gen import (
                analyse_loop, insert_pragma, render_pragma,
            )
            from loki import FindNodes
            from loki.ir.nodes import Loop

            routine = _parse_kernel(src)
            loops = FindNodes(Loop).visit(routine.body) if routine is not None else []
            if not loops:
                print(f"  ⏭ {name} — no compute loop found")
                updated.append({**kernel, "openacc_code": src})
                continue

            outer = loops[0]
            info = analyse_loop(outer, carried=bool(kernel.get("has_loop_carried_dep")))
            pragma = render_pragma(info, gpu_pragma)
            annotated = insert_pragma(src, str(outer.variable), pragma)
            annotated = re.sub(
                r"^(\s*)(PURE\s+|ELEMENTAL\s+)+(SUBROUTINE\b)",
                r"\1\3", annotated, flags=re.IGNORECASE | re.MULTILINE,
            )
            print(f"  🚀 {name} → {pragma}")
            updated.append({**kernel, "openacc_code": annotated})
        except Exception as e:  # noqa: BLE001 - never break the pipeline on one kernel
            print(f"  ❌ pragma derivation failed for {name}: {e}")
            updated.append({**kernel, "openacc_code": src, "error_log": str(e)})

    # Extension: .F90 when CPP feature flags are active
    feature_flags = state.get("ast_info", {}).get("feature_flags", {})
    out_ext = ".F90" if feature_flags else ".f90"

    # ── 4b : Driver data region — DERIVED from INTENT (no LLM) ────────────────
    # A wrong copyin/copyout is silent GPU corruption, and the LLM was guessing it
    # from a prompt without the INTENT. Derive it: each array's role is its
    # formal argument's INTENT, aggregated across the kernel calls in the loop.
    driver_src = state.get("driver_fortran", "")
    driver_with_acc = ""

    if driver_src:
        from fortranspire.agent.nodes.data_region import (
            analyse_liveness, array_actuals, clauses_from_liveness,
            derive_data_clauses, extract_kernel_calls, find_region_bounds,
            insert_data_region, render_data_pragma,
        )
        kernels = {k["routine_name"].lower(): k for k in updated}
        kernel_names = set(kernels.keys())
        try:
            calls = extract_kernel_calls(driver_src, kernel_names)
            bounds = find_region_bounds(driver_src, kernel_names)
            if bounds is not None:
                # Liveness-optimal: `create` for arrays the host never touches
                # outside the loop, copyin/copyout by host use before/after.
                arrays = array_actuals(calls, kernels)
                live = analyse_liveness(driver_src, arrays, bounds[0], bounds[1])
                clauses = clauses_from_liveness(live)
            else:
                # No time loop found → INTENT-based conservative copy.
                clauses = derive_data_clauses(calls, kernels)
            open_p, close_p = render_data_pragma(clauses, gpu_pragma)
            driver_with_acc = insert_data_region(driver_src, open_p, close_p, kernel_names)
            _save(_out("fortran_gpu") / f"driver_gpu{out_ext}", driver_with_acc)
            print(f"  driver → {open_p}")
        except Exception as e:  # noqa: BLE001 - never break the pipeline
            driver_with_acc = driver_src
            print(f"  driver data-region derivation failed: {e}")
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
