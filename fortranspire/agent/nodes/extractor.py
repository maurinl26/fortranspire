"""Node 2 — extract compute kernels from monolithic Fortran into a MODULE.

Single LLM call (`reasoning` stage). Removes COMMON blocks and SAVE state
by promoting them to explicit INTENT(INOUT) arguments.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def extractor_agent(state: Phase1State) -> dict:
    """LLM : extrait les boucles compute du PROGRAM en subroutines dans un MODULE.

    Cas typique : codes scientifiques monolithiques (seismic_CPML) où les kernels FD
    sont des blocs do/enddo inline dans le PROGRAM. L'extraction est nécessaire pour :
      - annoter chaque kernel avec PURE/ELEMENTAL individuellement
      - ajouter !$acc parallel loop sur les boucles spatiales 2D
      - générer un wrapper Cython sur des subroutines avec INTENT explicites

    Sorties :
      module_kernels.f90 — MODULE avec N subroutines (kernels GPU purs)
      driver.f90         — PROGRAM driver appelant USE module_kernels + les subroutines
    """
    print(f"\n{SEP}")
    print("  [Extractor] Identifying and extracting compute kernels into MODULE")
    print(SEP)

    # Reasoning stage: semantic refactoring of monolithic Fortran → modular form.
    # Best fit: Mistral-Large (or any large reasoning model on a sovereign endpoint).
    from fortranspire.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_llm("reasoning")

    # Read the original source to give LLM full context
    filepath = state["fortran_filepath"]
    try:
        full_source = Path(filepath).read_text(encoding="utf-8")
    except Exception:
        full_source = state.get("fortran_code", "")

    # Keep up to 2000 lines (covers most seismic CPML files fully)
    lines = full_source.split("\n")
    source_preview = "\n".join(lines[:2000]) if len(lines) > 2000 else full_source

    module_name = Path(filepath).stem.lower().replace("-", "_").replace(".", "_")

    ast_info      = state.get("ast_info", {})
    common_blocks = ast_info.get("common_blocks", [])
    feature_flags = ast_info.get("feature_flags", {})
    has_pointers  = ast_info.get("has_pointers", False)

    # Build context-dependent rule sections (passed to the prompt template
    # via load_prompt(); empty string = omitted block).
    common_rules = ""
    if common_blocks:
        names = ", ".join(f"/{b['name']}/" for b in common_blocks)
        common_rules = (
            f"\nCOMMON BLOCKS ({names}) — mandatory:\n"
            "  - Do NOT reproduce COMMON blocks in the MODULE.\n"
            "  - Variables only read by a kernel → INTENT(IN) argument.\n"
            "  - Variables modified by a kernel → INTENT(INOUT) argument.\n"
            "  - Global constants (PARAMETER) may stay as MODULE-level PARAMETER.\n"
        )

    save_rules = (
        "\nSAVE VARIABLES — mandatory:\n"
        "  - Variables with SAVE attribute (persistent state between calls) become\n"
        "    INTENT(INOUT) arguments of the subroutine. The driver declares them and\n"
        "    passes them at each call. Remove the SAVE attribute from the declaration.\n"
        "  Example: 'real, save :: psi_vx = 0.0' → 'real(dp), intent(inout) :: psi_vx'\n"
    )

    flag_rules = ""
    if feature_flags:
        active = [k for k, v in feature_flags.items() if ".TRUE." in v]
        inactive = [k for k, v in feature_flags.items() if ".FALSE." in v]
        flag_rules = (
            f"\nFEATURE FLAGS ({', '.join(feature_flags)}) — CPP preprocessing:\n"
            "  - Convert 'if (USE_xxx) then ... end if' → '#ifdef USE_xxx\\n...\\n#endif'\n"
            "  - Output file must use .F90 extension (triggers CPP automatically).\n"
            "  - Add a comment header listing active flags at top of MODULE.\n"
            f"  - Active (.TRUE.): {active}   Inactive (.FALSE.): {inactive}\n"
        )

    pointer_rules = ""
    if has_pointers:
        pointer_rules = (
            "\nPOINTERS — convert to safe alternatives:\n"
            "  - 'real, pointer :: field(:,:)' → 'real(dp), allocatable :: field(:,:)'\n"
            "    if the target is always one well-defined array.\n"
            "  - Otherwise, pass the target directly as INTENT(IN/INOUT) argument.\n"
            "  - Remove all 'field => target' association statements.\n"
        )

    from fortranspire.prompts.loader import load_prompt

    system = SystemMessage(content=load_prompt(
        "extractor", version="v1",
        common_rules=common_rules,
        save_rules=save_rules,
        flag_rules=flag_rules,
        pointer_rules=pointer_rules,
    ))

    prompt = HumanMessage(content=(
        f"Extract the GPU compute kernels from this Fortran PROGRAM into a MODULE.\n"
        f"Module name: {module_name}_kernels\n\n"
        f"Source code:\n```fortran\n{source_preview}\n```"
    ))

    try:
        resp = llm.invoke([system, prompt])
        content = resp.content

        # Parse [MODULE]...[/MODULE] and [DRIVER]...[/DRIVER] blocks
        module_match = re.search(r'\[MODULE\](.*?)\[/MODULE\]', content, re.DOTALL | re.IGNORECASE)
        driver_match = re.search(r'\[DRIVER\](.*?)\[/DRIVER\]', content, re.DOTALL | re.IGNORECASE)

        if module_match and driver_match:
            module_code = _strip_markdown(module_match.group(1).strip())
            driver_code = _strip_markdown(driver_match.group(1).strip())
        else:
            # Fallback: try to find two fortran code blocks
            blocks = re.findall(r'```fortran\n(.*?)\n```', content, re.DOTALL)
            if len(blocks) >= 2:
                module_code = blocks[0].strip()
                driver_code = blocks[1].strip()
            else:
                # Last resort: entire response is the module
                module_code = _strip_markdown(content)
                driver_code = ""
                print("  WARNING: could not parse separate MODULE/DRIVER blocks")

        # G3 safety net — ensure IMPLICIT NONE appears before CONTAINS
        if "implicit none" not in module_code.lower():
            module_code = re.sub(
                r"(\bcontains\b)", "  implicit none\n\\1",
                module_code, count=1, flags=re.IGNORECASE,
            )

        # G6 safety net — rename to .F90 if CPP flags are present
        out_ext = ".F90" if feature_flags else ".f90"

        # Extract kernel subroutine names from module code
        kernel_names = re.findall(r'^\s*subroutine\s+(\w+)\s*\(', module_code,
                                  re.IGNORECASE | re.MULTILINE)
        print(f"  Extracted {len(kernel_names)} kernel subroutine(s): {kernel_names}")

        # Save outputs (use .F90 extension when CPP flags are active)
        _save(_out("fortran_gpu") / f"module_kernels{out_ext}", module_code)
        if driver_code:
            _save(_out("fortran_gpu") / f"driver{out_ext}", driver_code)

        # Rebuild kernel_results from extracted subroutines
        # (the parser only saw the monolithic PROGRAM; now we have real subroutines)
        updated_kernels: List[KernelInfo] = []
        for name in kernel_names:
            # Extract this subroutine's source from the module code
            sub_match = re.search(
                rf'subroutine\s+{name}\s*\(.*?end\s+subroutine\s+{name}',
                module_code, re.DOTALL | re.IGNORECASE
            )
            sub_code = sub_match.group(0) if sub_match else f"! subroutine {name} (extraction failed)"

            # Parse INTENT from extracted code
            intent_map: Dict[str, str] = {}
            for m in re.finditer(r'intent\s*\(\s*(in|out|inout)\s*\)\s*::\s*([^\n!]+)',
                                  sub_code, re.IGNORECASE):
                intent_str = m.group(1).upper()
                for var in m.group(2).replace(' ', '').split(','):
                    if var.strip():
                        intent_map[var.strip()] = intent_str

            updated_kernels.append({
                "routine_name":       name,
                "fortran_code":       sub_code,
                "pure_elemental_code": "",
                "openacc_code":       "",
                "intent_map":         intent_map,
                "is_pure":            False,
                "is_elemental":       False,
                "has_io":             False,   # extracted kernels have no I/O by construction
                "has_save":           False,
                "loops":              [],
                "dimensions":         {},
                "status":             "extracted",
                "error_log":          "",
            })

        # If extraction produced no kernels, keep the parser's kernel_results
        if not updated_kernels:
            updated_kernels = state.get("kernel_results", [])
            print("  WARNING: no subroutines extracted — keeping parser results")

        return {
            "module_fortran":  module_code,
            "driver_fortran":  driver_code,
            "kernel_names":    kernel_names,
            "kernel_results":  updated_kernels,
            "executed_agents": list(state.get("executed_agents", [])) + ["extractor"],
        }

    except Exception as e:
        import traceback
        print(f"  LLM extraction failed: {e}")
        traceback.print_exc()
        return {
            "module_fortran":  "",
            "driver_fortran":  "",
            "kernel_names":    [],
            "kernel_results":  state.get("kernel_results", []),
            "executed_agents": list(state.get("executed_agents", [])) + ["extractor"],
        }
