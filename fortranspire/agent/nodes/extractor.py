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


def _split_entities(decl: str) -> List[str]:
    """Split a Fortran entity-declaration list on **top-level** commas.

    A declaration like ``rki(numcells,nrxns), yin(numcells,ischan)`` must
    split into two entities, not four: the commas *inside* the array
    dimension ``(...)`` are not separators. A naive ``split(',')`` breaks
    every multi-dimensional array argument, corrupting its name to
    ``rki(numcells`` — which then propagates into the derived signature and
    the emitted kernel's keyword arguments.
    """
    entities: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in decl:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            entities.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        entities.append("".join(current))
    return [e for e in (e.strip() for e in entities) if e]


def _entity_name(entity: str) -> str:
    """The bare argument name from a declared entity.

    ``rki(numcells,nrxns)`` → ``rki``; ``x => null()`` and ``y = 0.0`` keep
    only the identifier. The INTENT map is keyed by the argument name, not
    its shape or initialiser.
    """
    entity = entity.split("=", 1)[0]          # drop initialiser / pointer assoc
    entity = entity.split("(", 1)[0]          # drop the dimension spec
    return entity.strip()


def extractor_agent(state: Phase1State) -> dict:
    """Decompose the source into kernels. LLM **only** for a monolithic PROGRAM.

    There are two shapes of input, and only one needs an LLM:

    * **Modular** (a MODULE, or bare SUBROUTINEs) — the routines *are* the
      kernels. The parser already derived each one from Loki's AST with its
      real INTENT map, loops, I/O / SAVE flags and array dimensions. There is
      nothing to lift, so the LLM must not run: on modular code it re-guesses
      a decomposition against a prompt written for seismic finite-difference
      PROGRAMs and re-parses declarations by regex — which is exactly how it
      mangled CMAQ ``RBFEVAL``. This path is deterministic, general and
      spends no token.

    * **Monolithic PROGRAM** (e.g. seismic_CPML) — the compute loops are inline
      ``do/enddo`` blocks inside one PROGRAM. Here the loops must be lifted into
      subroutines, which is a semantic refactoring the LLM does. The parser
      set ``is_program`` so we route on it.

    Outputs (monolithic path):
      module_kernels.f90 — MODULE with N kernel subroutines
      driver.f90         — PROGRAM driver calling USE module_kernels
    """
    print(f"\n{SEP}")
    print("  [Extractor] Identifying and extracting compute kernels into MODULE")
    print(SEP)

    # ── Loki-native path: modular source is already decomposed ──────────────
    # `is_program` comes from the parser (a real PROGRAM block was seen). When
    # it is false the source is a module / subroutines and the parser's
    # kernel_results are the answer — pass them through untouched.
    if not state.get("is_program"):
        routines = state.get("kernel_results", [])
        names = [k["routine_name"] for k in routines]
        print(f"  Modular source — {len(names)} routine(s) already decomposed "
              f"by Loki's AST; skipping LLM extraction.")
        for n in names:
            print(f"    • {n}")
        module_src = "\n\n".join(k.get("fortran_code", "") for k in routines)
        if module_src.strip():
            _save(_out("fortran_gpu") / "module_kernels.f90", module_src)
        return {
            "module_fortran":  module_src,
            "driver_fortran":  "",
            "kernel_names":    names,
            "kernel_results":  routines,
            "executed_agents": list(state.get("executed_agents", [])) + ["extractor"],
        }

    # ── Monolithic PROGRAM path: lift inline compute loops into subroutines ──
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

    from fortranspire.agent.schemas import ExtractorOutput
    from fortranspire.prompts.loader import load_prompt

    system = SystemMessage(content=load_prompt(
        "extractor", version="v2",
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

    from fortranspire.observability import tracer
    from fortranspire.observability.llm_callback import token_callback

    try:
        # Prefer structured outputs (Mistral La Plateforme JSON-schema mode).
        # Fall back to the regex parser for legacy / self-hosted backends.
        module_code = ""
        driver_code = ""
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        with tracer.span(node="extractor", model=model_name) as span:
            cfg = {"callbacks": [token_callback(span)]}
            try:
                result = llm.with_structured_output(ExtractorOutput).invoke([system, prompt], config=cfg)
                module_code = _strip_markdown(result.module_fortran.strip())
                driver_code = _strip_markdown(result.driver_fortran.strip())
            except Exception as struct_exc:
                print(f"  Structured output unavailable ({struct_exc}); using regex fallback.")
                # Same v2 prompt — many models still emit usable JSON even when
                # `with_structured_output` is not wired. Parse whatever comes back.
                resp = llm.invoke([system, prompt], config=cfg)
                content = resp.content
                # JSON first (v2 prompt asks for it)
                import json as _json
                try:
                    blob = _json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0))
                    module_code = _strip_markdown(blob.get("module_fortran", "").strip())
                    driver_code = _strip_markdown(blob.get("driver_fortran", "").strip())
                except Exception:
                    # Last-resort fallback to the v1 delimiter blocks.
                    module_match = re.search(r'\[MODULE\](.*?)\[/MODULE\]', content, re.DOTALL | re.IGNORECASE)
                    driver_match = re.search(r'\[DRIVER\](.*?)\[/DRIVER\]', content, re.DOTALL | re.IGNORECASE)
                    if module_match and driver_match:
                        module_code = _strip_markdown(module_match.group(1).strip())
                        driver_code = _strip_markdown(driver_match.group(1).strip())
                    else:
                        blocks = re.findall(r'```fortran\n(.*?)\n```', content, re.DOTALL)
                        if len(blocks) >= 2:
                            module_code, driver_code = blocks[0].strip(), blocks[1].strip()
                        else:
                            module_code = _strip_markdown(content)
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
                for entity in _split_entities(m.group(2)):
                    name = _entity_name(entity)
                    if name:
                        intent_map[name] = intent_str

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
