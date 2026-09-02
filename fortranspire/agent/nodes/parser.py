"""Node 1 — Loki-based AST parsing.

Deterministic, no LLM. Also called standalone by `agent-analyze` and
`agent-doc`, hence the lazy Loki import: callers without the [gpu]
extra can still load this module, the imports just fail at parse time
with a clear error.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fortranspire.agent.nodes._common import SEP
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def _c_type(dtype, kind) -> str:
    """Map a Loki declared type to its C interop type (iso_c_binding).

    Deterministic, from dtype + KIND. A REAL whose KIND is a parameter name
    (not a concrete int Loki could fold) defaults to ``double`` — the safe HPC
    choice, and exactly the case where a resolved frontend (LFortran) would give
    certainty instead of a default.
    """
    try:
        k = int(kind) if kind is not None else None
    except (TypeError, ValueError):
        k = None  # symbolic KIND (e.g. `real(dp)`) — not folded by Loki
    name = str(getattr(dtype, "name", dtype) or "").upper()
    if "INTEGER" in name:
        return "long" if k == 8 else "int"
    if "LOGICAL" in name:
        return "int"
    if "REAL" in name or "DOUBLE" in name:
        return "float" if k == 4 else "double"
    if "COMPLEX" in name:
        return "double complex"
    return "double"


def parser_phase1(state: Phase1State) -> dict:
    """Parse the Fortran source with Loki and extract routines + global schema.

    Walks both top-level subroutines and module-contained subroutines so
    modern Fortran 90 modules are handled. Falls through to a regex-only
    fallback when Loki cannot parse the file.
    """
    filepath = state["fortran_filepath"]
    print(f"\n{SEP}")
    print(f"  [Parser] Loki AST — {filepath}")
    print(SEP)

    # Optional: prefer a local Loki checkout next to the repo (./loki/),
    # useful when iterating on Loki patches without re-publishing.
    _loki_local = Path(__file__).resolve().parents[3] / "loki"
    if _loki_local.is_dir() and str(_loki_local) not in sys.path:
        sys.path.insert(0, str(_loki_local))

    try:
        from loki import Sourcefile, Frontend, FindNodes, BasicType
        from loki.ir.nodes import VariableDeclaration, Loop, CallStatement

        raw_content = Path(filepath).read_text(encoding="utf-8")

        # An uppercase suffix (.F, .F90, .FOR) means "run cpp first" by
        # Fortran convention. Skipping it makes Loki read `#ifdef` lines as
        # Fortran and return **zero routines**, which surfaces as a parse
        # failure rather than as the missing preprocessing pass it is.
        # Line numbers survive the pass so findings still annotate the
        # original file.
        from fortranspire.agent.nodes._preprocess import needs_preprocessing, preprocess

        if needs_preprocessing(filepath):
            raw_content, note = preprocess(filepath, raw_content)
            if note:
                print(f"  WARNING: {note}")
            else:
                print(f"  Preprocessed {Path(filepath).name} (cpp, line numbers preserved).")

        is_program = bool(re.search(r'^\s*PROGRAM\s+', raw_content, re.IGNORECASE | re.MULTILINE))

        if is_program:
            print("  PROGRAM block detected — converting to SUBROUTINE for Loki.")
            raw_content = re.sub(
                r'^\s*PROGRAM\s+(\w+)', r'SUBROUTINE \1_master',
                raw_content, flags=re.IGNORECASE | re.MULTILINE
            )
            raw_content = re.sub(
                r'^\s*END\s+PROGRAM', r'END SUBROUTINE',
                raw_content, flags=re.IGNORECASE | re.MULTILINE
            )

        fd, tmp_path = tempfile.mkstemp(suffix='.f90')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(raw_content)

            # Safety check via subprocess (isolates fparser crashes)
            check = subprocess.run(
                [sys.executable, "-c",
                 f"from loki import Sourcefile; Sourcefile.from_file(r'{tmp_path}')"],
                capture_output=True, timeout=30
            )
            if check.returncode == 0:
                source = Sourcefile.from_file(tmp_path)
            else:
                print(f"  FParser subprocess failed (rc={check.returncode}) — using REGEX frontend.")
                source = Sourcefile.from_file(tmp_path, frontend=Frontend.REGEX)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Collect routines from both top-level (`subroutine foo` outside any
        # module) and module-contained (`module m; contains; subroutine foo`).
        # Loki's `source.routines` only returns the former — modern Fortran 90
        # codes use modules pervasively, so we must walk them explicitly.
        all_routines = list(source.routines)
        for module in (source.modules or []):
            for routine in (module.subroutines or []):
                if routine not in all_routines:
                    all_routines.append(routine)

        if not all_routines:
            raise ValueError("Loki found no routines in this file.")

        print(f"  Parsed {len(all_routines)} routine(s) "
              f"({len(source.routines)} top-level, "
              f"{len(all_routines) - len(source.routines)} module-contained).")

        # ── Global schema extraction ──────────────────────────────────────────
        all_params, all_statics, all_state = [], [], []
        for routine in all_routines:
            for decl in FindNodes(VariableDeclaration).visit(routine.spec):
                for var in decl.symbols:
                    is_param = getattr(var.type, 'parameter', False)
                    dtype    = getattr(var.type, 'dtype', BasicType.DEFERRED)
                    if is_param:
                        (all_statics if dtype == BasicType.LOGICAL else all_params).append(var.name)
                    elif hasattr(var, 'dimensions') and var.dimensions:
                        all_state.append(var.name)

        if not all_params and not all_state:
            print("  Loki schema empty — REGEX fallback.")
            raw_text = Path(filepath).read_text(encoding="utf-8")
            all_params  = re.findall(r'(\w+)\s*,\s*parameter', raw_text, re.IGNORECASE)
            all_statics = re.findall(r'LOGICAL\s*,\s*parameter\s*::\s*(\w+)', raw_text, re.IGNORECASE)
            all_state   = re.findall(r'(\w+)\s*\([^)]+\)', raw_text)

        schema = {
            "params":  sorted(set(all_params)),
            "statics": sorted(set(all_statics)),
            "state":   sorted(set(all_state)),
        }
        print(f"  Schema: {len(schema['params'])} params | "
              f"{len(schema['statics'])} statics | {len(schema['state'])} state vars")

        # ── Per-routine analysis ──────────────────────────────────────────────
        kernel_results: List[KernelInfo] = []
        io_keywords = {'print', 'write', 'read', 'open', 'close', 'inquire'}

        for routine in all_routines:
            intent_map: Dict[str, str] = {}
            arg_ctypes: Dict[str, str] = {}
            if hasattr(routine, 'arguments'):
                for v in routine.arguments:
                    intent = getattr(v.type, 'intent', None)
                    if intent:
                        intent_map[v.name] = intent.upper()
                    # C type for the Cython/iso_c_binding wrapper, from the
                    # declared dtype + KIND (deterministic — never guessed).
                    dt = getattr(v.type, 'dtype', None)
                    kind = getattr(v.type, 'kind', None)
                    arg_ctypes[v.name] = _c_type(dt, kind)

            loops = FindNodes(Loop).visit(routine.body)
            loop_descriptions = [
                str(lp.bounds) if hasattr(lp, 'bounds') else "?" for lp in loops
            ]

            has_io = any(
                any(k in str(node.name).lower() for k in io_keywords)
                for node in FindNodes(CallStatement).visit(routine.body)
            )
            if not has_io:
                fortran_str = routine.to_fortran().lower()
                has_io = any(k in fortran_str for k in io_keywords)

            has_save = False
            if hasattr(routine, 'variables'):
                has_save = any(getattr(v.type, 'save', False) for v in routine.variables)

            dimensions: Dict[str, Any] = {}
            for decl in FindNodes(VariableDeclaration).visit(routine.spec):
                for var in decl.symbols:
                    if hasattr(var, 'dimensions') and var.dimensions:
                        dimensions[var.name] = [str(d) for d in var.dimensions]

            # Free-variable (use-def) analysis: module state the routine reads
            # or writes but does not declare (issue #5). A JAX / gt4py function
            # cannot see module globals, so these get promoted to explicit
            # arguments downstream. Deterministic, from the AST.
            try:
                from fortranspire.agent.dataflow import (
                    free_symbols, infer_dtypes, integer_index_args,
                )
                fs = free_symbols(routine)
                free_reads, free_writes = fs.reads, fs.writes
                arg_dtypes = infer_dtypes(routine)
                index_args = integer_index_args(routine)
            except Exception as _exc:  # noqa: BLE001 - analysis must never break parsing
                free_reads, free_writes, arg_dtypes, index_args = [], [], {}, []
                print(f"  (free-symbol analysis skipped for {routine.name}: {_exc})")

            # Cross-module resolution (#99 Stage 0): resolve the shapes/types of
            # promoted module state from the modules that declare it (the
            # routine's own directory, plus any FORTRANSPIRE_MODULE_PATH dirs).
            # This lifts gradcheck's `needs_fixture` when an index table's shape
            # was the only thing missing. Never breaks parsing.
            resolved: Dict[str, Any] = {}
            if free_reads or free_writes:
                try:
                    from fortranspire.agent.resolve import (
                        default_search_dirs, resolve_for_routine,
                    )
                    extra = list(state.get("module_search_dirs") or [])
                    env_path = os.getenv("FORTRANSPIRE_MODULE_PATH", "")
                    extra += [d for d in env_path.split(os.pathsep) if d]
                    dirs = default_search_dirs(filepath, extra)
                    syms = resolve_for_routine(routine, dirs)
                    referenced = {n.lower() for n in (free_reads + free_writes)}
                    for low, sym in syms.items():
                        if low in referenced:
                            resolved[low] = {
                                "dtype": sym.dtype, "rank": sym.rank,
                                "is_parameter": sym.is_parameter, "module": sym.module,
                            }
                            # Resolution is authoritative for a promoted symbol's type.
                            if sym.dtype != "unknown":
                                arg_dtypes[low] = sym.dtype
                    if resolved:
                        print(f"  Resolved {len(resolved)} module symbol(s) for "
                              f"{routine.name} from USE modules")
                except Exception as _exc:  # noqa: BLE001
                    print(f"  (module resolution skipped for {routine.name}: {_exc})")
            if free_reads or free_writes:
                print(f"  Free module state in {routine.name}: "
                      f"+{len(free_reads)} read, +{len(free_writes)} written")

            kernel_results.append({
                "routine_name":       routine.name,
                "fortran_code":       routine.to_fortran(),
                "pure_elemental_code": "",
                "openacc_code":       "",
                "intent_map":         intent_map,
                "is_pure":            False,
                "is_elemental":       False,
                "has_io":             has_io,
                "has_save":           has_save,
                "loops":              loop_descriptions,
                "dimensions":         dimensions,
                "free_reads":         free_reads,
                "free_writes":        free_writes,
                "arg_dtypes":         arg_dtypes,
                "arg_ctypes":         arg_ctypes,
                "index_args":         index_args,
                "resolved":           resolved,
                "status":             "pending",
                "error_log":          "",
            })
            print(f"  Routine: {routine.name} | loops={len(loops)} | "
                  f"io={has_io} | save={has_save} | args={len(intent_map)}")

        # ── Source-level static analysis (regex on raw source) ───────────
        raw_src = Path(filepath).read_text(encoding="utf-8")

        # G3 — Implicit types / missing KIND
        has_implicit_none  = bool(re.search(r"^\s*IMPLICIT\s+NONE", raw_src, re.IGNORECASE | re.MULTILINE))
        has_implicit_types = bool(re.search(
            r"^\s*(REAL|INTEGER|COMPLEX|DOUBLE\s+PRECISION)\s+\w",
            raw_src, re.IGNORECASE | re.MULTILINE,
        ))

        # G4 — COMMON blocks
        common_blocks = [
            {"name": n, "vars": [v.strip() for v in vs.split(",") if v.strip()]}
            for n, vs in re.findall(r"COMMON\s*/(\w+)/\s*([^\n!]+)", raw_src, re.IGNORECASE)
        ]

        # G6 — Feature flags USE_xx / APPLY_xx (LOGICAL PARAMETER)
        feature_flags = {
            name: val.upper()
            for name, val in re.findall(
                r"LOGICAL\s*,\s*PARAMETER\s*::\s*(\w+)\s*=\s*(\.TRUE\.|\.FALSE\.)",
                raw_src, re.IGNORECASE,
            )
        }

        # G7 — POINTER attributes
        has_pointers = bool(re.search(r",\s*POINTER\s*::", raw_src, re.IGNORECASE))

        # G8 — Derived types (AoS candidate)
        has_derived_types = bool(re.search(r"^\s*TYPE\s*::", raw_src, re.IGNORECASE | re.MULTILINE))

        # G2 — Loop-carried dependency per kernel (same array read & written)
        for ki in kernel_results:
            src = ki["fortran_code"]
            lhs_names = set(re.findall(r"^\s*(\w+)\s*\(", src, re.MULTILINE))
            dep = any(
                bool(re.search(rf"\b{n}\s*\([^)]*[ij]\s*[-+]\s*1", src, re.IGNORECASE))
                and bool(re.search(rf"^\s*{n}\s*\(", src, re.IGNORECASE | re.MULTILINE))
                for n in lhs_names
            )
            ki["has_loop_carried_dep"] = dep

        if common_blocks:
            print(f"  ⚠️  COMMON blocks detected: {[b['name'] for b in common_blocks]}")
        if has_implicit_types:
            print("  ⚠️  Implicit type declarations detected (no KIND) — will normalize")
        if has_pointers:
            print("  ⚠️  POINTER attributes detected — will convert to allocatable/args")
        if has_derived_types:
            print("  ⚠️  Derived TYPE detected (AoS candidate) — flagged for review")
        if feature_flags:
            print(f"  🔧 Feature flags: {list(feature_flags.keys())}")

        return {
            "kernel_results": kernel_results,
            "schema": schema,
            "is_program": is_program,
            "ast_info": {
                "status":            "parsed",
                "routines":          [k["routine_name"] for k in kernel_results],
                "has_implicit_none": has_implicit_none,
                "has_implicit_types": has_implicit_types,
                "common_blocks":     common_blocks,
                "feature_flags":     feature_flags,
                "has_pointers":      has_pointers,
                "has_derived_types": has_derived_types,
            },
            "executed_agents": list(state.get("executed_agents", [])) + ["parser"],
        }

    except Exception as e:
        import traceback
        print(f"  Loki failed: {e}")
        traceback.print_exc()
        raw = Path(filepath).read_text(encoding="utf-8")
        return {
            "kernel_results": [{
                "routine_name": "kernel",
                "fortran_code": raw,
                "pure_elemental_code": "",
                "openacc_code": "",
                "intent_map": {},
                "is_pure": False,
                "is_elemental": False,
                "has_io": False,
                "has_save": False,
                "loops": [],
                "dimensions": {},
                "status": "error",
                "error_log": str(e),
            }],
            "schema": {"params": [], "statics": [], "state": []},
            "is_program": False,
            "ast_info": {"status": "error", "message": str(e)},
            "executed_agents": list(state.get("executed_agents", [])) + ["parser"],
        }
