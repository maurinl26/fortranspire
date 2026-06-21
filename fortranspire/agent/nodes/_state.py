"""Shared TypedDicts for the Phase 1 pipeline state.

Kept in its own module so every node imports it without pulling each
other in. The structure mirrors the LangGraph state passed between nodes.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class KernelInfo(TypedDict):
    routine_name: str
    fortran_code: str           # Code source Fortran original (extrait par Loki)
    pure_elemental_code: str    # Code annoté PURE/ELEMENTAL (ou original si non éligible)
    openacc_code: str           # Code avec pragmas OpenACC
    intent_map: Dict[str, str]  # {arg_name: "IN"|"OUT"|"INOUT"}
    is_pure: bool
    is_elemental: bool
    has_io: bool                # I/O Fortran (PRINT/WRITE/READ) détecté par Loki
    has_save: bool              # Variables SAVE détectées par Loki
    loops: List[str]            # Descriptions des bornes de boucles
    dimensions: Dict[str, Any]  # {var_name: [dim1, dim2, ...]}
    status: str                 # "pending" | "success" | "error"
    error_log: str


class Phase1State(TypedDict):
    fortran_filepath: str
    fortran_code: str
    # Loki AST analysis
    ast_info: Dict[str, Any]
    kernel_results: List[KernelInfo]   # Plain list — replaced (not appended) at each step
    schema: Dict[str, Any]             # {params, statics, state}
    is_program: bool
    # Extractor outputs (produced before PURE/ELEMENTAL)
    module_fortran: str     # MODULE contenant les kernels extraits (module_kernels.f90)
    driver_fortran: str     # PROGRAM driver appelant le MODULE (driver.f90)
    kernel_names: List[str] # Noms des subroutines extraites
    # Phase outputs
    pure_elemental_fortran: str        # Fortran annoté PURE/ELEMENTAL
    openacc_fortran: str               # Fortran avec pragmas OpenACC (kernels + driver data region)
    cython_pyx: str                    # Contenu .pyx
    cython_header: str                 # Contenu kernel_c.h (iso_c_binding)
    cython_setup: str                  # pyproject.toml build config
    # Validation
    validation_passed: bool
    validation_log: str
    # GPU directive family — "acc" (default, OpenACC) or "omp" (issue #18,
    # OpenMP target). Picked by the openacc node to select the right
    # prompt + emitted directives. Older callers that don't set this key
    # get "acc" by .get() default → backwards-compatible.
    gpu_pragma: str
    # Tracking
    executed_agents: List[str]
