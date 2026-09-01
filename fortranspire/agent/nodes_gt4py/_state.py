"""State for the GT4Py (gt4py.next) pipeline. Extends the Phase 2 state.

GT4Py reuses the Phase 2 functional analysis wholesale — the same INTENT
map, the same purity verdict, the same functional signature — so its state
is the Phase 2 state plus the GT4Py emission and validation fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class Phase_GT4Py_State(TypedDict, total=False):
    fortran_filepath: str
    fortran_code: str
    ast_info: Dict[str, Any]
    kernel_results: List[dict]
    is_program: bool
    module_fortran: str
    driver_fortran: str
    kernel_names: List[str]
    functionalized: bool
    smoothing: str
    # GT4Py outputs
    gt4py_module: str
    domain_validated: bool
    domain_check_skipped: bool
    domain_log: str
    executed_agents: List[str]
