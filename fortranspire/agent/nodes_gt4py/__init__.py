"""GT4Py (gt4py.next) pipeline nodes — Fortran → functional field operators.

gt4py.next is functional, exactly like the Phase 2 JAX target, so this
package reuses the deterministic `functionalize` node (INTENT → interface +
purity verdict) unchanged and adds only the emission and validation.

- :mod:`.portability`     — the `FORT032` score (deterministic)
- :mod:`.gt4py_kernel`    — LLM emission of the field operator
- :mod:`.domain_validate` — type-check against gt4py.next's frontend
"""
from fortranspire.agent.nodes_gt4py.domain_validate import (
    domain_validate_agent,
    type_check_source,
)
from fortranspire.agent.nodes_gt4py.gt4py_kernel import gt4py_kernel_agent
from fortranspire.agent.nodes_gt4py.portability import Gt4PyVerdict, score_routine

__all__ = [
    "score_routine", "Gt4PyVerdict",
    "gt4py_kernel_agent",
    "domain_validate_agent", "type_check_source",
]
