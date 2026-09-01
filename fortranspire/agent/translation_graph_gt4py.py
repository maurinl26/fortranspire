"""LangGraph wiring for the GT4Py target — Fortran → gt4py.next (#42).

gt4py.next is functional, so this graph is the Phase 2 graph with the
emission and validation swapped:

    init → parser → extractor → functionalize → gt4py_kernel → domain_validate

The first three nodes are shared with Phase 1; `functionalize` is shared
with Phase 2 (JAX) unchanged — the same purity analysis decides both
targets. Only `gt4py_kernel` (emit a field operator) and `domain_validate`
(type-check against gt4py.next's frontend) are GT4Py-specific.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from fortranspire.agent.nodes import extractor_agent, init_phase1, parser_phase1
from fortranspire.agent.nodes_gt4py import domain_validate_agent, gt4py_kernel_agent
from fortranspire.agent.nodes_gt4py._state import Phase_GT4Py_State
from fortranspire.agent.nodes_jax import functionalize_agent

workflow_gt4py = StateGraph(Phase_GT4Py_State)

workflow_gt4py.add_node("init",            init_phase1)
workflow_gt4py.add_node("parser",          parser_phase1)
workflow_gt4py.add_node("extractor",       extractor_agent)
workflow_gt4py.add_node("functionalize",   functionalize_agent)
workflow_gt4py.add_node("gt4py_kernel",    gt4py_kernel_agent)
workflow_gt4py.add_node("domain_validate", domain_validate_agent)

workflow_gt4py.set_entry_point("init")
workflow_gt4py.add_edge("init",            "parser")
workflow_gt4py.add_edge("parser",          "extractor")
workflow_gt4py.add_edge("extractor",       "functionalize")
workflow_gt4py.add_edge("functionalize",   "gt4py_kernel")
workflow_gt4py.add_edge("gt4py_kernel",    "domain_validate")
workflow_gt4py.add_edge("domain_validate", END)

translation_app_gt4py = workflow_gt4py.compile()
