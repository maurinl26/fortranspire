"""LangGraph wiring for Phase 1 — Fortran → OpenACC GPU + Cython.

This module used to hold all six pipeline nodes inline (~1400 lines, hard
to test, hard to extend). Now it's just the graph definition; each node
lives in its own module under :mod:`fortranspire.agent.nodes`.

Pipeline (sequential, no branching):

    init → parser → extractor → pure_elemental → openacc → cython_wrapper → validation → END

LLM call budget (4 maximum):

- ``extractor``      — 1 call (reasoning)
- ``openacc``        — 1 call (reasoning)
- ``cython_wrapper`` — 2 calls (code)

The other three nodes (``init``, ``parser``, ``pure_elemental``,
``validation``) are deterministic and never touch an LLM.

Compilation targets validated by ``validation``:

- ``gfortran -fsyntax-only`` (CPU flavour, stripped of ``!$acc`` directives)
- ``gfortran -fopenacc -fsyntax-only`` (CPU flavour with directives)
- ``nvfortran -acc -gpu=cc80 -fsyntax-only`` (GPU flavour, A100)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from fortranspire.agent.nodes import (
    Phase1State,
    init_phase1,
    parser_phase1,
    extractor_agent,
    pure_elemental_agent,
    openacc_insert_agent,
    cython_wrapper_agent,
    validation_agent,
)

workflow_phase1 = StateGraph(Phase1State)

workflow_phase1.add_node("init",           init_phase1)
workflow_phase1.add_node("parser",         parser_phase1)
workflow_phase1.add_node("extractor",      extractor_agent)
workflow_phase1.add_node("pure_elemental", pure_elemental_agent)
workflow_phase1.add_node("openacc",        openacc_insert_agent)
workflow_phase1.add_node("cython_wrapper", cython_wrapper_agent)
workflow_phase1.add_node("validation",     validation_agent)

workflow_phase1.set_entry_point("init")
workflow_phase1.add_edge("init",           "parser")
workflow_phase1.add_edge("parser",         "extractor")
workflow_phase1.add_edge("extractor",      "pure_elemental")
workflow_phase1.add_edge("pure_elemental", "openacc")
workflow_phase1.add_edge("openacc",        "cython_wrapper")
workflow_phase1.add_edge("cython_wrapper", "validation")
workflow_phase1.add_edge("validation",     END)

translation_app_phase1 = workflow_phase1.compile()

# Backwards-compat re-export — existing call sites
# (`from fortranspire.agent.translation_graph_phase1 import parser_phase1`)
# keep working. New code should import from
# `fortranspire.agent.nodes` directly.
__all__ = [
    "translation_app_phase1",
    "Phase1State",
    "init_phase1",
    "parser_phase1",
    "extractor_agent",
    "pure_elemental_agent",
    "openacc_insert_agent",
    "cython_wrapper_agent",
    "validation_agent",
]
