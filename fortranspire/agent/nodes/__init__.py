"""LangGraph pipeline nodes for Phase 1 (Fortran → OpenACC GPU + Cython).

The whole pipeline used to live in a single 1400-line file. Split here so
each node is independently readable, testable, and swappable. The graph
wiring stays in ``fortranspire.agent.translation_graph_phase1`` and reads
like a TOC of this package.

Public surface — re-exported for backwards compatibility with the old
``from fortranspire.agent.translation_graph_phase1 import ...`` callers:

- State types: :class:`Phase1State`, :class:`KernelInfo`
- Node functions: :func:`init_phase1`, :func:`parser_phase1`,
  :func:`extractor_agent`, :func:`pure_elemental_agent`,
  :func:`openacc_insert_agent`, :func:`cython_wrapper_agent`,
  :func:`validation_agent`
"""
from fortranspire.agent.nodes._state import KernelInfo, Phase1State
from fortranspire.agent.nodes.init import init_phase1
from fortranspire.agent.nodes.parser import parser_phase1
from fortranspire.agent.nodes.extractor import extractor_agent
from fortranspire.agent.nodes.pure_elemental import pure_elemental_agent
from fortranspire.agent.nodes.openacc import openacc_insert_agent
from fortranspire.agent.nodes.cython_wrapper import cython_wrapper_agent
from fortranspire.agent.nodes.validation import validation_agent

__all__ = [
    "KernelInfo",
    "Phase1State",
    "init_phase1",
    "parser_phase1",
    "extractor_agent",
    "pure_elemental_agent",
    "openacc_insert_agent",
    "cython_wrapper_agent",
    "validation_agent",
]
