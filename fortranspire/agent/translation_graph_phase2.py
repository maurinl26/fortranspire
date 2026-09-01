"""LangGraph wiring for Phase 2 — Fortran → functional refactoring → JAX.

Issue #73. Phase 2 is **not** Phase 1 followed by a translation: it never
emits an OpenACC pragma. It targets JAX directly, and the step that makes
that possible is a functional refactoring — purity, no hidden state,
explicit data flow — which is what `jit`, `grad` and `vmap` require.

Pipeline (sequential, no branching):

    init → parser → extractor → functionalize → jax_kernel → gradcheck → END

The first three nodes are **shared with Phase 1**, deliberately. Parsing is
the same work; and the extractor's promotion of ``COMMON`` / ``SAVE`` state
to explicit arguments is the first half of functionalisation whatever the
target. Forking them would mean maintaining two versions of the hardest
code in the project.

LLM call budget (2 maximum):

- ``extractor``  — 1 call (reasoning), shared with Phase 1
- ``jax_kernel`` — 1 call (reasoning)

``functionalize`` and ``gradcheck`` are deterministic and never touch an
LLM. That is the point: the functional *interface* is derived from the
INTENT map rather than guessed, and the gradient is verified numerically
rather than assumed.

What replaced what
------------------

The previous Phase 2 graph ran ``parser → explainer → dispatcher →
translator → consolidator``: no functionalisation step at all, and a
validation that stopped at ``make_jaxpr`` — which proves the code *traces*,
not that its gradients are right. A kernel that traces with a wrong
gradient is silently wrong exactly where the caller relies on it.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from fortranspire.agent.nodes import (
    extractor_agent,
    init_phase1,
    parser_phase1,
)
from fortranspire.agent.domain_model import domain_model_agent
from fortranspire.agent.nodes_jax import (
    functionalize_agent,
    gradcheck_agent,
    jax_kernel_agent,
)
from fortranspire.agent.nodes_jax.equivalence import equivalence_agent
from fortranspire.agent.nodes_jax._state import Phase2State

workflow_phase2 = StateGraph(Phase2State)

workflow_phase2.add_node("init",           init_phase1)
workflow_phase2.add_node("parser",         parser_phase1)
workflow_phase2.add_node("extractor",      extractor_agent)
workflow_phase2.add_node("functionalize",  functionalize_agent)
workflow_phase2.add_node("domain_model",   domain_model_agent)
workflow_phase2.add_node("jax_kernel",     jax_kernel_agent)
workflow_phase2.add_node("gradcheck",      gradcheck_agent)
workflow_phase2.add_node("equivalence",    equivalence_agent)

workflow_phase2.set_entry_point("init")
workflow_phase2.add_edge("init",          "parser")
workflow_phase2.add_edge("parser",        "extractor")
workflow_phase2.add_edge("extractor",     "functionalize")
workflow_phase2.add_edge("functionalize", "domain_model")
workflow_phase2.add_edge("domain_model",  "jax_kernel")
workflow_phase2.add_edge("jax_kernel",    "gradcheck")
# gradcheck proves differentiability; equivalence proves it computes what the
# Fortran does. Both blocking when they run; equivalence degrades to a skip
# without gfortran/meson, or for a routine that reads module state.
workflow_phase2.add_edge("gradcheck",     "equivalence")
workflow_phase2.add_edge("equivalence",   END)

translation_app_phase2 = workflow_phase2.compile()
