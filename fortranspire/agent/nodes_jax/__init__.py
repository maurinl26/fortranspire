"""Phase 2 pipeline nodes — Fortran → functional refactoring → JAX.

Mirrors :mod:`fortranspire.agent.nodes` (Phase 1). The two pipelines share
``parser`` and ``extractor``: parsing is the same work, and the extractor's
promotion of ``COMMON`` / ``SAVE`` state to explicit arguments is the first
half of functionalisation regardless of the target.

They diverge after that. Phase 1 annotates purity to satisfy OpenACC;
Phase 2 turns it into a functional interface and emits JAX.
"""
from fortranspire.agent.nodes_jax.functionalize import functionalize_agent
from fortranspire.agent.nodes_jax.gradcheck import GradcheckError, check_kernel, gradcheck_agent
from fortranspire.agent.nodes_jax.jax_kernel import jax_kernel_agent

__all__ = [
    "functionalize_agent",
    "jax_kernel_agent",
    "gradcheck_agent",
    "check_kernel",
    "GradcheckError",
]
