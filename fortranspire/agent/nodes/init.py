"""Node 0 — initialise the output directories and probe the GPU compiler.

Deterministic, no LLM. Always runs first.
"""
from __future__ import annotations

from fortranspire.agent.nodes._common import SEP, _gpu_compiler, _out
from fortranspire.agent.nodes._state import Phase1State


def init_phase1(state: Phase1State) -> dict:
    """Create the per-category `output/` tree and probe the GPU compiler."""
    print(f"\n{SEP}")
    print("  [Init] Fortran GPU Phase 1")
    print(SEP)

    for d in ["fortran_gpu", "cython"]:
        _out(d)
        print(f"  Dir : output/{d}/")

    compiler = _gpu_compiler()
    if compiler:
        print(f"  GPU compiler : {compiler}")
    else:
        print("  WARNING: nvfortran/pgfortran not found. Set FC env var or install NVIDIA HPC SDK.")
        print("           Compilation step will be skipped in validation.")

    print(SEP + "\n")
    return {}
