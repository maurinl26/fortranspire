"""Node 5 — generate the Cython `.pyx` + `iso_c_binding` C header.

Two LLM calls (`code` stage — pure boilerplate, Codestral is sufficient
and cheaper than Mistral-Large).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from fortranspire.agent.nodes._common import SEP, _out, _save, _strip_markdown
from fortranspire.agent.nodes._state import KernelInfo, Phase1State


def cython_wrapper_agent(state: Phase1State) -> dict:
    """LLM : génère un wrapper Cython (.pyx) avec NumPy typed memoryviews."""
    print(f"\n{SEP}")
    print("  [Cython] Generating Python wrapper")
    print(SEP)

    # Code-gen stage: boilerplate-heavy .pyx + iso_c_binding header.
    # Codestral is faster and cheaper than Mistral-Large for this kind of work.
    from fortranspire.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_llm("code")

    # Only wrap compute kernels (no I/O)
    eligible = [k for k in state.get("kernel_results", []) if not k["has_io"]]
    if not eligible:
        print("  No eligible routines for Cython wrapping (all have I/O).")
        return {
            "cython_pyx": "",
            "cython_header": "",
            "cython_setup": "",
            "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
        }

    filepath    = state["fortran_filepath"]
    module_name = Path(filepath).stem.lower().replace("-", "_").replace(".", "_")

    routines_summary = [
        {
            "name":       k["routine_name"],
            "intent_map": k["intent_map"],
            "dimensions": k["dimensions"],
        }
        for k in eligible
    ]

    # ── Generate .pyx ────────────────────────────────────────────────
    pyx_system = SystemMessage(content=(
        "You are a Cython expert specializing in Fortran interoperability. "
        "Generate clean, efficient Cython wrappers with correct memory layout."
    ))
    pyx_prompt = HumanMessage(content=(
        f"Generate a Cython wrapper (.pyx) for these Fortran subroutines "
        f"compiled with nvfortran -acc (OpenACC).\n"
        f"Module name: {module_name}\n"
        f"Routines: {routines_summary}\n\n"
        f"Requirements:\n"
        f"  1. cdef extern from 'kernel_c.h' block declaring C signatures\n"
        f"  2. cpdef functions with NumPy typed memoryviews:\n"
        f"       np.float64_t[:] for 1-D arrays\n"
        f"       np.float64_t[:,:] for 2-D arrays\n"
        f"  3. np.asfortranarray() to ensure Fortran column-major layout\n"
        f"  4. import numpy as np and cimport numpy as cnp at the top\n"
        f"  5. # distutils: language = fortran  directive\n"
        f"Return ONLY the .pyx file content."
    ))

    # ── Generate C header ────────────────────────────────────────────
    header_system = SystemMessage(content="You are a C/Fortran interop expert.")
    header_prompt = HumanMessage(content=(
        f"Generate a C header file (kernel_c.h) for these Fortran subroutines "
        f"using iso_c_binding:\n{routines_summary}\n"
        f"Use double* for REAL(8) arrays, int* for INTEGER, void return type.\n"
        f"Add include guards and extern 'C' block.\n"
        f"Return ONLY the header file content."
    ))

    # ── pyproject.toml build config ──────────────────────────────────
    build_content = (
        f"[build-system]\n"
        f'requires = ["setuptools>=68", "Cython>=3.0", "numpy"]\n'
        f'build-backend = "setuptools.backends.legacy:build"\n\n'
        f"[project]\n"
        f'name = "{module_name}_gpu"\n'
        f'version = "0.1.0"\n'
        f'description = "GPU-accelerated Fortran kernel via OpenACC + Cython"\n'
        f'dependencies = ["numpy"]\n\n'
        f"# Build the Cython extension with nvfortran -acc\n"
        f"# Usage: python setup.py build_ext --inplace\n"
    )

    setup_content = (
        f"from setuptools import setup\n"
        f"from Cython.Build import cythonize\n"
        f"from setuptools.extension import Extension\n"
        f"import numpy as np\n\n"
        f'ext = Extension(\n'
        f'    name="{module_name}",\n'
        f'    sources=["cython/{module_name}.pyx", "fortran_gpu/kernel_gpu.f90"],\n'
        f'    include_dirs=["cython", np.get_include()],\n'
        f'    extra_compile_args=["-acc", "-gpu=cc80", "-Minfo=accel"],\n'
        f'    extra_link_args=["-acc", "-gpu=cc80"],\n'
        f'    language="fortran",\n'
        f')\n\n'
        f"setup(name='{module_name}_gpu', ext_modules=cythonize([ext]))\n"
    )

    pyx_code, header_code = "", ""
    try:
        resp_pyx    = llm.invoke([pyx_system, pyx_prompt])
        pyx_code    = _strip_markdown(resp_pyx.content)

        resp_header = llm.invoke([header_system, header_prompt])
        header_code = _strip_markdown(resp_header.content)

        cython_dir = _out("cython")
        _save(cython_dir / f"{module_name}.pyx", pyx_code)
        _save(cython_dir / "kernel_c.h", header_code)
        _save(Path("output") / "pyproject.toml", build_content)
        _save(Path("output") / "setup.py", setup_content)
        print(f"  Generated: {module_name}.pyx, kernel_c.h, pyproject.toml, setup.py")

    except Exception as e:
        print(f"  LLM failed for Cython wrapper: {e}")

    return {
        "cython_pyx":    pyx_code,
        "cython_header": header_code,
        "cython_setup":  build_content,
        "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
    }
