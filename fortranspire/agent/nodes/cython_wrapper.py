"""Node 5 — generate the Cython `.pyx` + C header + bind(c) shim.

Deterministic, no LLM. The wrapper is pure boilerplate derived from the parsed
INTENT map, KIND (``arg_ctypes``) and dimensions — see :mod:`.cython_gen`. An LLM
guessing an ABI is the "guess what can be derived" anti-pattern, and an ABI it
gets subtly wrong is a silent failure; deriving it makes the three artifacts
reproducible, token-free and internally consistent by construction.
"""
from __future__ import annotations

import re
from pathlib import Path

from fortranspire.agent.nodes._common import SEP, _out, _save
from fortranspire.agent.nodes._state import Phase1State
from fortranspire.agent.nodes.cython_gen import generate_cython


def _kernels_module_name(state: Phase1State, default: str) -> str:
    """The module the shim must `use` — the one holding the compute kernels."""
    src = state.get("module_fortran") or ""
    m = re.search(r"^\s*MODULE\s+(\w+)", src, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else default


def cython_wrapper_agent(state: Phase1State) -> dict:
    """Generate the Cython/C wrapper deterministically from the AST."""
    print(f"\n{SEP}")
    print("  [Cython] Generating Python wrapper (deterministic, no LLM)")
    print(SEP)

    # Only wrap compute kernels (no I/O)
    eligible = [k for k in state.get("kernel_results", []) if not k.get("has_io")]
    if not eligible:
        print("  No eligible routines for Cython wrapping (all have I/O).")
        return {
            "cython_pyx": "", "cython_header": "", "cython_setup": "",
            "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
        }

    filepath    = state["fortran_filepath"]
    module_name = Path(filepath).stem.lower().replace("-", "_").replace(".", "_")
    kernels_mod = _kernels_module_name(state, f"{module_name}_kernels")

    art = generate_cython(eligible, module_name, kernels_module=kernels_mod)

    build_content = (
        f"[build-system]\n"
        f'requires = ["setuptools>=68", "Cython>=3.0", "numpy"]\n'
        f'build-backend = "setuptools.build_meta"\n\n'
        f"[project]\n"
        f'name = "{module_name}_gpu"\n'
        f'version = "0.1.0"\n'
        f'description = "GPU-accelerated Fortran kernel via OpenACC + Cython"\n'
        f'dependencies = ["numpy"]\n'
    )

    # The shim + the OpenACC kernels compile together; the .pyx links against them.
    setup_content = (
        f"from setuptools import setup\n"
        f"from Cython.Build import cythonize\n"
        f"from setuptools.extension import Extension\n"
        f"import numpy as np\n\n"
        f'ext = Extension(\n'
        f'    name="{module_name}",\n'
        f'    sources=["cython/{module_name}.pyx", "fortran_gpu/kernel_gpu.f90",\n'
        f'             "fortran_gpu/{module_name}_c_api.f90"],\n'
        f'    include_dirs=["cython", np.get_include()],\n'
        f'    extra_compile_args=["-acc", "-gpu=cc80", "-Minfo=accel"],\n'
        f'    extra_link_args=["-acc", "-gpu=cc80"],\n'
        f')\n\n'
        f"setup(name='{module_name}_gpu', ext_modules=cythonize([ext]))\n"
    )

    cython_dir = _out("cython")
    _save(cython_dir / f"{module_name}.pyx", art["pyx"])
    _save(cython_dir / "kernel_c.h", art["header"])
    _save(_out("fortran_gpu") / f"{module_name}_c_api.f90", art["shim"])
    _save(Path("output") / "pyproject.toml", build_content)
    _save(Path("output") / "setup.py", setup_content)
    print(f"  Generated: {module_name}.pyx, kernel_c.h, {module_name}_c_api.f90 "
          f"(bind(c) shim → module {kernels_mod}), pyproject.toml, setup.py")

    return {
        "cython_pyx":    art["pyx"],
        "cython_header": art["header"],
        "cython_shim":   art["shim"],
        "cython_setup":  build_content,
        "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
    }
