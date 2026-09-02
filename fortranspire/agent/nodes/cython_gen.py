"""Deterministic Cython/C wrapper generation — no LLM.

The Cython wrapper is pure boilerplate: the argument list, the C types, which
arguments come back — all of it is derived from the parsed INTENT map, the KIND
(``arg_ctypes``) and the array dimensions. Having an LLM *guess* an ABI is the
"guess what can be derived" anti-pattern, and an ABI the model gets subtly wrong
is a silent, hard-to-debug failure. This renders the three artifacts from the
AST instead, so they are reproducible, token-free and internally consistent by
construction.

ABI: a ``bind(c)`` Fortran shim per kernel gives each an **unmangled** C name
that the header declares and the ``.pyx`` calls — no compiler name-mangling
guesswork. The shim ``use``s the kernels module and forwards the call; Fortran
passes everything by reference, so every C argument (scalar or array) is a
pointer. Arrays are made Fortran-contiguous (``np.asfortranarray``) before the
pointer is taken.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_C_TO_NP = {
    "double": "np.float64", "float": "np.float32",
    "int": "np.int32", "long": "np.int64",
    "double complex": "np.complex128",
}
_C_TO_ISO = {
    "double": "real(c_double)", "float": "real(c_float)",
    "int": "integer(c_int)", "long": "integer(c_long)",
    "double complex": "complex(c_double_complex)",
}


def _args(kernel: dict) -> List[str]:
    """Argument names in declaration order (INTENT map preserves it)."""
    return list((kernel.get("intent_map") or {}).keys())


def _intent(kernel: dict, arg: str) -> str:
    return (kernel.get("intent_map") or {}).get(arg, "IN").upper()


def _ctype(kernel: dict, arg: str) -> str:
    return (kernel.get("arg_ctypes") or {}).get(arg, "double")


def _rank(kernel: dict, arg: str) -> int:
    return len((kernel.get("dimensions") or {}).get(arg, []))


def render_shim(kernels: List[dict], module_name: str) -> str:
    """One `bind(c)` C-API subroutine per kernel, forwarding into the module."""
    out: List[str] = [
        "! Auto-generated bind(c) C API — deterministic, do not edit by hand.",
        "module fortranspire_c_api",
        "  use iso_c_binding",
        f"  use {module_name}",
        "  implicit none",
        "contains",
    ]
    for k in kernels:
        name = k["routine_name"]
        args = _args(k)
        out.append(f'  subroutine {name}_c({", ".join(args)}) '
                   f'bind(c, name="{name}_c")')
        for a in args:
            iso = _C_TO_ISO.get(_ctype(k, a), "real(c_double)")
            shape = "(*)" if _rank(k, a) else ""
            intent = {"IN": "in", "OUT": "out", "INOUT": "inout"}.get(_intent(k, a), "in")
            out.append(f"    {iso}, intent({intent}) :: {a}{shape}")
        out.append(f'    call {name}({", ".join(args)})')
        out.append(f"  end subroutine {name}_c")
    out.append("end module fortranspire_c_api")
    return "\n".join(out) + "\n"


def render_header(kernels: List[dict], guard: str = "FORTRANSPIRE_KERNEL_C_H") -> str:
    out: List[str] = [
        "/* Auto-generated C header — deterministic, do not edit by hand. */",
        f"#ifndef {guard}", f"#define {guard}", "",
        '#ifdef __cplusplus', 'extern "C" {', "#endif", "",
    ]
    for k in kernels:
        name = k["routine_name"]
        params = ", ".join(f"{_ctype(k, a)}* {a}" for a in _args(k)) or "void"
        out.append(f"void {name}_c({params});")
    out += ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */"]
    return "\n".join(out) + "\n"


def render_pyx(kernels: List[dict], module_name: str,
               header: str = "kernel_c.h") -> str:
    out: List[str] = [
        "# distutils: language = c",
        "# Auto-generated Cython wrapper — deterministic, do not edit by hand.",
        "import numpy as np",
        "cimport numpy as cnp",
        "cnp.import_array()",
        "",
        f'cdef extern from "{header}":',
    ]
    for k in kernels:
        name = k["routine_name"]
        params = ", ".join(f"{_ctype(k, a)}* {a}" for a in _args(k))
        out.append(f"    void {name}_c({params})")
    out.append("")

    for k in kernels:
        name = k["routine_name"]
        args = _args(k)
        out.append(f"def {name}({', '.join(args)}):")
        call_parts: List[str] = []
        returns: List[str] = []
        for a in args:
            ct = _ctype(k, a)
            npd = _C_TO_NP.get(ct, "np.float64")
            if _rank(k, a):  # array → Fortran-contiguous, pass its data pointer
                out.append(f"    cdef cnp.ndarray {a}_arr = "
                           f"np.asfortranarray({a}, dtype={npd})")
                call_parts.append(f"<{ct}*>cnp.PyArray_DATA({a}_arr)")
                if _intent(k, a) in ("OUT", "INOUT"):
                    returns.append(f"{a}_arr")
            else:            # scalar → a C local, passed by reference
                out.append(f"    cdef {ct} {a}_c = {a}")
                call_parts.append(f"&{a}_c")
                if _intent(k, a) in ("OUT", "INOUT"):
                    returns.append(f"{a}_c")
        out.append(f"    {name}_c({', '.join(call_parts)})")
        if len(returns) == 1:
            out.append(f"    return {returns[0]}")
        elif returns:
            out.append(f"    return ({', '.join(returns)})")
        out.append("")
    return "\n".join(out) + "\n"


def generate_cython(kernels: List[dict], module_name: str,
                    kernels_module: Optional[str] = None) -> Dict[str, str]:
    """Return {'pyx', 'header', 'shim'} — the deterministic wrapper set."""
    km = kernels_module or f"{module_name}_kernels"
    return {
        "pyx": render_pyx(kernels, module_name),
        "header": render_header(kernels),
        "shim": render_shim(kernels, km),
    }
