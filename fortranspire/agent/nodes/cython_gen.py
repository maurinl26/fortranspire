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

Two entry points per kernel. The **host** entry (``name``) takes numpy arrays and
lets OpenACC copy host↔device. The **device** entry (``name_device``) takes GPU
arrays already resident on the device — anything exposing
``__cuda_array_interface__`` (CuPy, PyTorch, Numba, RAPIDS; a JAX array via
``cupy.from_dlpack``) — reads their raw device pointer and forwards it to a
``bind(c)`` shim that wraps the call in ``!$acc data deviceptr(...)``: no
copyin/copyout, no host round-trip, zero-copy interop via the standard protocol.
The device entry is emitted only when every array extent is explicitly sizable in
the shim (an integer literal or another argument) — ``deviceptr`` forbids
assumed-size arrays.
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
# numpy typestr (kind+size, byteorder-agnostic) — for the CAI dtype check.
_C_TO_TYPESTR = {
    "double": "f8", "float": "f4", "int": "i4", "long": "i8",
    "double complex": "c16",
}
_C_TO_ITEMSIZE = {
    "double": 8, "float": 4, "int": 4, "long": 8, "double complex": 16,
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


def _array_args(kernel: dict) -> List[str]:
    """Argument names that are arrays (a device pointer can be shared for these)."""
    return [a for a in _args(kernel) if _rank(kernel, a)]


def _device_shape(kernel: dict, arg: str) -> Optional[str]:
    """Explicit-shape declaration for the deviceptr variant, or None if unsizable.

    OpenACC `deviceptr` rejects assumed-size (`a(*)`) arrays — the array must have
    an explicit shape in scope. We can size it only when every extent is an integer
    literal or another argument of the kernel (which the shim also declares). A
    module-parameter or expression extent → None → no device entry for this kernel
    (the host entry still works)."""
    dims = (kernel.get("dimensions") or {}).get(arg, [])
    argset = set(_args(kernel))
    parts: List[str] = []
    for d in dims:
        d = str(d).strip()
        if d.isdigit() or d in argset:
            parts.append(d)
        else:
            return None
    return "(" + ", ".join(parts) + ")" if parts else None


def _can_deviceptr(kernel: dict) -> bool:
    """A device entry is emitted only when every array arg is explicitly sizable."""
    arrs = _array_args(kernel)
    return bool(arrs) and all(_device_shape(kernel, a) is not None for a in arrs)


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
    # Device-pointer variant: the array args are already on the GPU (CuPy /
    # PyTorch / JAX via DLPack). `deviceptr` tells OpenACC to use the incoming
    # addresses as-is — no copyin/copyout, no host round-trip. Scalars stay host
    # (the kernel's reduction/copy handles them). Only kernels with ≥1 array arg
    # get a device entry — there is nothing to share on the device otherwise.
    for k in kernels:
        if not _can_deviceptr(k):
            continue
        arrs = _array_args(k)
        name = k["routine_name"]
        args = _args(k)
        out.append(f'  subroutine {name}_c_device({", ".join(args)}) '
                   f'bind(c, name="{name}_c_device")')
        for a in args:
            iso = _C_TO_ISO.get(_ctype(k, a), "real(c_double)")
            # explicit shape (deviceptr forbids assumed-size); scalars have none.
            shape = _device_shape(k, a) if _rank(k, a) else ""
            intent = {"IN": "in", "OUT": "out", "INOUT": "inout"}.get(_intent(k, a), "in")
            out.append(f"    {iso}, intent({intent}) :: {a}{shape}")
        out.append(f'    !$acc data deviceptr({", ".join(arrs)})')
        out.append(f'    call {name}({", ".join(args)})')
        out.append("    !$acc end data")
        out.append(f"  end subroutine {name}_c_device")
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
        if _can_deviceptr(k):  # GPU-resident (deviceptr) entry — same ABI
            out.append(f"void {name}_c_device({params});")
    out += ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */"]
    return "\n".join(out) + "\n"


def render_pyx(kernels: List[dict], module_name: str,
               header: str = "kernel_c.h") -> str:
    any_device = any(_can_deviceptr(k) for k in kernels)
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
        if _can_deviceptr(k):
            out.append(f"    void {name}_c_device({params})")
    out.append("")

    # Device-array bridge: read a raw GPU pointer from __cuda_array_interface__
    # (CuPy / PyTorch / Numba / RAPIDS; JAX via cupy.from_dlpack). Zero-copy — the
    # buffer stays on the GPU, we only hand its address to the deviceptr kernel.
    if any_device:
        out += [
            "cdef unsigned long long _device_ptr(obj, str typestr, Py_ssize_t itemsize):",
            '    """Device pointer from __cuda_array_interface__, checked for dtype & F-order."""',
            '    cai = getattr(obj, "__cuda_array_interface__", None)',
            "    if cai is None:",
            '        raise TypeError(',
            '            "expected a device array exposing __cuda_array_interface__ "',
            '            "(CuPy / PyTorch / Numba / RAPIDS; JAX via cupy.from_dlpack); got "',
            '            + type(obj).__name__ + ". Use the host entry point for numpy arrays.")',
            '    if cai["typestr"][1:] != typestr:',
            '        raise TypeError("dtype mismatch: kernel expects <" + typestr',
            '                        + ">, got " + cai["typestr"])',
            '    shape = cai["shape"]',
            '    strides = cai.get("strides")',
            "    if len(shape) > 1:",
            "        expect = []",
            "        s = itemsize",
            "        for d in shape:",
            "            expect.append(s)",
            "            s *= d",
            "        if strides is None or tuple(strides) != tuple(expect):",
            '            raise ValueError("device array must be Fortran-ordered for this "',
            '                             "kernel — use cupy.asfortranarray(x) (or order=\'F\').")',
            '    ptr = cai["data"][0]',
            "    if ptr is None:",
            '        raise ValueError("device array has no data pointer (empty array?)")',
            "    return <unsigned long long>ptr",
            "",
        ]

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

        # GPU-resident entry: array args are device arrays (CuPy/PyTorch/Numba/
        # JAX-via-dlpack), passed as device pointers to the deviceptr kernel — no
        # host round-trip. Scalars stay host. Device arrays are returned in place.
        if not _can_deviceptr(k):
            continue
        out.append(f"def {name}_device({', '.join(args)}):")
        out.append(f'    """GPU-resident: array args expose __cuda_array_interface__ '
                   f'(zero-copy)."""')
        dcall: List[str] = []
        dreturns: List[str] = []
        for a in args:
            ct = _ctype(k, a)
            if _rank(k, a):
                ts = _C_TO_TYPESTR.get(ct, "f8")
                isz = _C_TO_ITEMSIZE.get(ct, 8)
                out.append(f'    cdef unsigned long long {a}_ptr = '
                           f'_device_ptr({a}, "{ts}", {isz})')
                dcall.append(f"<{ct}*>{a}_ptr")
                if _intent(k, a) in ("OUT", "INOUT"):
                    dreturns.append(a)          # written in place on the device
            else:
                out.append(f"    cdef {ct} {a}_c = {a}")
                dcall.append(f"&{a}_c")
                if _intent(k, a) in ("OUT", "INOUT"):
                    dreturns.append(f"{a}_c")
        out.append(f"    {name}_c_device({', '.join(dcall)})")
        if len(dreturns) == 1:
            out.append(f"    return {dreturns[0]}")
        elif dreturns:
            out.append(f"    return ({', '.join(dreturns)})")
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
