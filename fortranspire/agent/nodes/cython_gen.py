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
the shim (an integer literal, another argument, or an integer module PARAMETER we
can resolve — imported with a generated ``use …, only:``) — ``deviceptr`` forbids
assumed-size arrays — and every array dtype has an iso_c_binding equivalent (half
precision has none, so it stays on the dtype-agnostic JAX/DLPack path).
"""
from __future__ import annotations

from typing import Dict, List, Optional

_C_TO_NP = {
    "double": "np.float64", "float": "np.float32",
    "int": "np.int32", "long": "np.int64",
    "short": "np.int16", "signed char": "np.int8",
    "double complex": "np.complex128", "float complex": "np.complex64",
}
_C_TO_ISO = {
    "double": "real(c_double)", "float": "real(c_float)",
    "int": "integer(c_int)", "long": "integer(c_long)",
    "short": "integer(c_short)", "signed char": "integer(c_signed_char)",
    "double complex": "complex(c_double_complex)",
    "float complex": "complex(c_float_complex)",
}
# numpy typestr (kind+size, byteorder-agnostic) — for the CAI dtype check.
# Half precision (f2/bf16) has no iso_c_binding C type, so it cannot cross the
# bind(c) boundary — those live only on the dtype-agnostic JAX/DLPack path.
_C_TO_TYPESTR = {
    "double": "f8", "float": "f4", "int": "i4", "long": "i8",
    "short": "i2", "signed char": "i1",
    "double complex": "c16", "float complex": "c8",
}
_C_TO_ITEMSIZE = {
    "double": 8, "float": 4, "int": 4, "long": 8, "short": 2,
    "signed char": 1, "double complex": 16, "float complex": 8,
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


def _resolved(kernel: dict) -> dict:
    return kernel.get("resolved") or {}


def _extent_param(kernel: dict, d: str) -> Optional[dict]:
    """The resolved record for `d` when it is an integer module PARAMETER, else None.

    Such a parameter is a legal explicit extent: it is a compile-time constant that
    the shim brings into scope (it `use`s the kernels module, and we add a `use …,
    only:` for any parameter imported from another module)."""
    r = _resolved(kernel).get(str(d).strip().lower())
    if r and r.get("is_parameter") and r.get("dtype") == "integer":
        return r
    return None


def _device_shape(kernel: dict, arg: str) -> Optional[str]:
    """Explicit-shape declaration for the deviceptr variant, or None if unsizable.

    OpenACC `deviceptr` rejects assumed-size (`a(*)`) arrays — the array must have
    an explicit shape in scope. An extent is sizable when it is an integer literal,
    another argument of the kernel, or an integer module PARAMETER we can resolve
    (in scope via `use`). Any other extent (a non-parameter variable, an expression)
    → None → no device entry for this kernel (the host entry still works)."""
    dims = (kernel.get("dimensions") or {}).get(arg, [])
    argset = set(_args(kernel))
    parts: List[str] = []
    for d in dims:
        d = str(d).strip()
        if d.isdigit() or d in argset or _extent_param(kernel, d) is not None:
            parts.append(d)
        else:
            return None
    return "(" + ", ".join(parts) + ")" if parts else None


def _can_deviceptr(kernel: dict) -> bool:
    """Emit a device entry only when every array arg is explicitly sizable AND has a
    C dtype the CAI bridge maps (an unmappable dtype — e.g. half precision, which
    iso_c_binding cannot express — falls back to the host entry, never a wrong
    check)."""
    arrs = _array_args(kernel)
    if not arrs:
        return False
    if any(_ctype(kernel, a) not in _C_TO_TYPESTR for a in arrs):
        return False
    return all(_device_shape(kernel, a) is not None for a in arrs)


def _extent_param_imports(kernels: List[dict], kernels_module: str) -> Dict[str, List[str]]:
    """{module: sorted parameters} for extent params imported from OTHER modules.

    The shim already `use`s the kernels module, so a parameter defined there needs
    no extra import; one coming from a sibling module (the common CMAQ case — an
    array dimensioned by `rbdata_mod`'s `NRXN`) does."""
    imports: Dict[str, set] = {}
    for k in kernels:
        if not _can_deviceptr(k):
            continue
        argset = set(_args(k))
        for a in _array_args(k):
            for d in (k.get("dimensions") or {}).get(a, []):
                d = str(d).strip()
                if d.isdigit() or d in argset:
                    continue
                rec = _extent_param(k, d)
                if rec is None:
                    continue
                mod = rec.get("module") or ""
                if mod and mod.lower() != kernels_module.lower():
                    imports.setdefault(mod, set()).add(d)
    return {m: sorted(v) for m, v in imports.items()}


def render_shim(kernels: List[dict], module_name: str) -> str:
    """One `bind(c)` C-API subroutine per kernel, forwarding into the module."""
    out: List[str] = [
        "! Auto-generated bind(c) C API — deterministic, do not edit by hand.",
        "module fortranspire_c_api",
        "  use iso_c_binding",
        f"  use {module_name}",
    ]
    # extent parameters imported from sibling modules must be in scope for the
    # device shim's explicit-shape declarations (e.g. `a(NRXN)`).
    for mod, params in sorted(_extent_param_imports(kernels, module_name).items()):
        out.append(f"  use {mod}, only: {', '.join(params)}")
    out += [
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
