"""Deterministic DLPack / CAI interop for the JAX port — no LLM.

The emitted kernels are pure ``jnp`` functions: they accept ``jax.Array`` but
not, zero-copy, a CuPy / PyTorch / Numba device array. **DLPack** is the
cross-framework zero-copy protocol every GPU-array library implements;
``__cuda_array_interface__`` (CAI) is its CUDA-only sibling. This renders a small
adapter so a kernel can be called on *any* device array with no host round-trip,
and its result handed back to the caller's framework — interop via the standard
**protocol**, not a hard dependency on one library (CuPy is only one consumer).

Pure template: reproducible, token-free.
"""
from __future__ import annotations

from typing import List

_PRELUDE = '''"""Auto-generated DLPack / __cuda_array_interface__ interop — do not edit by hand.

Zero-copy bridge between this JAX port and any device-array framework — CuPy,
PyTorch, Numba, RAPIDS — via DLPack. Import a foreign array with ``as_jax`` (or
call a kernel through ``call``), export a result with ``to_cupy`` / ``to_torch``.
The bridge is zero-copy when the source already lives on the JAX device's GPU;
host (numpy) inputs fall back to a device copy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def as_jax(x):
    """Bring any array onto the JAX device, zero-copy where possible.

    Python scalars (int/float/bool) and None pass through untouched — they are
    static kernel arguments (shapes, bounds), not buffers to move. A ``jax.Array``
    is returned unchanged. Anything exposing ``__dlpack__`` (CuPy, PyTorch, Numba,
    another framework) is imported zero-copy on the same GPU via DLPack; CuPy's
    legacy ``toDlpack()`` is honoured. numpy / host arrays fall back to a copy.
    """
    if x is None or isinstance(x, (int, float, bool)):
        return x
    if isinstance(x, jax.Array):
        return x
    if hasattr(x, "__dlpack__"):
        return jax.dlpack.from_dlpack(x)
    if hasattr(x, "toDlpack"):          # legacy CuPy capsule
        return jax.dlpack.from_dlpack(x.toDlpack())
    return jnp.asarray(x)               # numpy / host → device copy


def call(fn, *args):
    """Call a pure JAX kernel on arrays from any framework (inputs coerced)."""
    return fn(*(as_jax(a) for a in args))


def to_cupy(x):
    """Export a jax.Array to CuPy, zero-copy on the same GPU (needs cupy)."""
    import cupy
    return cupy.from_dlpack(x)


def to_torch(x):
    """Export a jax.Array to PyTorch, zero-copy on the same device (needs torch)."""
    import torch.utils.dlpack as _td
    return _td.from_dlpack(x)
'''


def render_interop(kernel_names: List[str]) -> str:
    """Render ``_interop.py``: the shared bridge + one wrapper per kernel.

    Each ``<name>`` wrapper imports the sibling emitted kernel and calls it with
    inputs coerced via :func:`as_jax`, so ``<name>(cupy_or_torch_array, ...)``
    works zero-copy. Kernel names are sorted for byte-reproducible output.
    """
    out = [_PRELUDE]
    for name in sorted(set(kernel_names)):
        out.append(
            f"\ndef {name}(*args):\n"
            f'    """Call the `{name}` kernel on arrays from any framework '
            f'(DLPack/CAI); returns a jax.Array."""\n'
            f"    from {name} import {name} as _kernel   # sibling emitted kernel\n"
            f"    return call(_kernel, *args)\n"
        )
    return "".join(out)
