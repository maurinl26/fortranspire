"""DLPack / __cuda_array_interface__ interop shim for the JAX port.

The emitted kernels are pure ``jnp`` functions; the interop shim lets them be
called on arrays from any framework (CuPy, PyTorch, Numba) via the standard
zero-copy protocols — no CuPy dependency. We test the rendered module for
structure and validity, and exercise ``as_jax`` / the per-kernel wrapper on the
host (numpy) path (the device branches need a GPU and are covered by the
protocol dispatch).
"""
import ast
import sys

import numpy as np

from fortranspire.agent.nodes_jax.interop import render_interop


def test_rendered_interop_is_valid_python_and_has_the_bridge():
    src = render_interop(["scale_arr"])
    ast.parse(src)                                   # valid module
    for fn in ("def as_jax", "def call", "def to_cupy", "def to_torch"):
        assert fn in src
    assert "jax.dlpack.from_dlpack" in src           # the zero-copy import path
    assert "__cuda_array_interface__" not in src or "DLPack" in src


def test_one_wrapper_per_kernel_sorted_for_reproducibility():
    src = render_interop(["beta", "alpha", "beta"])   # dup + unsorted input
    assert src.count("def alpha(") == 1 and src.count("def beta(") == 1
    assert src.index("def alpha(") < src.index("def beta(")   # sorted → reproducible


def test_as_jax_moves_numpy_passes_scalars_and_keeps_jax(tmp_path):
    src = render_interop(["k"])
    mod = _load(src, tmp_path, "iop_a")
    import jax
    import jax.numpy as jnp

    x = np.arange(4, dtype=np.float64)
    jx = mod.as_jax(x)
    assert isinstance(jx, jax.Array) and jnp.allclose(jx, x)
    assert mod.as_jax(jx) is jx                       # already-jax passes through
    assert mod.as_jax(3) == 3 and mod.as_jax(None) is None   # static scalars untouched


def test_call_coerces_inputs_and_runs_the_kernel(tmp_path):
    src = render_interop(["k"])
    mod = _load(src, tmp_path, "iop_b")
    import jax.numpy as jnp

    def double(a):                                    # a "pure jax kernel"
        return 2.0 * a

    out = mod.call(double, np.arange(3, dtype=np.float64))
    assert jnp.allclose(out, jnp.asarray([0.0, 2.0, 4.0]))


def _load(src: str, tmp_path, name: str):
    """Import a rendered interop module from a temp file."""
    p = tmp_path / f"{name}.py"
    p.write_text(src)
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))
