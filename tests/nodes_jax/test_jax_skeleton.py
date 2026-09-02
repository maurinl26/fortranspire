"""Deterministic JAX skeleton — element-wise kernels derived without the LLM."""
import os
import tempfile

import pytest

from fortranspire.agent.nodes_jax.jax_skeleton import _fortran_float, lower_kernel


def test_fortran_float_literals():
    assert _fortran_float("2.0d0") == "2.0e0"
    assert _fortran_float("1.5D-3") == "1.5e-3"
    assert _fortran_float("0.5_dp") == "0.5"


def _kernel_from(src: str, name: str, inputs, outputs, **over):
    """Parse `src` into a kernel dict shaped like the pipeline's."""
    pytest.importorskip("loki")
    from fortranspire.agent.nodes.parser import parser_phase1

    fd, path = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, src.encode())
        os.close(fd)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = parser_phase1({"fortran_filepath": path})
        k = next(k for k in st["kernel_results"] if k["routine_name"].lower() == name)
        k["inputs"], k["outputs"] = list(inputs), list(outputs)
        k.update(over)
        return k
    finally:
        os.remove(path)


_AXPY = """
subroutine axpy(n, a, x, y, z)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a, x(n), y(n)
  real(8), intent(out) :: z(n)
  integer :: i
  do i = 1, n
     z(i) = a * x(i) + y(i)
  end do
end subroutine axpy
"""

_STENCIL = """
subroutine sten(n, t, d)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: t(n)
  real(8), intent(out) :: d(n)
  integer :: i
  do i = 2, n
     d(i) = t(i) - t(i-1)
  end do
end subroutine sten
"""


def test_elementwise_kernel_is_derived():
    k = _kernel_from(_AXPY, "axpy", ["n", "a", "x", "y"], ["z"])
    code = lower_kernel(k)
    assert code is not None
    assert "import jax.numpy as jnp" in code
    assert "def axpy(n, a, x, y):" in code
    assert "z = ((a * x) + y)" in code
    assert "return z" in code


def test_stencil_shift_falls_back_to_the_llm():
    k = _kernel_from(_STENCIL, "sten", ["n", "t"], ["d"])
    assert lower_kernel(k) is None            # t(i-1) is not element-wise


def test_module_state_kernel_is_not_derived():
    k = _kernel_from(_AXPY, "axpy", ["n", "a", "x", "y"], ["z"],
                     free_reads=["coef"])
    assert lower_kernel(k) is None


def test_recurrence_is_not_derived():
    k = _kernel_from(_AXPY, "axpy", ["n", "a", "x", "y"], ["z"],
                     has_loop_carried_dep=True)
    assert lower_kernel(k) is None


def test_derived_code_is_importable_and_differentiable():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    k = _kernel_from(_AXPY, "axpy", ["n", "a", "x", "y"], ["z"])
    code = lower_kernel(k)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - our own derived code
    z = ns["axpy"](8, 2.0, jnp.arange(4.0), jnp.ones(4))
    assert jnp.allclose(z, 2.0 * jnp.arange(4.0) + 1.0)
    g = jax.grad(lambda x: jnp.sum(ns["axpy"](8, 2.0, x, jnp.ones(4))))(jnp.arange(4.0))
    assert jnp.allclose(g, 2.0)               # d/dx (2x) = 2
