"""Deterministic OpenACC pragma derivation (reductions, collapse, private).

The headline is the reduction clause: a scalar accumulator parallelised without
`reduction(op:var)` is a silent GPU race. These pin that it is derived.
"""
import os
import tempfile

import pytest

from fortranspire.agent.nodes.openacc_gen import (
    analyse_loop,
    insert_pragma,
    render_pragma,
)

pytest.importorskip("loki")


def _outer_loop(src: str):
    from loki import Sourcefile, FindNodes
    from loki.ir.nodes import Loop

    fd, path = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, src.encode())
        os.close(fd)
        r = Sourcefile.from_file(path).routines[0]
        return FindNodes(Loop).visit(r.body)[0]
    finally:
        os.remove(path)


_SUM = """
subroutine k(n, a, s)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: s
  real(8) :: tmp
  integer :: i
  do i = 1, n
     tmp = a(i) * 2.0d0
     s = s + tmp
  end do
end subroutine k
"""

_MINMAX = """
subroutine k(n, a, lo, hi)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: lo, hi
  integer :: i
  do i = 1, n
     lo = min(lo, a(i))
     hi = max(hi, a(i))
  end do
end subroutine k
"""

_NEST3 = """
subroutine k(nx, ny, nz, u, v)
  integer, intent(in) :: nx, ny, nz
  real(8), intent(in) :: u(nx, ny, nz)
  real(8), intent(out) :: v(nx, ny, nz)
  integer :: i, j, l
  do l = 1, nz
   do j = 1, ny
    do i = 1, nx
      v(i,j,l) = u(i,j,l) * 2.0d0
    end do
   end do
  end do
end subroutine k
"""


def test_sum_reduction_and_private():
    info = analyse_loop(_outer_loop(_SUM))
    assert info.reductions == {"s": "+"}
    assert info.privates == {"tmp"}
    assert "reduction(+:s)" in render_pragma(info)
    assert "private(tmp)" in render_pragma(info)


def test_min_and_max_reductions():
    info = analyse_loop(_outer_loop(_MINMAX))
    assert info.reductions == {"lo": "min", "hi": "max"}
    p = render_pragma(info)
    assert "reduction(min:lo)" in p and "reduction(max:hi)" in p


def test_collapse_from_real_nest_depth():
    info = analyse_loop(_outer_loop(_NEST3))
    assert info.depth == 3
    assert "collapse(3)" in render_pragma(info)      # not a hard-coded 2


def test_carried_dependency_is_sequential_not_parallel():
    info = analyse_loop(_outer_loop(_SUM), carried=True)
    assert render_pragma(info) == "!$acc loop seq"


def test_omp_target_variant():
    info = analyse_loop(_outer_loop(_NEST3))
    p = render_pragma(info, gpu_pragma="omp")
    assert p.startswith("!$omp target teams distribute parallel do")
    assert "collapse(3)" in p


def test_insert_pragma_before_the_loop_keeps_indent():
    src = "  do i = 1, n\n    x = 1\n  end do\n"
    out = insert_pragma(src, "i", "!$acc parallel loop")
    lines = out.splitlines()
    assert lines[0].strip() == "!$acc parallel loop"
    assert lines[0].startswith("  ")                 # matches the do indent
    assert lines[1].strip().startswith("do i")
