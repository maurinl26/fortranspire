"""`recon` — arrival triage / ranked porting worklist.

The pure ranking logic is tested directly; the end-to-end survey (which needs
Loki) skips where the extra is absent.
"""
import os
import tempfile

import pytest

from fortranspire.agent.recon import RoutineTarget, _rank, survey


# ── ranking logic (no Loki) ─────────────────────────────────────────────────

def test_driver_is_never_a_target():
    t = RoutineTarget(name="main", file="x.f90", role="driver", has_io=True)
    target, rank, reason = _rank(t)
    assert target == "none" and rank == 0.0
    assert "driver" in reason


def test_blocked_kernel_scores_zero():
    t = RoutineTarget(name="k", file="x.f90", role="kernel",
                      purity="blocked", jax_score=0, gt4py_score=0)
    target, rank, _ = _rank(t)
    assert target == "none" and rank == 0.0


def test_pure_leaf_ranks_above_a_stateful_one():
    light = RoutineTarget(name="a", file="x", role="kernel", purity="pure",
                          jax_score=5, is_leaf=True, n_free_reads=2)
    heavy = RoutineTarget(name="b", file="x", role="kernel", purity="pure",
                          jax_score=5, is_leaf=True, n_free_reads=18)
    _, rank_light, _ = _rank(light)
    _, rank_heavy, _ = _rank(heavy)
    assert rank_light > rank_heavy          # less state to promote ⇒ better target


def test_target_dsl_follows_the_higher_score():
    jaxy = RoutineTarget(name="a", file="x", role="kernel", purity="pure",
                         jax_score=5, gt4py_score=3)
    meshy = RoutineTarget(name="b", file="x", role="kernel", purity="threaded",
                          jax_score=3, gt4py_score=5)
    assert _rank(jaxy)[0] == "jax"
    assert _rank(meshy)[0] == "gt4py"


# ── end-to-end survey (needs Loki) ──────────────────────────────────────────

_KERNEL = """
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

_DRIVER = """
subroutine run(n, x)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  print *, 'running'          ! I/O ⇒ driver
end subroutine run
"""


def test_survey_separates_kernels_from_drivers():
    pytest.importorskip("loki")
    with tempfile.TemporaryDirectory() as d:
        (open(os.path.join(d, "axpy.f90"), "w")).write(_KERNEL)
        (open(os.path.join(d, "run.f90"), "w")).write(_DRIVER)
        targets = {t.name.lower(): t for t in survey([d])}

    assert "axpy" in targets and "run" in targets
    assert targets["axpy"].role == "kernel"
    assert targets["axpy"].rank > 0
    assert targets["axpy"].target == "jax"
    assert targets["run"].role == "driver"
    assert targets["run"].rank == 0.0
