"""Cross-module symbol resolution — Stage 0 of #99.

Loki-dependent, so the whole module skips without the extra.
"""
import os

import pytest

pytest.importorskip("loki")

from fortranspire.agent.resolve import resolve_modules, resolve_for_routine  # noqa: E402


_MODULE = """
module netdata
  implicit none
  integer, allocatable :: irm2(:,:,:)   ! species reaction table
  real(8), allocatable :: rki(:,:)      ! rate constants
  integer :: numcells
  integer, parameter :: nrxns = 42
end module netdata
"""

_ROUTINE = """
subroutine rhs(ncsp, yin, ydot)
  use netdata
  implicit none
  integer, intent(in) :: ncsp
  real(8), intent(in) :: yin(:,:)
  real(8), intent(out) :: ydot(:,:)
  ydot = 0.0d0
end subroutine rhs
"""


def _write(dir_, name, src):
    path = os.path.join(dir_, name)
    with open(path, "w") as f:
        f.write(src)
    return path


def test_resolve_modules_reads_declarations(tmp_path):
    _write(str(tmp_path), "netdata.f90", _MODULE)
    res = resolve_modules(["netdata"], (str(tmp_path),))

    assert res["irm2"].dtype == "integer" and res["irm2"].rank == 3
    assert res["rki"].dtype == "real" and res["rki"].rank == 2
    assert res["numcells"].dtype == "integer" and res["numcells"].rank == 0
    assert res["nrxns"].is_parameter


def test_resolve_for_routine_follows_use(tmp_path):
    from loki import Sourcefile

    _write(str(tmp_path), "netdata.f90", _MODULE)
    rpath = _write(str(tmp_path), "rhs.f90", _ROUTINE)
    routine = Sourcefile.from_file(rpath).routines[0]

    res = resolve_for_routine(routine, (str(tmp_path),))
    # symbols reachable through `use netdata`, resolved to their declarations:
    assert {"irm2", "rki", "numcells"} <= set(res)
    assert res["irm2"].rank == 3


def test_missing_module_resolves_nothing(tmp_path):
    # No file defines `nowhere` → empty, never an error.
    assert resolve_modules(["nowhere"], (str(tmp_path),)) == {}
