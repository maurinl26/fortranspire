"""GT4Py driver generation + static domain/halo validation (issue #82).

The domain, halos and offset providers live in the driver, not the field
operator, and the type check cannot see them. This generates the driver
deterministically from the typed domain model (#84) and checks it
statically — no gt4py execution, so no toolchain needed (a Cartesian shift
only runs on the compiled gtfn backend).
"""
from __future__ import annotations

import pytest

from fortranspire.agent.domain_model import build_domain_model
from fortranspire.agent.nodes_gt4py.driver import (
    build_driver,
    domain_check_agent,
    field_offsets_used,
    interior_domain,
    validate_domain,
)

DIFFUSE = """subroutine diffuse(t, tnew, kdiff, nlev)
  implicit none
  integer, intent(in) :: nlev
  real(kind=8), intent(in) :: t(nlev), kdiff
  real(kind=8), intent(out) :: tnew(nlev)
  integer :: k
  do k = 2, nlev-1
     tnew(k) = t(k) + kdiff * (t(k+1) - 2.0d0*t(k) + t(k-1))
  end do
end subroutine diffuse
"""


@pytest.mark.slow
class TestInteriorDomain:
    def test_offsets_restrict_the_domain(self):
        """offsets {-1,+1} → interior [1, n-1): the boundary layers are read,
        not written."""
        m = build_domain_model(DIFFUSE)
        dom = interior_domain(m)
        assert dom["K"] == ("1", "n_K - 1")

    def test_no_shift_leaves_the_full_domain(self):
        src = ("subroutine s(a, b, n)\n  integer :: n\n"
               "  real(8), intent(in) :: a(n)\n  real(8), intent(out) :: b(n)\n"
               "  integer :: i\n  do i=1,n\n    b(i) = a(i) * 2.0d0\n  end do\n"
               "end subroutine s\n")
        m = build_domain_model(src)
        assert field_offsets_used(m) == {}
        for lo, hi in interior_domain(m).values():
            assert lo == "0" and "-" not in hi


@pytest.mark.slow
class TestDriverGeneration:
    def test_driver_declares_dims_offset_and_domain(self):
        m = build_domain_model(DIFFUSE)
        driver = build_driver(m, "diffuse")
        assert "gtx.Dimension(\"K\"" in driver
        assert "FieldOffset(\"Koff\"" in driver
        assert "CartesianConnectivity(K)" in driver
        assert "domain = {K: (1, n_K - 1)}" in driver
        assert "out=tnew" in driver

    def test_driver_is_valid_python(self):
        import ast

        m = build_domain_model(DIFFUSE)
        ast.parse(build_driver(m, "diffuse"))


@pytest.mark.slow
class TestStaticValidation:
    def test_halo_reported_and_boundary_noted(self):
        m = build_domain_model(DIFFUSE)
        report = validate_domain(m, "return 0.5*(t + t(Koff[-1]))")
        assert report.ok is True
        assert report.halo == {"K": 1}
        assert any("boundary layers" in n for n in report.notes)

    def test_a_shift_with_no_matching_offset_is_flagged(self):
        """The operator shifts by an offset the model never found — the
        halo/domain would be wrong."""
        m = build_domain_model(DIFFUSE)  # model knows only Koff
        report = validate_domain(m, "return a(Joff[1])")  # stray Joff
        assert report.ok is False
        assert report.problems

    def test_scalars_only_needs_no_domain(self):
        src = ("subroutine s(a, b)\n  real(8), intent(in) :: a\n"
               "  real(8), intent(out) :: b\n  b = a\nend subroutine s\n")
        m = build_domain_model(src)
        report = validate_domain(m)
        assert report.ok and any("scalars only" in n for n in report.notes)


class TestNodeAndFinding:
    def test_domain_check_is_the_last_gt4py_node(self):
        pytest.importorskip("langgraph")
        from fortranspire.agent.translation_graph_gt4py import translation_app_gt4py

        names = [n for n in translation_app_gt4py.get_graph().nodes if not n.startswith("__")]
        assert names[-1] == "domain_check"
        assert names.index("type_check") < names.index("domain_check")

    def test_fort033_registered_and_labelled(self):
        from fortranspire.agent.analyze import RULES
        from fortranspire.agent.explain import _RISK_LABELS

        assert RULES["FORT033"]["severity"] == "note"
        assert "FORT033" in _RISK_LABELS

    def test_fort033_light_halo_scan(self):
        from fortranspire.agent.analyze import _gt4py_halo

        assert _gt4py_halo("x = a(k+2) + a(k-1)") == 2
        assert _gt4py_halo("x = a(i) * b(j)") == 0
        assert _gt4py_halo("! a(k+5) in a comment\nx = a(k)") == 0
