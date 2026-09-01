"""Typed domain model — scan Fortran, propose the typed Python equivalent.

Deterministic, Loki-based, target-agnostic: it derives dtypes, dimensions,
axis roles, and stencil offsets once, so both the JAX and gt4py.next
emitters read one typed model instead of re-inferring types from the
source. It also gives the GT4Py driver / halo work (issue #82) its input.
"""
from __future__ import annotations

import pytest

from fortranspire.agent.domain_model import (
    _python_dtype,
    build_domain_model,
    to_gt4py_hints,
    to_jax_hints,
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


class TestDtypeMapping:
    @pytest.mark.parametrize("basic,kind,expected", [
        ("BasicType.REAL", "8", "float64"),
        ("BasicType.REAL", "4", "float32"),
        ("BasicType.REAL", None, "float64"),   # ambiguous, defaults
        ("BasicType.INTEGER", None, "int64"),
        ("BasicType.INTEGER", "4", "int32"),
        ("BasicType.LOGICAL", None, "bool"),
    ])
    def test_kind_maps_to_python_dtype(self, basic, kind, expected):
        assert _python_dtype(basic, kind) == expected


@pytest.mark.slow  # shells out to Loki
class TestExtraction:
    @pytest.fixture(scope="class")
    def model(self):
        return build_domain_model(DIFFUSE)

    def test_routine_and_loop_axis(self, model):
        assert model.routine == "diffuse"
        assert model.loop_axes == {"k": "nlev"}

    def test_field_dtypes_and_ranks(self, model):
        t = model.field("t")
        assert t.dtype == "float64" and t.rank == 1 and t.intent == "IN"
        assert model.field("tnew").intent == "OUT"

    def test_scalars_are_scalars(self, model):
        assert model.field("kdiff").is_scalar
        assert model.field("nlev").is_scalar and model.field("nlev").dtype.startswith("int")

    def test_vertical_axis_inferred(self, model):
        """`nlev` reads as the vertical axis — heuristic, and flagged."""
        axis = model.field("t").axes[0]
        assert axis.role == "vertical"
        assert axis.role_inferred is True
        assert any("heuristic" in n for n in model.notes)

    def test_stencil_offsets_and_halo(self, model):
        """t(k+1) and t(k-1) → offsets {-1,0,1}, halo 1. tnew is read-only in place."""
        assert model.field("t").axes[0].offsets == frozenset({-1, 0, 1})
        assert model.field("t").axes[0].halo == 1
        assert model.field("tnew").axes[0].halo == 0
        assert model.max_halo == 1

    def test_ambiguous_dtype_flag(self):
        """A REAL with no KIND is flagged (the FORT007 concern)."""
        src = ("subroutine s(a, n)\n  integer :: n\n  real :: a(n)\n"
               "  a(1) = 1.0\nend subroutine s\n")
        m = build_domain_model(src)
        assert m.field("a").dtype_ambiguous is True


@pytest.mark.slow
class TestRenderers:
    @pytest.fixture(scope="class")
    def model(self):
        return build_domain_model(DIFFUSE)

    def test_gt4py_hints_declare_dimensions_types_and_offsets(self, model):
        hints = " ".join(to_gt4py_hints(model))
        assert "Dims[K]" in hints
        assert "float64" in hints
        assert "FieldOffset" in hints
        assert "Halo on K: 1" in hints
        assert "#82" in hints  # halo is the driver's concern

    def test_gt4py_flags_the_heuristic_roles(self, model):
        assert any("heuristic" in h for h in to_gt4py_hints(model))

    def test_jax_hints_give_shapes_and_dtypes(self, model):
        hints = " ".join(to_jax_hints(model))
        assert "shape (nlev)" in hints
        assert "float64" in hints

    def test_scalar_only_routine_renders_cleanly(self):
        src = ("subroutine s(a, b)\n  real(8), intent(in) :: a\n"
               "  real(8), intent(out) :: b\n  b = a * 2.0d0\nend subroutine s\n")
        m = build_domain_model(src)
        assert "Scalars only" in to_gt4py_hints(m)[0]
        assert "Scalars only" in to_jax_hints(m)[0]


class TestNodeWiring:
    def test_node_attaches_the_model_and_records_itself(self):
        from fortranspire.agent.domain_model import domain_model_agent

        state = {"kernel_results": [{"routine_name": "k", "fortran_code": DIFFUSE}],
                 "executed_agents": []}
        out = domain_model_agent(state)
        assert "domain_model" in out["kernel_results"][0]
        assert out["executed_agents"] == ["domain_model"]

    def test_both_graphs_run_domain_model_before_emission(self):
        pytest.importorskip("langgraph")
        from fortranspire.agent.translation_graph_gt4py import translation_app_gt4py
        from fortranspire.agent.translation_graph_phase2 import translation_app_phase2

        for app, emitter in [(translation_app_phase2, "jax_kernel"),
                             (translation_app_gt4py, "gt4py_kernel")]:
            names = [n for n in app.get_graph().nodes if not n.startswith("__")]
            assert names.index("domain_model") < names.index(emitter)
            assert names.index("functionalize") < names.index("domain_model")

    def test_emitters_read_the_typed_model(self):
        """Both emitters prefix their hints with the deterministic types."""
        from fortranspire.agent.domain_model import domain_model_agent
        from fortranspire.agent.nodes_gt4py.gt4py_kernel import _render_hints as g
        from fortranspire.agent.nodes_jax.jax_kernel import _render_hints as j

        k = domain_model_agent(
            {"kernel_results": [{"routine_name": "diffuse", "fortran_code": DIFFUSE}],
             "executed_agents": []}
        )["kernel_results"][0]
        assert "Dims[K]" in g(k)
        assert "shape (nlev)" in j(k)

    def test_missing_model_does_not_break_emitters(self):
        """A kernel with no domain_model still renders (loki-less fallback)."""
        from fortranspire.agent.nodes_gt4py.gt4py_kernel import _render_hints as g
        from fortranspire.agent.nodes_jax.jax_kernel import _render_hints as j

        k = {"routine_name": "k", "intent_map": {}, "dimensions": {}, "hints": [],
             "fortran_code": ""}
        assert isinstance(g(k), str) and isinstance(j(k), str)
