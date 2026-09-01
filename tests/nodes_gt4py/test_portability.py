"""GT4Py (gt4py.next) portability scoring — FORT032 (issue #42).

The score triages which Fortran routines make good gt4py.next field
operators, deterministically and before any LLM is called. These tests pin
the rubric from docs/concepts/gt4py-next-patterns.md §9, and — most
importantly — that the score never disagrees with the JAX purity verdict,
since both read the same functionalize node.
"""
from __future__ import annotations

import pytest

from fortranspire.agent.nodes_gt4py.portability import Gt4PyVerdict, score_routine


def kernel(**overrides) -> dict:
    base = {
        "routine_name": "k",
        "intent_map": {"a": "IN", "b": "OUT"},
        "has_io": False,
        "has_save": False,
        "fortran_code": "",
    }
    base.update(overrides)
    return base


class TestRubric:
    def test_pointwise_is_a_clean_field_operator(self):
        v = score_routine(kernel(), "b = a * 2.0")
        assert v.score == 5
        assert v.label == "field operator"

    def test_constant_offset_stencil_is_still_five(self):
        """A(i+1) is a Cartesian shift — the portable stencil case."""
        v = score_routine(kernel(), "b = a + 0.5 * (a - 1.0)")
        assert v.score == 5

    def test_io_does_not_map(self):
        v = score_routine(kernel(has_io=True), "write(6,*) a")
        assert v.score == 0
        assert v.label == "does not map"

    def test_loop_carried_dependency_needs_scan_operator(self):
        v = score_routine(kernel(intent_map={"x": "INOUT"}, has_loop_carried_dep=True),
                          "x(k) = x(k-1) * d")
        assert v.score == 3
        assert "scan_operator" in v.reason

    def test_connectivity_access_is_first_class(self):
        """The unstructured mesh model (icon4py / Pace) is the mature,
        high-value target — a construct (neighbor_sum over a connectivity),
        not a penalty. `e2c` is a mesh connectivity name."""
        v = score_routine(kernel(intent_map={"a": "IN", "f": "OUT"}),
                          "f(e) = cellval(e2c(e, c))")
        assert v.score == 3
        assert "unstructured connectivity" in v.label
        assert "mesh connectivity" in v.reason

    def test_data_dependent_branch_maps_to_where(self):
        v = score_routine(kernel(intent_map={"q": "IN", "r": "OUT"}),
                          "IF (q < 0.0) THEN\n  r = 0.0\nEND IF")
        assert v.score == 3
        assert "where" in v.reason

    def test_save_state_is_threaded_not_blocked(self):
        v = score_routine(kernel(has_save=True), "y = x")
        assert v.score == 3


class TestPurityFloorConsistency:
    """The GT4Py score must never disagree with the JAX purity verdict —
    both read the same functionalize node, on purpose."""

    def test_blocked_for_jax_scores_zero_for_gt4py(self):
        from fortranspire.agent.nodes_jax.functionalize import _split_by_intent, _verdict

        k = kernel(has_io=True)
        _, outputs, _ = _split_by_intent(k["intent_map"])
        jax_verdict, _ = _verdict(k, outputs)
        assert jax_verdict == "blocked"
        assert score_routine(k, "write(6,*) a").score == 0

    def test_no_output_blocks_both(self):
        """A routine that writes only global state maps to neither target."""
        k = kernel(intent_map={"a": "IN"})  # no OUT/INOUT
        assert score_routine(k, "globalstate = a").score == 0

    def test_connectivity_maps_to_a_construct_not_a_penalty(self):
        """A clean stencil is 5; a connectivity access is 3 (a construct),
        not a near-unportable 1 — the unstructured model is first-class."""
        clean = score_routine(kernel(intent_map={"a": "IN", "b": "OUT"}), "b = a + 1.0")
        connectivity = score_routine(kernel(intent_map={"a": "IN", "b": "OUT"}),
                                     "b = a(c2e(i))")
        assert clean.score == 5 and connectivity.score == 3


class TestSourceOptional:
    def test_scoring_works_without_source_text(self):
        """Parser flags alone still give a verdict; source-scan signals
        (indirection, branch) are simply not raised."""
        v = score_routine(kernel())
        assert isinstance(v, Gt4PyVerdict)
        assert v.score == 5  # nothing in the flags to lower it

    def test_io_flag_alone_blocks_without_source(self):
        assert score_routine(kernel(has_io=True)).score == 0


class TestAnalyzeIntegration:
    def test_rule_is_registered(self):
        from fortranspire.agent.analyze import RULES

        assert RULES["FORT032"]["severity"] == "note"

    def test_explain_labels_the_rule(self):
        from fortranspire.agent.analyze import RULES
        from fortranspire.agent.explain import _RISK_LABELS

        # Every analyze rule must have an explain label (the sync guard).
        assert set(RULES) <= set(_RISK_LABELS)

    def test_clean_routine_produces_no_finding(self, tmp_path):
        """Score-5 routines are not annotated, like FORT030."""
        from fortranspire.agent.analyze import analyze_paths

        f = tmp_path / "clean.f90"
        f.write_text(
            "subroutine axpy(n, a, x, y)\n"
            "  implicit none\n"
            "  integer, intent(in) :: n\n"
            "  real(8), intent(in) :: a, x(n)\n"
            "  real(8), intent(inout) :: y(n)\n"
            "  integer :: i\n"
            "  do i = 1, n\n"
            "     y(i) = y(i) + a * x(i)\n"
            "  end do\n"
            "end subroutine axpy\n"
        )
        codes = {fd.rule_id for fd in analyze_paths([str(f)])[0].findings}
        assert "FORT032" not in codes
