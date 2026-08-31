"""Tests for the functional-interface derivation (issue #73).

This node decides two things no LLM should decide: the signature the
emitted kernel must have, and whether the routine can become a pure
function at all. Both are pure data-flow, and both are load-bearing —
a wrong signature means the LLM invents one, which is the failure this
node exists to prevent.
"""
from __future__ import annotations

import pytest

from fortranspire.agent.nodes_jax.functionalize import (
    _render_signature,
    _split_by_intent,
    _verdict,
    functionalize_agent,
)


def kernel(**overrides) -> dict:
    base = {
        "routine_name": "k",
        "fortran_code": "SUBROUTINE k\nEND SUBROUTINE",
        "intent_map": {"x": "IN", "y": "OUT"},
        "dimensions": {},
        "has_io": False,
        "has_save": False,
        "loops": [],
    }
    base.update(overrides)
    return base


class TestSignature:
    """A subroutine mutates; a JAX function returns. The map must be exact."""

    def test_in_becomes_argument_out_becomes_return(self):
        inputs, outputs, carried = _split_by_intent({"a": "IN", "b": "OUT"})
        assert inputs == ["a"]
        assert outputs == ["b"]
        assert carried == []

    def test_inout_appears_on_both_sides(self):
        """INTENT(INOUT) is read *and* written — it is an argument and a result."""
        inputs, outputs, carried = _split_by_intent({"v": "INOUT"})
        assert inputs == ["v"]
        assert outputs == ["v"]
        assert carried == ["v"]

    def test_unknown_intent_is_treated_pessimistically(self):
        """Fortran's default is effectively INOUT.

        Assuming read-only would silently drop a mutation from the returned
        tuple, which is the worst possible way to be wrong here.
        """
        inputs, outputs, carried = _split_by_intent({"z": ""})
        assert inputs == ["z"] and outputs == ["z"] and carried == ["z"]

    def test_case_and_whitespace_are_normalised(self):
        inputs, outputs, _ = _split_by_intent({"a": " in ", "b": "Out"})
        assert inputs == ["a"] and outputs == ["b"]

    @pytest.mark.parametrize(
        "intents,expected",
        [
            ({"a": "IN", "b": "OUT"}, "def k(a) -> b"),
            ({"a": "IN", "b": "OUT", "c": "OUT"}, "def k(a) -> tuple[b, c]"),
            ({"v": "INOUT"}, "def k(v) -> v"),
            ({"a": "IN"}, "def k(a) -> None"),
        ],
    )
    def test_rendered_signature(self, intents, expected):
        inputs, outputs, _ = _split_by_intent(intents)
        assert _render_signature("k", inputs, outputs) == expected


class TestPurityVerdict:
    def test_io_blocks(self):
        purity, reason = _verdict(kernel(has_io=True), ["y"])
        assert purity == "blocked"
        assert "I/O" in reason

    def test_no_output_blocks(self):
        """No OUT/INOUT means the effect is invisible — it writes globals."""
        purity, reason = _verdict(kernel(), [])
        assert purity == "blocked"
        assert "global state" in reason

    def test_save_requires_threading_but_is_not_blocked(self):
        purity, reason = _verdict(kernel(has_save=True), ["y"])
        assert purity == "threaded"
        assert "carry" in reason

    def test_clean_routine_is_pure(self):
        purity, _ = _verdict(kernel(), ["y"])
        assert purity == "pure"

    def test_io_takes_precedence_over_save(self):
        """Both wrong: report the one that cannot be worked around."""
        purity, reason = _verdict(kernel(has_io=True, has_save=True), ["y"])
        assert purity == "blocked" and "I/O" in reason


class TestScanDetection:
    def test_loop_carried_dependency_requires_scan(self):
        """A recurrence vectorised as an array expression is silently wrong."""
        out = functionalize_agent(
            {"kernel_results": [kernel(has_loop_carried_dep=True, loops=["1:n"])],
             "executed_agents": []}
        )
        entry = out["kernel_results"][0]
        assert entry["needs_scan"] is True
        assert any("scan" in h for h in entry["hints"])

    def test_independent_loops_vectorise(self):
        out = functionalize_agent(
            {"kernel_results": [kernel(loops=["1:n"])], "executed_agents": []}
        )
        entry = out["kernel_results"][0]
        assert entry["needs_scan"] is False
        assert any("vectorise" in h for h in entry["hints"])


class TestNodeContract:
    def test_blocked_kernels_are_marked_skipped_not_dropped(self):
        """A refusal must be reported, never silently omitted."""
        out = functionalize_agent(
            {"kernel_results": [kernel(has_io=True)], "executed_agents": []}
        )
        assert len(out["kernel_results"]) == 1
        assert out["kernel_results"][0]["status"] == "skipped"

    def test_inout_produces_a_mutation_hint(self):
        out = functionalize_agent(
            {"kernel_results": [kernel(intent_map={"v": "INOUT"})], "executed_agents": []}
        )
        hints = " ".join(out["kernel_results"][0]["hints"])
        assert ".at[" in hints and "never mutated" in hints

    def test_node_records_itself(self):
        out = functionalize_agent({"kernel_results": [], "executed_agents": []})
        assert out["executed_agents"] == ["functionalize"]
        assert out["functionalized"] is True

    def test_parser_fields_survive(self):
        """The node augments the kernel dict; it must not truncate it."""
        out = functionalize_agent(
            {"kernel_results": [kernel(dimensions={"x": ["8"]})], "executed_agents": []}
        )
        assert out["kernel_results"][0]["dimensions"] == {"x": ["8"]}
        assert out["kernel_results"][0]["fortran_code"].startswith("SUBROUTINE")


class TestAnalyzeIntegration:
    """FORT030 — the JAX verdict surfaced before anyone is quoted (#73)."""

    def test_rule_is_registered(self):
        from fortranspire.agent.analyze import RULES

        assert "FORT030" in RULES
        assert RULES["FORT030"]["severity"] == "note"

    def test_analyze_reuses_the_node_rather_than_restating_them(self):
        """One definition of "pure", or `explain` quotes a port that is refused."""
        from fortranspire.agent.analyze import _jax_verdict

        assert _jax_verdict(kernel(has_io=True))[0] == "blocked"
        assert _jax_verdict(kernel(has_save=True))[0] == "threaded"
        assert _jax_verdict(kernel())[0] == "pure"

    def test_pure_routines_produce_no_finding(self):
        """A clean routine needs no annotation.

        Emitting one per routine would bury real findings under a wall of
        "this is fine" in Code Scanning, which the composite action uploads
        on every pull request.
        """
        from fortranspire.agent.analyze import RULES, _jax_verdict

        verdict, _ = _jax_verdict(kernel())
        assert verdict == "pure"
        assert "FORT030" in RULES  # registered, but only emitted when non-pure

    def test_explain_labels_every_analyze_rule(self):
        """`explain` keeps its label map in sync with `analyze` by hand.

        Its own comment says so. A rule added on one side and forgotten on
        the other silently loses its label in the port-cost report — the
        document a client reads before paying.
        """
        from fortranspire.agent.analyze import RULES
        from fortranspire.agent.explain import _RISK_LABELS

        missing = sorted(set(RULES) - set(_RISK_LABELS))
        assert not missing, f"rules with no label in explain: {missing}"
