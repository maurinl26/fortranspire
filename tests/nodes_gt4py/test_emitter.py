"""GT4Py emitter, validation, graph and CLI wiring (issue #42).

gt4py depends on a dace pre-release the locked resolver won't pull, so it
is not a project dependency and is absent from the dev/CI environment.
These tests therefore pin the *structure* — the graph, the reuse of the
functional analysis, the CLI routing, the prompt, and the skip-clean
behaviour when gt4py is absent. The actual type-check against gt4py's
frontend is verified separately in a venv that has gt4py installed.
"""
from __future__ import annotations

import inspect

import pytest


# ── Emission node ──────────────────────────────────────────────────────────

class TestEmissionNode:
    def test_blocked_routine_is_not_emitted(self):
        from fortranspire.agent.nodes_gt4py.gt4py_kernel import gt4py_kernel_agent

        # A blocked routine must be skipped before any LLM call.
        state = {
            "kernel_results": [{"routine_name": "k", "purity": "blocked",
                                "fortran_code": "x", "intent_map": {}}],
            "executed_agents": [],
        }
        out = gt4py_kernel_agent(state)
        assert out["kernel_results"][0]["status"] == "skipped"
        assert "gt4py_kernel" in out["executed_agents"]

    def test_hints_carry_dimensions_and_the_portability_verdict(self):
        from fortranspire.agent.nodes_gt4py.gt4py_kernel import _render_hints

        hints = _render_hints({
            "routine_name": "k",
            "intent_map": {"t": "IN", "out": "OUT"},
            "dimensions": {"t": ["nlev"], "out": ["nlev"]},
            "fortran_code": "out = t * 2.0",
            "hints": [],
        })
        assert "Field dimensions" in hints
        assert "Dims[K]" in hints or "1-D" in hints


# ── Validation node — skips cleanly without gt4py ──────────────────────────

class TestValidationSkip:
    def test_type_check_skips_when_gt4py_absent(self, monkeypatch):
        import fortranspire.agent.nodes_gt4py.domain_validate as dv

        monkeypatch.setattr(dv, "_gt4py_available", lambda: False)
        report = dv.type_check_source("import gt4py.next\n")
        assert report["status"] == "skipped"
        assert report["gt4py"] is False
        assert "not installed" in report["reason"]

    def test_node_reports_skip_not_failure_without_gt4py(self, monkeypatch):
        import fortranspire.agent.nodes_gt4py.domain_validate as dv

        monkeypatch.setattr(dv, "_gt4py_available", lambda: False)
        state = {
            "kernel_results": [{"routine_name": "k", "gt4py_code": "x=1",
                                "status": "pending"}],
            "executed_agents": [],
        }
        out = dv.domain_validate_agent(state)
        # A skip is neither a pass nor a failure — the caller must not read
        # "no failures" as validation.
        assert out["domain_check_skipped"] is True
        assert out["domain_validated"] is False

    def test_validation_uses_a_real_file_not_exec(self):
        """gt4py reads operator source via inspect.getsourcelines, so the
        node must write a file — never exec a string like the JAX path."""
        import fortranspire.agent.nodes_gt4py.domain_validate as dv

        src = inspect.getsource(dv._load_module_from_source)
        assert "write_text" in src
        assert "exec(" not in src  # not exec()-ing a code string


# ── Graph wiring ───────────────────────────────────────────────────────────

class TestGraph:
    @pytest.fixture(scope="class")
    def graph(self):
        pytest.importorskip("langgraph")
        from fortranspire.agent.translation_graph_gt4py import translation_app_gt4py

        return translation_app_gt4py.get_graph()

    def test_pipeline_order(self, graph):
        edges = {(e.source, e.target) for e in graph.edges}
        assert edges == {
            ("__start__", "init"),
            ("init", "parser"),
            ("parser", "extractor"),
            ("extractor", "functionalize"),
            ("functionalize", "gt4py_kernel"),
            ("gt4py_kernel", "domain_validate"),
            ("domain_validate", "__end__"),
        }

    def test_functionalize_is_the_phase2_node_reused(self):
        """The whole argument: gt4py.next is functional like JAX, so the
        purity analysis is shared, not forked."""
        from fortranspire.agent import translation_graph_gt4py as g
        from fortranspire.agent.nodes_jax import functionalize_agent

        assert g.functionalize_agent is functionalize_agent

    def test_parser_and_extractor_are_the_phase1_nodes(self):
        from fortranspire.agent import nodes, translation_graph_gt4py as g

        assert g.parser_phase1 is nodes.parser_phase1
        assert g.extractor_agent is nodes.extractor_agent

    def test_no_openacc_or_jax_emission_here(self, graph):
        names = " ".join(n for n in graph.nodes if not n.startswith("__"))
        for forbidden in ("openacc", "jax_kernel", "gradcheck", "cython"):
            assert forbidden not in names


# ── CLI wiring ─────────────────────────────────────────────────────────────

class TestCli:
    def test_verb_is_dispatched(self):
        from fortranspire.cli import _DISPATCH

        assert _DISPATCH["gt4py"] == ("fortranspire.agent.cli", "_gt4py_main")

    def test_gt4py_does_not_warm_loki_pointlessly(self):
        """gt4py DOES use loki (it parses), so it must NOT be in the skip set."""
        from fortranspire.cli import _NO_LOKI_VERBS

        assert "gt4py" not in _NO_LOKI_VERBS

    def test_gt4py_file_returns_an_exit_code(self):
        from fortranspire.agent.cli import gt4py_file

        assert "-> int" in str(inspect.signature(gt4py_file))
        src = inspect.getsource(gt4py_file)
        # Skip (no gt4py) returns 0; a failed type-check returns 1.
        assert "domain_check_skipped" in src
        assert "domain_validated" in src


# ── Prompt ─────────────────────────────────────────────────────────────────

class TestPrompt:
    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_prompt_renders(self, lang):
        from fortranspire.prompts.loader import load_prompt

        t = load_prompt("gt4py_kernel", version="v1", lang=lang,
                        signature="def k(a) -> b", hints="- none",
                        fortran_code="SUBROUTINE k")
        assert "def k(a) -> b" in t
        assert "{signature}" not in t and "{hints}" not in t

    def test_prompt_teaches_the_core_rules(self):
        from fortranspire.prompts.loader import load_prompt

        t = load_prompt("gt4py_kernel", version="v1", lang="en",
                        signature="s", hints="h", fortran_code="f").lower()
        assert "field_operator" in t
        assert "where" in t and "koff" in t
        assert "scan_operator" in t
        # The one rule that governs everything: drop the grid loop.
        assert "loop" in t and "point" in t

    def test_prompt_pins_the_confirmed_api(self):
        from fortranspire.prompts.loader import load_prompt

        t = load_prompt("gt4py_kernel", version="v1", lang="en",
                        signature="s", hints="h", fortran_code="f")
        assert "DimensionKind.VERTICAL" in t
        assert "gtx.FieldOffset" in t
