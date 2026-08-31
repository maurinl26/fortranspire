"""Wiring tests for the Phase 2 graph (issue #73).

Phase 2 is defined as Fortran → functional refactoring → JAX, **without
OpenACC**. These tests pin that definition against the graph, and pin the
reuse of the Phase 1 nodes that #73 requires rather than a fork.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from fortranspire.agent.translation_graph_phase2 import (  # noqa: E402
    translation_app_phase2,
)


@pytest.fixture(scope="module")
def graph():
    return translation_app_phase2.get_graph()



def _smoothing_kwargs() -> dict:
    """The smoothing placeholders every render of this prompt needs."""
    from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS
    from fortranspire.jax_smooth import catalogue_for_prompt

    return {
        "smoothing_mode": "none",
        "smoothing_instruction": _INSTRUCTIONS["none"],
        "smoothing_catalogue": catalogue_for_prompt(),
    }


def node_names(graph) -> list[str]:
    return [n for n in graph.nodes if not n.startswith("__")]


def test_pipeline_order(graph):
    edges = {(e.source, e.target) for e in graph.edges}
    expected = {
        ("__start__", "init"),
        ("init", "parser"),
        ("parser", "extractor"),
        ("extractor", "functionalize"),
        ("functionalize", "jax_kernel"),
        ("jax_kernel", "gradcheck"),
        ("gradcheck", "__end__"),
    }
    assert edges == expected


def test_functionalize_precedes_emission(graph):
    """The interface is derived before the body is generated.

    That ordering is the issue: emitting first means the model invents the
    signature, which is pure data-flow and must not be guessed.
    """
    names = node_names(graph)
    assert names.index("functionalize") < names.index("jax_kernel")


def test_gradcheck_is_last(graph):
    """Nothing runs after the gradient verdict — it is the gate."""
    names = node_names(graph)
    assert names[-1] == "gradcheck"


def test_no_openacc_node(graph):
    """Phase 2 targets JAX directly; a pragma node here would be a bug."""
    names = " ".join(node_names(graph))
    for forbidden in ("openacc", "openmp", "cython", "pragma"):
        assert forbidden not in names


def test_parser_and_extractor_are_the_phase1_nodes():
    """#73 requires reuse, not a fork — one hard code path, not two."""
    from fortranspire.agent import nodes, translation_graph_phase2 as p2

    assert p2.parser_phase1 is nodes.parser_phase1
    assert p2.extractor_agent is nodes.extractor_agent
    assert p2.init_phase1 is nodes.init_phase1


def test_deterministic_nodes_need_no_llm():
    """`functionalize` and `gradcheck` must not import the LLM stack.

    Their value is that they are computed, not generated. An accidental
    LLM call in either would make the interface and the gradient verdict
    non-reproducible.
    """
    import inspect

    from fortranspire.agent.nodes_jax import functionalize, gradcheck

    for module in (functionalize, gradcheck):
        source = inspect.getsource(module)
        assert "get_llm" not in source, f"{module.__name__} reaches for an LLM"
        assert "load_prompt" not in source, f"{module.__name__} loads a prompt"


def test_emission_uses_externalised_prompts():
    """Issue #3 established versioned prompt files; Phase 2 had none."""
    import inspect

    from fortranspire.agent.nodes_jax import jax_kernel

    source = inspect.getsource(jax_kernel)
    assert "load_prompt" in source
    assert 'load_prompt(\n            "jax_kernel"' in source or '"jax_kernel"' in source


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_jax_prompt_exists_and_renders(lang):
    from fortranspire.prompts.loader import load_prompt

    text = load_prompt(
        "jax_kernel", version="v1", lang=lang,
        signature="def k(x) -> y", hints="- none", fortran_code="SUBROUTINE k",
        **_smoothing_kwargs(),
    )
    assert "def k(x) -> y" in text
    assert "{signature}" not in text and "{hints}" not in text


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_prompt_carries_the_where_guard_rule(lang):
    """The NaN-gradient defect is the one gradcheck catches most often.

    The prompt must teach the fix, or every run pays for the same failure.
    """
    from fortranspire.prompts.loader import load_prompt

    text = load_prompt(
        "jax_kernel", version="v1", lang=lang,
        signature="s", hints="h", fortran_code="f",
        **_smoothing_kwargs(),
    ).lower()
    assert "where" in text
    assert "nan" in text
    assert "scan" in text


def test_translate_verb_routes_to_phase2():
    import inspect

    from fortranspire.agent.cli import translate_file

    source = inspect.getsource(translate_file)
    assert "translation_app_phase2" in source
    assert "gradcheck_passed" in source, "the gradient verdict must gate the exit code"
