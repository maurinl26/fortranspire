"""Tests for the LangChain callback handler that pipes token usage into a Span."""
from __future__ import annotations

import pytest

# Skip the whole file if langchain isn't installed (analyze-only image).
pytest.importorskip("langchain_core")

from fortranspire.observability.llm_callback import token_callback
from fortranspire.observability.tracer import Span


def _make_fake_llm_response(prompt_tokens: int, completion_tokens: int):
    """Build the minimal LangChain LLMResult shape token_callback reads."""
    from langchain_core.outputs import LLMResult
    return LLMResult(
        generations=[[]],
        llm_output={"token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    )


def test_callback_records_token_counts():
    span = Span(node="doc_routine")
    cb = token_callback(span)
    cb.on_llm_end(_make_fake_llm_response(100, 50))
    assert span.prompt_tokens == 100
    assert span.completion_tokens == 50


def test_callback_accumulates_across_calls():
    span = Span(node="extractor")
    cb = token_callback(span)
    cb.on_llm_end(_make_fake_llm_response(10, 20))
    cb.on_llm_end(_make_fake_llm_response(5, 7))
    assert span.prompt_tokens == 15
    assert span.completion_tokens == 27


def test_callback_handles_missing_llm_output():
    span = Span(node="doc_routine")
    cb = token_callback(span)
    # Response without llm_output (some backends) — must not crash.
    from langchain_core.outputs import LLMResult
    cb.on_llm_end(LLMResult(generations=[[]], llm_output=None))
    assert span.prompt_tokens == 0
    assert span.completion_tokens == 0


def test_callback_records_error():
    span = Span(node="openacc_kernel")
    cb = token_callback(span)
    cb.on_llm_error(ValueError("rate limited"))
    assert span.error and "rate limited" in span.error
