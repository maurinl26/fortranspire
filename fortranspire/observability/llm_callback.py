"""LangChain callback handler that pipes token usage into a :class:`Span`.

Lazy-imports ``langchain_core`` so this module can be loaded by the
analyze-only image (no LLM stack installed) — only callers that
actually instantiate the handler need ``[gpu]``.

Usage::

    from fortranspire.observability import tracer
    from fortranspire.observability.llm_callback import token_callback

    with tracer.span(node="doc_routine", model="codestral-latest") as span:
        result = llm.with_structured_output(Schema).invoke(
            messages, config={"callbacks": [token_callback(span)]}
        )

The callback handles both regular ``invoke()`` and
``with_structured_output().invoke()`` — token usage comes from
``response.llm_output["token_usage"]`` in both cases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fortranspire.observability.tracer import Span


def token_callback(span: "Span"):
    """Return a fresh LangChain callback handler bound to ``span``."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _TokenCapture(BaseCallbackHandler):
        """Capture prompt/completion token counts from LangChain's response."""

        def on_llm_end(self, response: Any, **_: Any) -> None:
            usage = {}
            try:
                if response.llm_output and isinstance(response.llm_output, dict):
                    usage = response.llm_output.get("token_usage") or {}
            except AttributeError:
                usage = {}
            span.record_tokens(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )

        def on_llm_error(self, error: BaseException, **_: Any) -> None:
            span.record_error(error)

    return _TokenCapture()
