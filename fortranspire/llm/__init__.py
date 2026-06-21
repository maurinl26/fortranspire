"""LangChain ↔ Mistral / OpenAI-compatible endpoint dispatch — issue #9.

Two model roles via ``get_llm(stage)``:

- ``"reasoning"`` — semantic refactoring stages (extractor, openacc).
  Default: ``mistral-large-latest``.
- ``"code"`` — boilerplate-heavy generation stages (cython, doc_routine).
  Default: ``codestral-latest``.

Two backends, selected automatically (override via ``FORTRANSPIRE_LLM_BACKEND``):

- **``mistral``** (preferred when the endpoint matches Mistral La Plateforme) —
  uses ``langchain_mistralai.ChatMistralAI`` which wraps the official
  ``mistralai`` Python SDK. Enables Mistral-native features: function
  calling, JSON-mode, streaming, Mistral-specific safety guardrails.
- **``openai``** — uses ``langchain_openai.ChatOpenAI`` against an
  OpenAI-compatible endpoint. Works against vLLM / TGI / Ollama / Scaleway
  Generative APIs / OVH AI Endpoints / any other OpenAI-shaped server.

The legacy ``MISTRAL_MODEL`` env var still works as a single override for
both roles, so existing ``.env`` files keep working unchanged.

No Azure dependency: the agent talks directly to the endpoint of your
choice — La Plateforme, a self-hosted vLLM/TGI/Ollama server, or a
sovereign-EU gateway.
"""
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from fortranspire.config import config

# Install the content-addressed LLM response cache (#7) once per process.
# `install_global_cache` is idempotent and a no-op when
# `FORTRANSPIRE_CACHE=off`, so this is safe in every install profile.
from fortranspire.cache import install_global_cache as _install_cache
_install_cache()

Stage   = Literal["reasoning", "code"]
Backend = Literal["mistral", "openai", "auto"]

_DEFAULTS: dict[Stage, str] = {
    "reasoning": "mistral-large-latest",
    "code":      "codestral-latest",
}

_ENV_PER_STAGE: dict[Stage, str] = {
    "reasoning": "MISTRAL_MODEL_REASONING",
    "code":      "MISTRAL_MODEL_CODE",
}


def _resolve_model(stage: Stage) -> str:
    """Pick the model name for a given pipeline stage.

    Resolution order (first hit wins):
      1. ``MISTRAL_MODEL_REASONING`` / ``MISTRAL_MODEL_CODE`` — per-stage override
      2. ``MISTRAL_MODEL`` — single legacy override
      3. role-appropriate default (``mistral-large-latest`` / ``codestral-latest``)
    """
    per_stage = os.getenv(_ENV_PER_STAGE[stage])
    if per_stage:
        return per_stage
    legacy = os.getenv("MISTRAL_MODEL")
    if legacy:
        return legacy
    return _DEFAULTS[stage]


def _resolve_backend(endpoint: str) -> Backend:
    """Pick the backend implementation. Env var override > endpoint sniff.

    Auto-detection: an endpoint that matches Mistral La Plateforme
    (``api.mistral.ai``) uses the native ``mistralai`` SDK via
    ``langchain-mistralai`` — gets first-class Mistral features. Any
    other endpoint defaults to the generic OpenAI-compatible client so
    self-hosted vLLM / TGI / Ollama / SoCloud / etc. keep working.

    Manual override:
      FORTRANSPIRE_LLM_BACKEND=mistral   force native SDK
      FORTRANSPIRE_LLM_BACKEND=openai    force OpenAI-compatible
    """
    override = os.getenv("FORTRANSPIRE_LLM_BACKEND", "auto").lower()
    if override in ("mistral", "openai"):
        return override  # type: ignore[return-value]
    if "api.mistral.ai" in endpoint:
        return "mistral"
    return "openai"


def get_llm(stage: Stage = "reasoning") -> Any:
    """Return a LangChain chat client wired to the resolved backend.

    Returns either ``ChatMistralAI`` (native SDK, when the endpoint is
    Mistral La Plateforme) or ``ChatOpenAI`` (OpenAI-compatible
    fallback). Both expose the same LangChain ``BaseChatModel``
    interface, so callers (extractor / openacc / cython / document)
    don't need to branch.
    """
    api_key  = os.getenv("MISTRAL_API_KEY")
    endpoint = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1").rstrip("/")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY must be set (see .env). "
            "MISTRAL_ENDPOINT defaults to https://api.mistral.ai/v1."
        )

    model = _resolve_model(stage)
    backend = _resolve_backend(endpoint)

    if backend == "mistral":
        # Lazy-import so the OpenAI-only path doesn't require langchain-mistralai
        # to be present (some self-hosted images strip it).
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            # Graceful fallback: if langchain-mistralai isn't installed,
            # use ChatOpenAI even when the endpoint is Mistral. The user
            # loses native SDK perks but the call still works.
            return ChatOpenAI(
                base_url=endpoint,
                api_key=api_key,
                model=model,
                temperature=config.temperature,
            )
        return ChatMistralAI(
            mistral_api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=config.temperature,
        )

    # backend == "openai"
    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=model,
        temperature=config.temperature,
    )


# Backward-compatibility aliases — old call sites that used these names
# get the right model role automatically.
def get_reasoning_llm() -> Any:
    return get_llm("reasoning")


def get_translator_llm() -> Any:
    """Alias kept for legacy callers; routes to the code-gen model."""
    return get_llm("code")
