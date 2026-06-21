"""LangChain ↔ Mistral OpenAI-compatible endpoint.

Two model roles are exposed via ``get_llm(stage)``:

- ``"reasoning"`` (default) — semantic refactoring stages
  (extractor, openacc data-region planning). Default model:
  ``mistral-large-latest``.
- ``"code"`` — boilerplate-heavy generation stages
  (Cython wrapper, headers). Default model: ``codestral-latest``.

The legacy ``MISTRAL_MODEL`` env var still works as a single override for
both roles, so existing ``.env`` files keep working unchanged.

No Azure dependency: the agent talks directly to any OpenAI-compatible
endpoint — La Plateforme Mistral, a self-hosted vLLM/TGI/Ollama server,
or a sovereign-EU gateway (Scaleway Generative APIs, OVH AI Endpoints).
"""
from __future__ import annotations

import os
from typing import Literal

from langchain_openai import ChatOpenAI

from fortranspire.config import config

# Install the content-addressed LLM response cache (#7) once per process.
# `install_global_cache` is idempotent and a no-op when
# `FORTRANSPIRE_CACHE=off`, so this is safe in every install profile.
from fortranspire.cache import install_global_cache as _install_cache
_install_cache()

Stage = Literal["reasoning", "code"]

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


def get_llm(stage: Stage = "reasoning") -> ChatOpenAI:
    """Return a LangChain ``ChatOpenAI`` client wired to a Mistral-compatible endpoint.

    Args:
        stage: ``"reasoning"`` for semantic stages (default), ``"code"`` for
            boilerplate code-generation stages. Picks the right model name
            automatically — see module docstring for the resolution rules.

    Why ``ChatOpenAI`` rather than ``ChatMistralAI``: the Mistral SDK hard-codes
    paths that work on ``api.mistral.ai`` but break against vLLM / TGI / Ollama.
    A single OpenAI-compatible client keeps every backend on the same code path.
    """
    api_key  = os.getenv("MISTRAL_API_KEY")
    endpoint = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1").rstrip("/")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY must be set (see .env). "
            "MISTRAL_ENDPOINT defaults to https://api.mistral.ai/v1."
        )

    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=_resolve_model(stage),
        temperature=config.temperature,
    )


# Backward-compatibility aliases — old call sites that used these names
# get the right model role automatically.
def get_reasoning_llm() -> ChatOpenAI:
    return get_llm("reasoning")


def get_translator_llm() -> ChatOpenAI:
    """Alias kept for legacy callers; routes to the code-gen model."""
    return get_llm("code")
