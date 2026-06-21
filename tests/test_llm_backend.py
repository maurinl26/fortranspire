"""Tests for the LLM backend dispatch — issue #9.

These tests don't fire any actual LLM call. They verify the backend
selection logic, env-var overrides, and graceful fallback when
``langchain-mistralai`` isn't installed.
"""
from __future__ import annotations

import sys
import types

import pytest

# Skip the whole module on analyze-only installs (no langchain).
pytest.importorskip("langchain_openai")

from fortranspire.llm import _resolve_backend, _resolve_model, get_llm


# ── _resolve_backend ────────────────────────────────────────────────────────

def test_resolve_backend_auto_picks_mistral_for_la_plateforme(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORTRANSPIRE_LLM_BACKEND", raising=False)
    assert _resolve_backend("https://api.mistral.ai/v1") == "mistral"


def test_resolve_backend_auto_picks_openai_for_self_hosted(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORTRANSPIRE_LLM_BACKEND", raising=False)
    assert _resolve_backend("http://my-vllm-host:8000/v1") == "openai"
    assert _resolve_backend("https://generative-api.scaleway.com/v1") == "openai"


@pytest.mark.parametrize("override,expected", [
    ("mistral", "mistral"),
    ("MISTRAL", "mistral"),
    ("openai",  "openai"),
    ("OPENAI",  "openai"),
])
def test_resolve_backend_env_override(
    override: str, expected: str, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORTRANSPIRE_LLM_BACKEND", override)
    # The override beats endpoint auto-detection.
    assert _resolve_backend("https://api.mistral.ai/v1") == expected
    assert _resolve_backend("http://vllm:8000/v1") == expected


def test_resolve_backend_unknown_override_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORTRANSPIRE_LLM_BACKEND", "garbage")
    # Auto applies → Mistral endpoint → mistral
    assert _resolve_backend("https://api.mistral.ai/v1") == "mistral"


# ── _resolve_model (covered elsewhere; quick check for regression) ──────────

def test_resolve_model_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    monkeypatch.delenv("MISTRAL_MODEL_REASONING", raising=False)
    monkeypatch.delenv("MISTRAL_MODEL_CODE", raising=False)
    assert _resolve_model("reasoning") == "mistral-large-latest"
    assert _resolve_model("code") == "codestral-latest"


# ── get_llm dispatch ────────────────────────────────────────────────────────

def test_get_llm_returns_chatopenai_for_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORTRANSPIRE_LLM_BACKEND", "openai")
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setenv("MISTRAL_ENDPOINT", "http://my-vllm:8000/v1")
    llm = get_llm("reasoning")
    # ChatOpenAI from langchain_openai
    assert type(llm).__name__ == "ChatOpenAI"


def test_get_llm_returns_chatmistralai_for_mistral_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORTRANSPIRE_LLM_BACKEND", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
    try:
        from langchain_mistralai import ChatMistralAI  # noqa: F401
    except ImportError:
        pytest.skip("langchain-mistralai not installed")
    llm = get_llm("reasoning")
    assert type(llm).__name__ == "ChatMistralAI"


def test_get_llm_raises_without_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        get_llm("reasoning")


def test_get_llm_falls_back_to_openai_when_mistralai_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """If we force backend=mistral but langchain_mistralai is unavailable,
    the call must still succeed via the OpenAI-compatible client."""
    monkeypatch.setenv("FORTRANSPIRE_LLM_BACKEND", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
    # Simulate langchain_mistralai not being importable by stubbing sys.modules.
    saved = sys.modules.get("langchain_mistralai")
    sys.modules["langchain_mistralai"] = None  # type: ignore[assignment]
    try:
        llm = get_llm("reasoning")
        assert type(llm).__name__ == "ChatOpenAI"
    finally:
        if saved is not None:
            sys.modules["langchain_mistralai"] = saved
        else:
            sys.modules.pop("langchain_mistralai", None)
