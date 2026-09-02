"""Reproducibility defaults — a code translation must be deterministic (#agentic-1).

A non-zero temperature made two runs of the same port emit different code. The
default must be 0.0, and the LLM client must actually receive it.
"""


def test_default_temperature_is_zero():
    from fortranspire.config import AgentConfig
    assert AgentConfig().temperature == 0.0


def test_llm_client_receives_zero_temperature(monkeypatch):
    # get_llm must pass config.temperature (0.0) to the client, not a sampled value.
    import os
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("MISTRAL_ENDPOINT", "https://example.invalid/v1")  # openai backend
    from fortranspire.llm import get_llm
    llm = get_llm("code")
    # ChatOpenAI stores temperature on the instance.
    assert getattr(llm, "temperature", None) == 0.0
