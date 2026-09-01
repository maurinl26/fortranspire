"""The monolithic-extractor prompt must be domain-agnostic (was seismic-locked).

The extractor only runs on a monolithic PROGRAM; if its prompt names one domain
(seismic CPML: stress/velocity/PML), it misdirects the model on every other
monolithic code. This guards against the lock-in returning.
"""
import pytest

from fortranspire.prompts.loader import load_prompt

_LOCKED_TERMS = ["seismic", "cpml", "pml", "stress", "velocity", "seismogram",
                 "2d spatial finite-difference", "finite-difference loop"]


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_extractor_prompt_is_not_domain_locked(lang):
    prompt = load_prompt("extractor", version="v2", lang=lang,
                         common_rules="", save_rules="", flag_rules="",
                         pointer_rules="").lower()
    hits = [t for t in _LOCKED_TERMS if t in prompt]
    assert not hits, f"extractor/{lang}/v2 still domain-locked on: {hits}"


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_extractor_prompt_keeps_its_placeholders(lang):
    # Rendering must still consume the four rule blocks (no stray braces).
    prompt = load_prompt("extractor", version="v2", lang=lang,
                         common_rules="C", save_rules="S", flag_rules="F",
                         pointer_rules="P")
    assert "{" not in prompt and "}" not in prompt
