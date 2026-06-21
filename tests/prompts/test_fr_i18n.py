"""Tests for the French prompt variants — issue #16.

Covers the 6 prompt families × 2 versions = 12 FR files added in this PR.
Verifies the loader picks them up, that substitution still works in FR,
and that the FR copy is genuinely different content (not an EN fallback).
"""
from __future__ import annotations

import pytest

from fortranspire.prompts.loader import clear_cache, load_prompt


PROMPTS_WITH_FR = [
    # (name, version, EN-only signature word, FR-only signature word)
    ("extractor",      "v1", "Fortran HPC expert", "expert HPC Fortran"),
    ("extractor",      "v2", "Fortran HPC expert", "expert HPC Fortran"),
    ("openacc_kernel", "v1", "OpenACC GPU expert", "expert GPU OpenACC"),
    ("openacc_kernel", "v2", "OpenACC GPU expert", "expert GPU OpenACC"),
    ("openacc_driver", "v1", "OpenACC GPU expert", "expert GPU OpenACC"),
    ("openacc_driver", "v2", "OpenACC GPU expert", "expert GPU OpenACC"),
    ("cython_pyx",     "v1", "Cython expert",      "expert Cython"),
    ("cython_pyx",     "v2", "Cython expert",      "expert Cython"),
    ("cython_header",  "v1", "C/Fortran interop",  "interopérabilité C/Fortran"),
    ("cython_header",  "v2", "C/Fortran interop",  "interopérabilité C/Fortran"),
    ("doc_routine",    "v1", "non-developer",      "non-développeurs"),
    # doc_routine v2 intentionally omitted — only v1 is used by the documenter
    # (no migration was needed when #4 landed structured outputs there).
]


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.mark.parametrize("name,version,en_word,fr_word", PROMPTS_WITH_FR)
def test_fr_variant_exists_and_differs_from_en(name, version, en_word, fr_word):
    en = load_prompt(name, version=version, lang="en")
    fr = load_prompt(name, version=version, lang="fr")
    # Sanity: variants are non-empty
    assert en.strip() and fr.strip()
    # FR variant must actually be French (not silently falling back to EN)
    assert fr_word in fr, f"{name}/{version}: expected FR signature {fr_word!r} in FR variant"
    # And EN must remain EN
    assert en_word in en, f"{name}/{version}: expected EN signature {en_word!r} in EN variant"
    # The two must not be identical strings (would mean FR fell back to EN)
    assert en != fr, f"{name}/{version}: FR and EN are identical — fallback bug"


def test_extractor_fr_v1_accepts_rule_substitution():
    text = load_prompt(
        "extractor", version="v1", lang="fr",
        common_rules="--COMMON-FR--",
        save_rules="--SAVE-FR--",
        flag_rules="--FLAG-FR--",
        pointer_rules="--POINTER-FR--",
    )
    for marker in ("--COMMON-FR--", "--SAVE-FR--", "--FLAG-FR--", "--POINTER-FR--"):
        assert marker in text


def test_extractor_fr_v2_accepts_rule_substitution():
    text = load_prompt(
        "extractor", version="v2", lang="fr",
        common_rules="--COMMON-FR--",
        save_rules="--SAVE-FR--",
        flag_rules="--FLAG-FR--",
        pointer_rules="--POINTER-FR--",
    )
    for marker in ("--COMMON-FR--", "--SAVE-FR--", "--FLAG-FR--", "--POINTER-FR--"):
        assert marker in text


def test_default_lang_env_picks_up_fr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_LANG", "fr")
    clear_cache()
    text = load_prompt("openacc_kernel", version="v1")  # no explicit lang
    assert "expert GPU OpenACC" in text
