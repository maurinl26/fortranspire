"""Tests for the versioned, localized prompt loader."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fortranspire.prompts.loader import clear_cache, load_prompt


def test_load_known_prompt_en_v1():
    text = load_prompt("doc_routine", version="v1", lang="en")
    assert "non-developer stakeholders" in text
    assert text.strip().endswith("no preamble.")


def test_load_fr_variant():
    text = load_prompt("doc_routine", version="v1", lang="fr")
    # The FR variant exists — must read the French file, not fall back to EN.
    assert "parties prenantes" in text
    assert "Réponds toujours" in text


def test_lang_falls_back_to_english_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FORTRANSPIRE_LANG", raising=False)
    clear_cache()
    # Request a locale that doesn't exist (no `es/` directory shipped) — the
    # loader must transparently fall back to the EN copy rather than crash.
    text = load_prompt("openacc_kernel", version="v1", lang="es")
    assert "OpenACC GPU expert" in text


def test_template_substitution_keys():
    text = load_prompt(
        "extractor", version="v1",
        common_rules="--COMMON-RULES--",
        save_rules="--SAVE-RULES--",
        flag_rules="--FLAG-RULES--",
        pointer_rules="--POINTER-RULES--",
    )
    for marker in ("--COMMON-RULES--", "--SAVE-RULES--",
                   "--FLAG-RULES--", "--POINTER-RULES--"):
        assert marker in text


def test_partial_substitution_raises_keyed_error():
    # When the caller passes *some* kwargs but not all, the formatter runs and
    # surfaces a clear error naming the prompt and the missing variable.
    with pytest.raises(KeyError) as excinfo:
        load_prompt("extractor", version="v1", common_rules="X")  # missing 3 others
    msg = str(excinfo.value)
    assert "extractor" in msg


def test_zero_kwargs_returns_raw_template():
    # Short-circuit: calling without any variables returns the template as-is.
    # This is intentional — some prompts contain literal `{` braces (JSON
    # schema examples, regex patterns) and we don't want the formatter to
    # choke on them when the caller didn't ask for substitution.
    text = load_prompt("extractor", version="v1")
    assert "{common_rules}" in text


def test_missing_prompt_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("does_not_exist", version="v1", lang="en")
    assert "does_not_exist" in str(excinfo.value)
    assert "FORTRANSPIRE_PROMPTS_DIR" in str(excinfo.value)


def test_override_dir_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    override_root = tmp_path / "myprompts"
    target = override_root / "doc_routine" / "en" / "v1.md"
    target.parent.mkdir(parents=True)
    target.write_text("OVERRIDDEN", encoding="utf-8")

    monkeypatch.setenv("FORTRANSPIRE_PROMPTS_DIR", str(override_root))
    clear_cache()

    assert load_prompt("doc_routine", version="v1", lang="en") == "OVERRIDDEN"


def test_lang_env_var_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_LANG", "fr")
    clear_cache()
    text = load_prompt("doc_routine", version="v1")   # no explicit lang
    assert "parties prenantes" in text
