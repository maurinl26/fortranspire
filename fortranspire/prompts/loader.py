"""Versioned, localized prompt loader.

Prompts live as plain `.md` files at
``fortranspire/prompts/<name>/<lang>/<version>.md`` so they can be:

- reviewed and audited without diffing Python source,
- A/B'd by pinning a different ``version`` per call site,
- translated (FR for Météo-France / CEA / ENM Toulouse, EN default),
- overridden per-deployment via the ``FORTRANSPIRE_PROMPTS_DIR`` env var.

Templates use Python's ``str.format`` syntax — no Jinja dependency. Pass
keyword arguments to fill placeholders:

    >>> load_prompt("openacc_kernel", version="v1", lang="en")
    '<rendered text>'
    >>> load_prompt("extractor", version="v1", common_rules="…")
    '<rendered text with conditional sections>'

Missing keys raise ``KeyError`` with the prompt name in the message so
the bug is easy to spot.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from string import Formatter

_BUILTIN_ROOT = Path(__file__).resolve().parent
_USER_ROOT_ENV = "FORTRANSPIRE_PROMPTS_DIR"


def _override_root() -> Path | None:
    raw = os.getenv(_USER_ROOT_ENV)
    return Path(raw).resolve() if raw else None


@lru_cache(maxsize=128)
def _read(name: str, version: str, lang: str) -> str:
    """Resolve `<root>/<name>/<lang>/<version>.md` with EN fallback."""
    rel = Path(name) / lang / f"{version}.md"
    fallback = Path(name) / "en" / f"{version}.md"

    for root in (_override_root(), _BUILTIN_ROOT):
        if root is None:
            continue
        for candidate in (rel, fallback):
            path = root / candidate
            if path.is_file():
                return path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Prompt not found: name={name!r} version={version!r} lang={lang!r}. "
        f"Looked under: {_override_root() or '(no override)'} and {_BUILTIN_ROOT}. "
        f"Set {_USER_ROOT_ENV} to point at a custom prompts directory."
    )


class _SafeFormatter(Formatter):
    """`str.format` variant that errors clearly on missing keys."""

    def __init__(self, prompt_name: str) -> None:
        self._prompt_name = prompt_name
        super().__init__()

    def get_value(self, key, args, kwargs):  # type: ignore[override]
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, IndexError) as exc:
            raise KeyError(
                f"Prompt {self._prompt_name!r} requires variable {key!r}; "
                f"pass it as a keyword argument to load_prompt()."
            ) from exc


def load_prompt(
    name: str,
    *,
    version: str = "v1",
    lang: str | None = None,
    **variables: str,
) -> str:
    """Return the rendered text of a prompt.

    Args:
        name: prompt directory under ``fortranspire/prompts/`` — e.g.
            ``"extractor"``, ``"openacc_kernel"``, ``"doc_routine"``.
        version: pin the prompt revision (default: ``"v1"``). Bump when
            you change the wording so the cache keys (#7) stay correct.
        lang: ``"en"`` or ``"fr"``. Defaults to ``$FORTRANSPIRE_LANG`` and
            then to ``"en"``. Always falls back to the EN copy if a FR
            file is missing.
        **variables: passed to ``str.format`` — every ``{key}`` in the
            prompt must have a matching kwarg or :class:`KeyError`
            is raised.
    """
    resolved_lang = lang or os.getenv("FORTRANSPIRE_LANG", "en")
    template = _read(name, version, resolved_lang)
    if not variables:
        # Common case — no substitutions needed, skip the formatter to
        # avoid surprising errors when the prompt happens to contain a
        # literal `{`.
        return template
    return _SafeFormatter(name).vformat(template, (), variables)


def clear_cache() -> None:
    """Wipe the on-disk read cache — useful for tests that edit prompt files."""
    _read.cache_clear()


# Stage → role mapping. Nodes can import this rather than hard-coding strings.
STAGE_FOR_PROMPT: dict[str, str] = {
    "extractor":      "reasoning",
    "openacc_kernel": "reasoning",
    "openacc_driver": "reasoning",
    "cython_pyx":     "code",
    "cython_header":  "code",
    "doc_routine":    "code",
}
