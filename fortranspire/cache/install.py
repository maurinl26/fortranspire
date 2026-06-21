"""Install the fortranspire LRU cache as LangChain's process-wide LLM cache.

Called automatically when :mod:`fortranspire.llm` is first imported.
Reads env vars to decide whether to enable the cache, where to put the
DB, and what byte cap to enforce.

Manual install is also fine — useful for tests or notebooks that want
to bind a per-run cache::

    from fortranspire.cache import install_global_cache
    install_global_cache(db_path="/tmp/run.db", max_gb=0.1)
"""
from __future__ import annotations

import os
from pathlib import Path

from fortranspire.cache.lru import FortranspireLRUCache

_INSTALLED: FortranspireLRUCache | None = None


def is_enabled() -> bool:
    """``False`` when ``FORTRANSPIRE_CACHE=off``."""
    return os.getenv("FORTRANSPIRE_CACHE", "on").lower() not in ("off", "0", "false", "no")


def _default_dir() -> Path:
    raw = os.getenv("FORTRANSPIRE_CACHE_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "fortranspire"


def _default_max_bytes() -> int:
    raw = os.getenv("FORTRANSPIRE_CACHE_MAX_GB", "1")
    try:
        gb = float(raw)
    except ValueError:
        gb = 1.0
    return int(gb * (1 << 30))


def get_cache() -> FortranspireLRUCache | None:
    """Return the currently installed cache instance, if any."""
    return _INSTALLED


def install_global_cache(
    *,
    db_path: str | Path | None = None,
    max_gb: float | None = None,
) -> FortranspireLRUCache | None:
    """Wire :class:`FortranspireLRUCache` as the global LangChain LLM cache.

    No-op when ``FORTRANSPIRE_CACHE=off`` or when langchain isn't
    installed (analyze-only image). Idempotent — calling twice replaces
    the previous instance.

    Returns the installed cache (or ``None`` if disabled).
    """
    global _INSTALLED

    if not is_enabled():
        return None

    try:
        from langchain_core.globals import set_llm_cache
    except ImportError:
        # `[gpu]` extra not installed — silently do nothing. The caller
        # is most likely an analyze-only or doc-only run that doesn't
        # fire any LLM call anyway.
        return None

    path = Path(db_path) if db_path else _default_dir() / "llm.db"
    max_bytes = int(max_gb * (1 << 30)) if max_gb is not None else _default_max_bytes()

    cache = FortranspireLRUCache(db_path=path, max_bytes=max_bytes)
    set_llm_cache(cache)
    _INSTALLED = cache
    return cache


def disable() -> None:
    """Unset the global cache (testing aid)."""
    global _INSTALLED
    try:
        from langchain_core.globals import set_llm_cache
        set_llm_cache(None)
    except ImportError:
        pass
    _INSTALLED = None
