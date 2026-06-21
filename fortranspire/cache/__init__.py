"""Content-addressed LLM response cache — issue #7.

Wraps every LLM call routed through LangChain's global cache so re-runs
on unchanged input return instantly with **zero token cost**.

Key shape: SHA-256 of ``(model, prompt_text, llm_params)`` — derived by
LangChain's :class:`langchain_core.caches.BaseCache` contract. Adding
the prompt-version metadata from issue #3 happens automatically because
the rendered prompt body changes when we bump ``v1`` → ``v2``, which
changes the hash, which forces a re-fetch.

Backend: SQLite (stdlib, no dep), one DB file under
``~/.cache/fortranspire/llm.db`` by default. LRU eviction at the
configured byte cap (default 1 GB).

Configuration (all opt-out, defaults sane):

- ``FORTRANSPIRE_CACHE=off``     — disable entirely (no install, no read, no write)
- ``FORTRANSPIRE_CACHE_DIR=...`` — override the DB directory
- ``FORTRANSPIRE_CACHE_MAX_GB=N`` — LRU cap in GiB (default 1)
- ``FORTRANSPIRE_CACHE_HIT_LOG=path`` — append one line per hit/miss for
  metrics (default off)

Install hook: :func:`install_global_cache` is called automatically when
``fortranspire.llm`` is imported (which only happens under the [gpu]
extra). Manual install is fine for tests.
"""
from fortranspire.cache.lru import FortranspireLRUCache
from fortranspire.cache.install import install_global_cache, is_enabled, get_cache

__all__ = ["FortranspireLRUCache", "install_global_cache", "is_enabled", "get_cache"]
