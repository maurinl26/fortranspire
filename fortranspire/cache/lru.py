"""SQLite-backed LRU cache implementing LangChain's ``BaseCache`` protocol.

Stores one row per ``(prompt, llm_string)`` key. The ``value`` column
holds the pickled ``list[Generation]`` LangChain hands back from
``BaseCache.update``. Pickle is acceptable here because every value we
serialize comes from LangChain's own runtime — never from untrusted
external input.

LRU eviction triggers on every ``update()`` when the total size of
stored values exceeds ``max_bytes``. Eviction deletes the least
recently *accessed* rows (hit + insert both bump ``last_access``).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from threading import RLock
from typing import Any, Optional, Sequence

try:
    # Available only under the [gpu] extra; the cache itself loads without it,
    # so analyze-only callers can still import the module.
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation
    _LANGCHAIN_OK = True
except ImportError:  # pragma: no cover — tests under [tests] only
    BaseCache = object  # type: ignore[misc, assignment]
    Generation = None   # type: ignore[assignment]
    _LANGCHAIN_OK = False


_HIT_COUNTER = {"hits": 0, "misses": 0, "writes": 0}


def stats() -> dict[str, int]:
    """Return process-wide hit/miss/write counters since module load."""
    return dict(_HIT_COUNTER)


def reset_stats() -> None:
    """Reset the counters — useful for per-run accounting in tests."""
    _HIT_COUNTER["hits"] = 0
    _HIT_COUNTER["misses"] = 0
    _HIT_COUNTER["writes"] = 0


class FortranspireLRUCache(BaseCache):  # type: ignore[misc, valid-type]
    """SQLite-backed LRU cache for LLM responses.

    Thread-safe via an internal RLock plus SQLite's connection-per-call
    pattern (avoids the "SQLite objects created in a thread can only be
    used in that same thread" trap when LangChain's runtime fans calls
    across threads).
    """

    def __init__(self, db_path: str | Path, max_bytes: int = 1 << 30) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = RLock()
        self._init_db()

    # ── SQLite plumbing ────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key         TEXT PRIMARY KEY,
                    value       BLOB NOT NULL,
                    size        INTEGER NOT NULL,
                    last_access REAL NOT NULL,
                    created_at  REAL NOT NULL,
                    model       TEXT,
                    prompt_sha  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON cache(last_access)")

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(llm_string.encode("utf-8"))
        return h.hexdigest()

    # ── BaseCache contract ────────────────────────────────────────────────

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Any]]:
        key = self._key(prompt, llm_string)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                _HIT_COUNTER["misses"] += 1
                self._maybe_log("miss", key, llm_string)
                return None
            conn.execute("UPDATE cache SET last_access = ? WHERE key = ?",
                         (time.time(), key))
        _HIT_COUNTER["hits"] += 1
        self._maybe_log("hit", key, llm_string)
        try:
            return pickle.loads(row[0])
        except Exception as exc:
            print(f"fortranspire.cache: pickle restore failed for key={key[:12]}… "
                  f"({type(exc).__name__}); evicting", file=sys.stderr)
            self._delete(key)
            return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
        key = self._key(prompt, llm_string)
        blob = pickle.dumps(return_val)
        size = len(blob)
        now = time.time()
        model = self._guess_model(llm_string)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache
                  (key, value, size, last_access, created_at, model, prompt_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (key, blob, size, now, now, model, prompt_sha))
            self._evict_if_needed(conn)
        _HIT_COUNTER["writes"] += 1

    def clear(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

    # ── Maintenance helpers ───────────────────────────────────────────────

    def _delete(self, key: str) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def _evict_if_needed(self, conn: sqlite3.Connection) -> int:
        """Drop LRU rows until total size ≤ max_bytes. Returns count evicted."""
        total = conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]
        if total <= self.max_bytes:
            return 0
        evicted = 0
        for key, size in conn.execute(
            "SELECT key, size FROM cache ORDER BY last_access ASC"
        ):
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            total -= size
            evicted += 1
            if total <= self.max_bytes:
                break
        return evicted

    @staticmethod
    def _guess_model(llm_string: str) -> str | None:
        """Best-effort model-name extraction from LangChain's llm_string.

        ``llm_string`` looks like ``[llm=ChatOpenAI(model='mistral-large-latest',
        temperature=0.0, ...), stop=...]``. We grep for ``model='...'`` —
        good enough for billing-by-model rollups; not load-bearing.
        """
        import re
        m = re.search(r"model['\"]?\s*[:=]\s*['\"]([\w.\-]+)", llm_string)
        return m.group(1) if m else None

    def _maybe_log(self, outcome: str, key: str, llm_string: str) -> None:
        log_path = os.getenv("FORTRANSPIRE_CACHE_HIT_LOG")
        if not log_path:
            return
        model = self._guess_model(llm_string) or "unknown"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.time():.3f} {outcome} {model} {key[:16]}\n")
        except OSError:
            pass  # Telemetry never crashes the LLM path.

    # ── Inspection helpers (used by tests + a future `agent-cache` CLI) ──

    def total_bytes(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]

    def n_entries(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
