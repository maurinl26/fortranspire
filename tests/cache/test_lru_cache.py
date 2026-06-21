"""Tests for the SQLite-backed LRU LLM response cache — issue #7.

These tests do not fire any LLM call — they verify the cache's read /
write / evict semantics directly through the BaseCache contract.
"""
from __future__ import annotations

import pickle
import sqlite3
import time
from pathlib import Path
from threading import Thread

import pytest

# The cache module imports langchain_core.caches lazily; the tests that
# touch the BaseCache contract need it on PATH. Skip the file when not.
pytest.importorskip("langchain_core")

from langchain_core.outputs import Generation

from fortranspire.cache.lru import (
    FortranspireLRUCache,
    reset_stats,
    stats,
)


def _gen(text: str) -> list[Generation]:
    return [Generation(text=text)]


# ── Basic read / write ──────────────────────────────────────────────────────

def test_lookup_returns_none_when_empty(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    assert cache.lookup("hello", "llm-string") is None


def test_round_trip(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("hello", "llm-x", _gen("world"))
    out = cache.lookup("hello", "llm-x")
    assert out is not None
    assert isinstance(out, list)
    assert out[0].text == "world"


def test_different_llm_string_means_different_key(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("prompt-A", "llm-1", _gen("v1"))
    cache.update("prompt-A", "llm-2", _gen("v2"))
    assert cache.lookup("prompt-A", "llm-1")[0].text == "v1"
    assert cache.lookup("prompt-A", "llm-2")[0].text == "v2"


def test_clear_wipes_everything(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("p", "l", _gen("v"))
    assert cache.n_entries() == 1
    cache.clear()
    assert cache.n_entries() == 0
    assert cache.lookup("p", "l") is None


# ── LRU eviction ────────────────────────────────────────────────────────────

def _measure_pickle_size(cache: FortranspireLRUCache, value: str) -> int:
    """Pickle size of one entry — used to size the LRU cap precisely so the
    eviction tests assert deterministic behavior regardless of pickle overhead."""
    cache.update("__probe__", "__llm__", _gen(value))
    size = cache.total_bytes()
    cache.clear()
    return size


def test_evicts_oldest_when_over_cap(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db", max_bytes=999_999)
    big = "x" * 150
    per_entry = _measure_pickle_size(cache, big)
    # Size cap to fit exactly 2 entries — the 3rd insert evicts one.
    cache.max_bytes = int(per_entry * 2.5)

    cache.update("a", "l", _gen(big))
    time.sleep(0.01)
    cache.update("b", "l", _gen(big))
    time.sleep(0.01)
    cache.update("c", "l", _gen(big))
    # 'a' is oldest by last_access → evicted.
    assert cache.lookup("a", "l") is None
    assert cache.lookup("b", "l") is not None
    assert cache.lookup("c", "l") is not None


def test_hit_promotes_in_lru(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db", max_bytes=999_999)
    big = "x" * 150
    per_entry = _measure_pickle_size(cache, big)
    cache.max_bytes = int(per_entry * 2.5)

    cache.update("a", "l", _gen(big))
    time.sleep(0.01)
    cache.update("b", "l", _gen(big))
    time.sleep(0.01)
    # Read 'a' → it becomes the most recently accessed.
    cache.lookup("a", "l")
    time.sleep(0.01)
    cache.update("c", "l", _gen(big))
    # Now 'b' (oldest access) is evicted instead of 'a'.
    assert cache.lookup("b", "l") is None
    assert cache.lookup("a", "l") is not None
    assert cache.lookup("c", "l") is not None


def test_total_bytes_tracking(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db", max_bytes=10_000)
    cache.update("p", "l", _gen("hello"))
    assert cache.total_bytes() > 0
    assert cache.n_entries() == 1


# ── Counters ────────────────────────────────────────────────────────────────

def test_hit_miss_counters(tmp_path: Path):
    reset_stats()
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("p", "l", _gen("v"))
    cache.lookup("p", "l")            # hit
    cache.lookup("p", "different-llm")  # miss
    s = stats()
    assert s["hits"] >= 1
    assert s["misses"] >= 1
    assert s["writes"] >= 1


def test_reset_stats_clears_counters(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("p", "l", _gen("v"))
    cache.lookup("p", "l")
    reset_stats()
    assert stats() == {"hits": 0, "misses": 0, "writes": 0}


# ── Hit log ────────────────────────────────────────────────────────────────

def test_hit_log_writes_one_line_per_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    log_path = tmp_path / "hits.log"
    monkeypatch.setenv("FORTRANSPIRE_CACHE_HIT_LOG", str(log_path))
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    cache.update("p", "[llm=ChatOpenAI(model='mistral-large-latest')]", _gen("v"))
    cache.lookup("p", "[llm=ChatOpenAI(model='mistral-large-latest')]")
    cache.lookup("other", "[llm=ChatOpenAI(model='mistral-large-latest')]")
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert " hit "  in lines[0]
    assert " miss " in lines[1]
    assert "mistral-large-latest" in lines[0]
    assert "mistral-large-latest" in lines[1]


# ── Pickle corruption recovery ─────────────────────────────────────────────

def test_corrupted_pickle_evicts_silently(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    # Inject a corrupt blob directly into the table.
    key = FortranspireLRUCache._key("p", "l")
    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("""
            INSERT INTO cache (key, value, size, last_access, created_at, model, prompt_sha)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (key, b"\x80\x99\x00not-a-pickle", 16, time.time(), time.time(), None, "x"))
    # Lookup must not crash; cache evicts the bad row and returns None.
    assert cache.lookup("p", "l") is None
    assert cache.n_entries() == 0


# ── Concurrency (lightweight) ──────────────────────────────────────────────

def test_thread_safety_smoke(tmp_path: Path):
    cache = FortranspireLRUCache(db_path=tmp_path / "c.db")
    errors: list[Exception] = []

    def worker(prefix: str) -> None:
        try:
            for i in range(20):
                cache.update(f"{prefix}-{i}", "l", _gen(f"value-{i}"))
                cache.lookup(f"{prefix}-{i}", "l")
        except Exception as exc:
            errors.append(exc)

    threads = [Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cache.n_entries() >= 60  # 4 threads × 20 keys = 80, modulo eviction
