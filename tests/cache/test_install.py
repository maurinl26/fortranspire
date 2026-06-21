"""Tests for the global-install hook + env-var driven config."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from fortranspire.cache import (
    FortranspireLRUCache,
    get_cache,
    install_global_cache,
    is_enabled,
)
from fortranspire.cache.install import disable


@pytest.fixture(autouse=True)
def _isolate_install():
    disable()
    yield
    disable()


def test_is_enabled_default_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FORTRANSPIRE_CACHE", raising=False)
    assert is_enabled() is True


def test_is_enabled_off_variants(monkeypatch: pytest.MonkeyPatch):
    for value in ("off", "OFF", "0", "false", "no"):
        monkeypatch.setenv("FORTRANSPIRE_CACHE", value)
        assert is_enabled() is False, f"FORTRANSPIRE_CACHE={value!r} should disable"


def test_install_returns_none_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORTRANSPIRE_CACHE", "off")
    result = install_global_cache(db_path=tmp_path / "c.db")
    assert result is None
    assert get_cache() is None


def test_install_sets_global_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_CACHE", "on")
    cache = install_global_cache(db_path=tmp_path / "c.db", max_gb=0.001)
    assert cache is not None
    assert isinstance(cache, FortranspireLRUCache)
    assert get_cache() is cache


def test_install_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_CACHE", "on")
    a = install_global_cache(db_path=tmp_path / "a.db")
    b = install_global_cache(db_path=tmp_path / "b.db")
    assert a is not b
    assert get_cache() is b   # second install replaces first


def test_env_var_directory_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FORTRANSPIRE_CACHE", "on")
    monkeypatch.setenv("FORTRANSPIRE_CACHE_DIR", str(tmp_path))
    cache = install_global_cache()
    assert cache is not None
    assert Path(cache.db_path).is_relative_to(tmp_path)


def test_env_var_max_gb_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FORTRANSPIRE_CACHE", "on")
    monkeypatch.setenv("FORTRANSPIRE_CACHE_MAX_GB", "0.0001")  # ~100 KB
    monkeypatch.setenv("FORTRANSPIRE_CACHE_DIR", str(tmp_path))
    cache = install_global_cache()
    assert cache is not None
    assert cache.max_bytes < 200_000
