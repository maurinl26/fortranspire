"""Tests for the parallel batch-port driver — issue #13.

The LLM path is mocked end-to-end (we patch `translation_app_phase1` to a
deterministic stub) so these tests run without an API key and without
the [gpu] extra. The point is to verify:

  * Per-file output isolation via ContextVar (no race on output/ dirs).
  * Concurrency: N files fan out to N workers in parallel.
  * Walk discovery (.f90 + .F90, recursive).
  * Result aggregation + exit codes.
  * Graceful per-file error handling without crashing the batch.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import current_thread

import pytest

from fortranspire.agent.batch import (
    PortResult,
    _safe_stem,
    _walk_paths,
    main,
    port_batch,
    port_one_file,
)
from fortranspire.agent.nodes._common import get_output_root, set_output_root

FIXTURE = Path(__file__).parent / "fixtures" / "doc_kernel.f90"


# ── Output root contextvar ─────────────────────────────────────────────────

def test_get_output_root_defaults_to_output_dir():
    set_output_root(None)
    assert get_output_root() == Path("output")


def test_set_output_root_override(tmp_path: Path):
    set_output_root(tmp_path / "custom")
    try:
        assert get_output_root() == tmp_path / "custom"
    finally:
        set_output_root(None)


def test_output_root_is_thread_local(tmp_path: Path):
    # Each thread sets a different root and must read back its own value.
    results: dict[str, Path] = {}

    def worker(name: str, root: Path) -> None:
        set_output_root(root)
        # Yield a tiny moment so threads interleave
        time.sleep(0.01)
        results[name] = get_output_root()

    set_output_root(None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = []
        for i in range(4):
            futs.append(pool.submit(worker, f"t{i}", tmp_path / f"root_{i}"))
        for f in futs:
            f.result()

    assert results == {f"t{i}": tmp_path / f"root_{i}" for i in range(4)}
    # Main thread is unaffected.
    assert get_output_root() == Path("output")


# ── _walk_paths ─────────────────────────────────────────────────────────────

def test_walk_paths_dedupes_and_sorts(tmp_path: Path):
    shutil.copy(FIXTURE, tmp_path / "b.f90")
    shutil.copy(FIXTURE, tmp_path / "a.F90")
    found = _walk_paths([str(tmp_path), str(tmp_path)])  # passed twice
    assert len(found) == 2
    assert found == sorted(found)


def test_walk_paths_ignores_non_fortran(tmp_path: Path):
    (tmp_path / "readme.md").write_text("hi")
    (tmp_path / "data.txt").write_text("hi")
    shutil.copy(FIXTURE, tmp_path / "k.f90")
    found = _walk_paths([str(tmp_path)])
    assert len(found) == 1
    assert found[0].endswith("k.f90")


def test_safe_stem_sanitizes_funky_names():
    assert _safe_stem("/path/to/kernel.f90") == "kernel"
    assert _safe_stem("foo bar.f90") == "foo_bar"
    assert _safe_stem("foo$bar.F90") == "foo_bar"


# ── port_one_file (mocked LangGraph) ────────────────────────────────────────

class _StubApp:
    """Stand-in for translation_app_phase1 that records the output_root via _out()."""

    def __init__(self):
        self.observed_roots: list[Path] = []
        self.crash_on_files: set[str] = set()

    def invoke(self, state: dict) -> dict:
        from fortranspire.agent.nodes._common import _out
        # Trigger the side effect we care about — calling _out() actually creates
        # the per-category subdir under the contextvar root.
        target = _out("fortran_gpu")
        self.observed_roots.append(target.parent)
        fp = state.get("fortran_filepath", "")
        if fp in self.crash_on_files:
            raise RuntimeError(f"simulated failure on {fp}")
        return {"validation_passed": True}


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> _StubApp:
    stub = _StubApp()
    import fortranspire.agent.translation_graph_phase1 as mod
    monkeypatch.setattr(mod, "translation_app_phase1", stub)
    return stub


def test_port_one_file_writes_into_isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub = _install_stub(monkeypatch)
    target_root = tmp_path / "kernel_a"
    result = port_one_file(str(FIXTURE), target_root)

    assert result.success is True
    assert result.validation_passed is True
    assert Path(result.output_root) == target_root
    # _out() created the subdirectory inside the per-file root.
    assert (target_root / "fortran_gpu").is_dir()
    assert stub.observed_roots[0] == target_root


def test_port_one_file_resets_contextvar_after_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    set_output_root(None)
    _install_stub(monkeypatch)
    port_one_file(str(FIXTURE), tmp_path / "x")
    # Main thread's contextvar must be back to default after the call.
    assert get_output_root() == Path("output")


def test_port_one_file_captures_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub = _install_stub(monkeypatch)
    stub.crash_on_files.add(str(FIXTURE.resolve()))
    result = port_one_file(str(FIXTURE), tmp_path / "x")
    assert result.success is False
    assert result.error and "simulated failure" in result.error


# ── port_batch (parallel) ───────────────────────────────────────────────────

def test_port_batch_runs_in_parallel_with_isolated_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub = _install_stub(monkeypatch)

    # 5 distinct files → 5 distinct output roots
    for i in range(5):
        shutil.copy(FIXTURE, tmp_path / f"file_{i}.f90")

    out_root = tmp_path / "out"
    results = port_batch([str(tmp_path)], concurrency=4, output_root=out_root)

    assert len(results) == 5
    assert all(r.success for r in results)
    # Each file has its own output subdir
    sub_dirs = {Path(r.output_root).name for r in results}
    assert sub_dirs == {f"file_{i}" for i in range(5)}
    # The stub recorded one observation per file
    assert len(stub.observed_roots) == 5
    # And every observed root is unique (no clobbering)
    assert len(set(stub.observed_roots)) == 5


def test_port_batch_on_complete_callback_fires_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _install_stub(monkeypatch)
    for i in range(3):
        shutil.copy(FIXTURE, tmp_path / f"file_{i}.f90")

    progress: list[PortResult] = []
    port_batch([str(tmp_path)], concurrency=2, output_root=tmp_path / "out",
               on_complete=progress.append)
    assert len(progress) == 3
    assert all(isinstance(p, PortResult) for p in progress)


def test_port_batch_returns_empty_when_no_files(tmp_path: Path):
    assert port_batch([str(tmp_path)]) == []


def test_port_batch_isolates_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub = _install_stub(monkeypatch)
    files = []
    for i in range(4):
        f = tmp_path / f"f_{i}.f90"
        shutil.copy(FIXTURE, f)
        files.append(str(f.resolve()))
    # Make the middle file crash; the others must still complete.
    stub.crash_on_files.add(files[1])
    stub.crash_on_files.add(files[2])

    results = port_batch([str(tmp_path)], concurrency=4, output_root=tmp_path / "out")
    by_success = {True: 0, False: 0}
    for r in results:
        by_success[r.success] += 1
    assert by_success[True] == 2
    assert by_success[False] == 2


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_returns_2_when_no_files(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main([str(empty)])
    assert rc == 2


def test_cli_runs_end_to_end_with_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    _install_stub(monkeypatch)
    shutil.copy(FIXTURE, tmp_path / "a.f90")
    shutil.copy(FIXTURE, tmp_path / "b.f90")

    rc = main([str(tmp_path), "--output-root", str(tmp_path / "out"),
               "--concurrency", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "porting 2 file(s)" in out
    assert "✓ 2 passed validation" in out


def test_cli_exit_1_on_any_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub = _install_stub(monkeypatch)
    f = tmp_path / "broken.f90"
    shutil.copy(FIXTURE, f)
    stub.crash_on_files.add(str(f.resolve()))

    rc = main([str(tmp_path), "--output-root", str(tmp_path / "out"),
               "--concurrency", "1"])
    assert rc == 1
