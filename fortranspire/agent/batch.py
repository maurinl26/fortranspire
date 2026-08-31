"""``agent-port-batch`` — parallel Fortran → GPU port across many files.

Closes issue #13.

The single-file CLI (`agent-gpu`) is fine for one kernel. For a 100-routine
codebase (typical seismic / NWP / CFD ports), running it sequentially
takes ~3 hours and burns ~6 USD in tokens. This batch driver pipelines
the pipeline:

- Walks the input paths, collects every `.f90` / `.F90`.
- Runs N ports concurrently via a thread pool (default `min(4, cpu_count/2)`).
- Each worker sets its own output root via :func:`set_output_root` so the
  ports don't clobber each other's `output/fortran_gpu/` and `output/cython/`.
- Per-kernel status line as each completes; final summary table at the end.
- Mistral rate-limit handling delegated to LangChain's built-in retry
  (`max_retries`) on the LLM client — no custom backoff needed here.

Per-file output layout::

    output/<file_stem>/fortran_gpu/...
    output/<file_stem>/cython/...
    output/<file_stem>/tests/...
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fortranspire.agent.nodes._common import collect_fortran_files, set_output_root


@dataclass
class PortResult:
    """Outcome of one file's port — keyed back to the input path."""

    path: str
    output_root: str
    success: bool
    validation_passed: bool
    duration_s: float
    error: str | None = None


def _initial_state(filepath: str) -> dict:
    """Shape of the Phase1State the LangGraph workflow expects."""
    code = Path(filepath).read_text(encoding="utf-8")
    return {
        "fortran_filepath": str(Path(filepath).resolve()),
        "fortran_code": code,
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "module_fortran": "",
        "driver_fortran": "",
        "kernel_names": [],
        "pure_elemental_fortran": "",
        "openacc_fortran": "",
        "cython_pyx": "",
        "cython_header": "",
        "cython_setup": "",
        "validation_passed": False,
        "validation_log": "",
        "executed_agents": [],
    }


def port_one_file(filepath: str, output_root: Path) -> PortResult:
    """Port a single file, isolating its outputs under ``output_root``.

    Designed to be called from a worker thread — sets the per-thread
    ``_OUTPUT_ROOT`` contextvar so the pipeline's `_out()` helper writes
    everything under the per-file root. Other threads in the same pool
    each have their own contextvar, so concurrent calls don't collide.
    """
    # The full pipeline is sync; we wrap its imports here so this module
    # loads without the [gpu] extra (matches the lazy-import pattern in
    # cli.py).
    from fortranspire.agent.translation_graph_phase1 import translation_app_phase1

    set_output_root(output_root)
    start = time.time()
    try:
        final = translation_app_phase1.invoke(_initial_state(filepath))
        return PortResult(
            path=filepath,
            output_root=str(output_root),
            success=True,
            validation_passed=bool(final.get("validation_passed")),
            duration_s=time.time() - start,
        )
    except Exception as exc:
        return PortResult(
            path=filepath,
            output_root=str(output_root),
            success=False,
            validation_passed=False,
            duration_s=time.time() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # Reset so the thread is clean if the executor reuses it.
        set_output_root(None)


def _walk_paths(paths: Iterable[str]) -> list[str]:
    """Flatten file + directory paths into a deduped, sorted list of .f90 files."""
    files: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            # Shared discovery: fixed-form suffixes too (.F, .f, .for),
            # which are most of the legacy corpus.
            files.extend(collect_fortran_files([p]))
        elif p.is_file():
            files.append(str(p))
    seen: set[str] = set()
    return [f for f in sorted(files) if not (f in seen or seen.add(f))]


def _safe_stem(filepath: str) -> str:
    """Sanitize a filename into a directory-friendly stem."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filepath).stem)


def port_batch(
    paths: Iterable[str],
    *,
    concurrency: int | None = None,
    output_root: str | Path = "output",
    on_complete=None,
) -> list[PortResult]:
    """Port every Fortran file under ``paths`` in parallel.

    Args:
        paths: files and/or directories. Directories are walked for
            ``*.f90`` / ``*.F90``.
        concurrency: max simultaneous ports. Default
            ``min(4, cpu_count/2)`` — Mistral API rate-limit-friendly.
        output_root: parent directory; each file gets
            ``<output_root>/<file_stem>/``.
        on_complete: optional callback receiving each :class:`PortResult`
            as soon as the corresponding worker finishes. Useful for CLI
            progress lines / tests.

    Returns the list of :class:`PortResult` in completion order.
    """
    files = _walk_paths(paths)
    if not files:
        return []

    n_workers = concurrency or max(1, min(4, multiprocessing.cpu_count() // 2))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[PortResult] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(port_one_file, fp, output_root / _safe_stem(fp)): fp
            for fp in files
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if on_complete is not None:
                on_complete(result)
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_progress(result: PortResult, *, total: int, counter: list[int]) -> None:
    counter[0] += 1
    n = counter[0]
    status = "✓" if result.success and result.validation_passed else (
        "⚠" if result.success else "✗"
    )
    suffix = ""
    if result.error:
        suffix = f"  ({result.error})"
    elif result.success and not result.validation_passed:
        suffix = "  (validation failed — see output)"
    print(f"  [{n:>3}/{total}] {status} {Path(result.path).name}  "
          f"{result.duration_s:5.1f}s{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-port-batch",
        description=(
            "Port many Fortran kernels to GPU in parallel. Each file gets "
            "its own `output/<stem>/` subdirectory so concurrent workers "
            "don't collide. Concurrency defaults to min(4, CPU/2) — kind "
            "to Mistral API rate limits."
        ),
        epilog=(
            "Examples:\n"
            "  agent-port-batch src/                       # whole codebase\n"
            "  agent-port-batch --concurrency 8 src/       # 8 parallel ports\n"
            "  agent-port-batch -o /tmp/ports src/         # custom output root\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+",
                        help="Fortran files or directories to port")
    parser.add_argument("--concurrency", "-j", type=int, default=None,
                        help="Max simultaneous ports (default: min(4, CPU/2))")
    parser.add_argument("--output-root", "-o", default="output",
                        help="Parent directory for per-file outputs (default: output)")
    args = parser.parse_args(argv)

    files = _walk_paths(args.paths)
    if not files:
        print("agent-port-batch: no .f90 / .F90 file found in the given paths",
              file=sys.stderr)
        return 2

    n_workers = args.concurrency or max(1, min(4, multiprocessing.cpu_count() // 2))
    print(f"agent-port-batch: porting {len(files)} file(s) with concurrency={n_workers}")
    print(f"  Output root: {args.output_root}/<file_stem>/")
    print()

    counter = [0]
    started = time.time()
    results = port_batch(
        args.paths,
        concurrency=n_workers,
        output_root=args.output_root,
        on_complete=lambda r: _print_progress(r, total=len(files), counter=counter),
    )
    total_duration = time.time() - started

    n_ok    = sum(1 for r in results if r.success and r.validation_passed)
    n_warn  = sum(1 for r in results if r.success and not r.validation_passed)
    n_fail  = sum(1 for r in results if not r.success)

    print()
    print(f"agent-port-batch: done in {total_duration:.1f}s")
    print(f"  ✓ {n_ok} passed validation")
    print(f"  ⚠ {n_warn} ported but validation failed")
    print(f"  ✗ {n_fail} errored out")

    # Exit code: 0 if every file ported (validation may have failed but the
    # pipeline didn't crash). Non-zero only when at least one worker raised.
    return 1 if n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
