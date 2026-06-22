"""Equivalence harness on real (small but realistic) kernels — issue #45.

For each kernel under ``tests/fixtures/equivalence/<kernel>/``:

1. Compile ``original.f90`` + ``driver.f90`` with ``gfortran -O2`` → binary A.
2. Compile ``openacc.f90`` + ``driver.f90`` with ``gfortran -O2 -fopenacc``
   → binary B. (gfortran parses the OpenACC pragmas and parallelises via
   libgomp on CPU — no GPU required.)
3. Run both binaries; capture stdout.
4. Parse the ``(i, j) value`` lines into ``np.ndarray``s.
5. Assert ``np.allclose`` within the tolerance documented in the
   kernel's ``TOLERANCE.md``.

Tests are marked ``slow`` because they shell out to gfortran. Run with
``pytest -m slow`` or ``pytest tests/test_equivalence_real_kernels.py``.

The mechanic is intentionally **LLM-free**: ``openacc.f90`` is checked
into the fixture, hand-written, byte-identical-arithmetic to
``original.f90`` with just pragmas added. An end-to-end LLM-generated
variant belongs in a separate test (e.g. ``pytest -m llm``) so CI
without an API key still runs this gate.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "equivalence"


def _has_gfortran() -> bool:
    return shutil.which("gfortran") is not None


def _compile(workdir: Path, sources: list[Path], output: Path,
             flags: list[str]) -> tuple[bool, str]:
    """Compile ``sources`` to ``output`` with the given flags. Returns
    ``(ok, combined_stderr)``."""
    cmd = ["gfortran", *flags, *(str(s) for s in sources), "-o", str(output)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                       cwd=str(workdir))
    return r.returncode == 0, (r.stderr or r.stdout)


def _run(binary: Path) -> str:
    r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"{binary.name} failed:\n{r.stderr}"
    return r.stdout


def _parse_probes(stdout: str) -> np.ndarray:
    """Each line is ``i j value`` (1-indexed, space-separated). Returns
    a 2D array sorted by (i, j) so the comparison is order-independent."""
    rows = []
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
        rows.append((i, j, v))
    rows.sort()
    return np.asarray([r[2] for r in rows], dtype=np.float64)


# ── Per-kernel parameter set ────────────────────────────────────────────

# (name, atol, rtol) per kernel. Tolerances are also documented in the
# kernel's TOLERANCE.md; this dict is the single source of truth for
# the test. Adding a kernel = add a row + a fixture directory.
KERNELS = [
    ("wave_kernels", 1e-12, 1e-10),
]


@pytest.mark.slow
@pytest.mark.parametrize("kernel,atol,rtol", KERNELS,
                         ids=[k[0] for k in KERNELS])
def test_fortran_vs_openacc_equivalence(
    tmp_path: Path, kernel: str, atol: float, rtol: float,
):
    """Original Fortran and OpenACC variants must agree within tolerance."""
    if not _has_gfortran():
        pytest.skip("gfortran not in PATH — install with `brew install gcc` "
                    "or `apt install gfortran`")

    src_dir = FIXTURE_DIR / kernel
    assert src_dir.is_dir(), f"missing fixture directory: {src_dir}"

    # Copy sources into tmp_path so we don't write build artifacts into
    # the repo tree.
    work = tmp_path / kernel
    work.mkdir()
    for f in ("original.f90", "openacc.f90", "driver.f90"):
        shutil.copy(src_dir / f, work / f)

    # ── Build A — original (serial, no OpenACC) ───────────────────────────
    bin_a = work / "binary_original"
    ok, log = _compile(
        work, [work / "original.f90", work / "driver.f90"],
        bin_a, ["-O2"],
    )
    assert ok, f"original failed to build:\n{log}"

    # ── Build B — OpenACC variant (parsed by gfortran -fopenacc) ──────────
    bin_b = work / "binary_openacc"
    ok, log = _compile(
        work, [work / "openacc.f90", work / "driver.f90"],
        bin_b, ["-O2", "-fopenacc"],
    )
    assert ok, f"openacc failed to build:\n{log}"

    # ── Run both ──────────────────────────────────────────────────────────
    out_a = _parse_probes(_run(bin_a))
    out_b = _parse_probes(_run(bin_b))

    assert out_a.shape == out_b.shape, (
        f"probe-grid mismatch: original={out_a.shape}, openacc={out_b.shape}"
    )
    assert out_a.size > 0, "driver emitted no probe points"

    # ── Assert numerical equivalence ─────────────────────────────────────
    if not np.allclose(out_a, out_b, atol=atol, rtol=rtol):
        # Build a detailed diagnostic — surface where the divergence
        # happens (issue #45 deliverable E: "report" must say *why*).
        absdiff = np.abs(out_a - out_b)
        worst = int(np.argmax(absdiff))
        msg = [
            f"Kernel `{kernel}` failed equivalence check "
            f"(atol={atol:g}, rtol={rtol:g}).",
            f"  N probes     : {out_a.size}",
            f"  Max abs diff : {absdiff.max():.3e} (at probe index {worst})",
            f"  original[{worst}] = {out_a[worst]:.17e}",
            f"  openacc[{worst}]  = {out_b[worst]:.17e}",
        ]
        pytest.fail("\n".join(msg))


@pytest.mark.slow
def test_tolerance_file_exists_for_every_kernel():
    """Every fixture directory must ship a TOLERANCE.md so the rationale
    for the tolerance values lives next to the data."""
    for kernel, _, _ in KERNELS:
        tol_file = FIXTURE_DIR / kernel / "TOLERANCE.md"
        assert tol_file.is_file(), (
            f"missing TOLERANCE.md for kernel `{kernel}` — every fixture "
            f"directory must document its tolerance choice."
        )
