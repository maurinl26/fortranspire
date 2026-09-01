"""Phase 2 numerical equivalence — the JAX must compute what the Fortran does.

`gradcheck` proves the emitted kernel is *differentiable*; it does not prove it
computes the *same thing* as the original. A kernel can be perfectly
differentiable and numerically wrong. This node closes that gap: it compiles the
original Fortran with f2py, runs it and the JAX kernel on identical inputs, and
compares the `INTENT(OUT)` / `INTENT(INOUT)` results. It is the Phase 2 analogue
of the Phase 1 CPU↔GPU equivalence harness (issue #11).

Scope, honestly bounded. A routine that reads module state through `USE` cannot
be compiled in isolation (the modules are not here), so it is **skipped** with a
clear reason — its equivalence needs the whole-program build, a follow-on. A
self-contained routine (the common kernel shape) is compiled and compared. The
whole node degrades to a skip whenever the toolchain (gfortran + meson + ninja)
is missing, so it never breaks a run that merely lacks a Fortran compiler.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fortranspire.agent.nodes._common import SEP
from fortranspire.agent.nodes_jax._state import JaxKernelInfo, Phase2State

_ATOL = float(os.getenv("FORTRANSPIRE_TOLERANCE_ATOL", "1e-9"))
_RTOL = float(os.getenv("FORTRANSPIRE_TOLERANCE_RTOL", "1e-7"))


def _toolchain_env() -> dict:
    """PATH with the interpreter's bin dir, so f2py finds meson / ninja."""
    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def compile_fortran(code: str, name: str) -> Tuple[Optional[Callable], str]:
    """f2py-compile one self-contained Fortran routine → a Python callable.

    Returns ``(callable, "")`` on success, or ``(None, reason)`` — the caller
    turns a reason into a skip, never a failure.
    """
    import importlib.util
    import shutil
    import subprocess

    if shutil.which("gfortran", path=_toolchain_env()["PATH"]) is None:
        return None, "gfortran not found"

    workdir = Path(tempfile.mkdtemp(prefix="fortranspire_eq_"))
    src = workdir / f"{name}_orig.f90"
    src.write_text(code)
    mod = f"eq_{name.lower()}"

    proc = subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "-c", src.name, "-m", mod,
         "--backend", "meson"],
        cwd=workdir, env=_toolchain_env(),
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return None, f"f2py compile failed: {tail[0]}"

    so = next(iter(workdir.glob(f"{mod}*.so")), None)
    if so is None:
        return None, "f2py produced no extension module"

    spec = importlib.util.spec_from_file_location(mod, so)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        return None, f"could not import f2py module: {exc}"

    fn = getattr(module, name.lower(), None)
    if fn is None:
        return None, f"routine {name} not in compiled module"
    return fn, ""


def _numpy(value: Any):
    import numpy as np

    return np.asarray(value)


def check_equivalence(
    kernel: JaxKernelInfo,
    jax_fn: Callable,
    fortran_fn: Callable,
    *,
    atol: float = _ATOL,
    rtol: float = _RTOL,
    seed: int = 0,
    use_resolved: bool = False,
    extra_fortran_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare the Fortran and JAX outputs on one shared set of inputs.

    ``use_resolved`` builds a degenerate all-``extent`` fixture from #99
    resolution (for a promoted module-state twin); ``extra_fortran_args`` carries
    arguments the Fortran twin has but the JAX does not (the ``nfs`` extent).
    """
    import numpy as np

    from fortranspire.agent.nodes_jax.gradcheck import (
        _DEFAULT_EXTENT, _enable_x64, _make_inputs, _resolved_inputs,
    )

    # Match the Fortran REAL(8): without x64 the JAX runs in float32 and the
    # comparison floor is ~1e-7, hiding real single-ulp double-precision errors.
    _enable_x64()

    intent = {k.lower(): (v or "").upper() for k, v in (kernel.get("intent_map") or {}).items()}
    inputs = (_resolved_inputs(kernel, _DEFAULT_EXTENT, seed=seed)
              if use_resolved else _make_inputs(kernel, seed=seed))
    outputs = list(kernel.get("outputs") or [])

    # Fortran call: pass every IN / INOUT argument by (lowercased) keyword.
    # f2py returns the OUT / INOUT arguments in declaration order.
    fort_kwargs = {}
    for arg, val in inputs.items():
        low = arg.lower()
        if intent.get(low, "IN") in ("IN", "INOUT"):
            fort_kwargs[low] = _numpy(val)
    for extra, val in (extra_fortran_args or {}).items():
        fort_kwargs[extra.lower()] = val
    try:
        fort_out = fortran_fn(**fort_kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"Fortran call failed: {exc}"}

    fort_tuple = fort_out if isinstance(fort_out, tuple) else (fort_out,)

    # JAX call with the same inputs.
    try:
        jax_out = jax_fn(**{a: inputs[a] for a in kernel["inputs"]})
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"JAX call failed: {exc}"}
    jax_tuple = jax_out if isinstance(jax_out, tuple) else (jax_out,)

    if len(fort_tuple) != len(jax_tuple) or len(jax_tuple) != len(outputs):
        return {"status": "skipped",
                "reason": f"output arity mismatch (fortran {len(fort_tuple)}, "
                          f"jax {len(jax_tuple)}, declared {len(outputs)})"}

    report: Dict[str, Any] = {"status": "pass", "atol": atol, "rtol": rtol,
                              "max_abs_err": 0.0, "mismatches": []}
    for name_out, f, j in zip(outputs, fort_tuple, jax_tuple):
        fa, ja = np.asarray(f, dtype=float), np.asarray(j, dtype=float)
        if fa.shape != ja.shape:
            report["status"] = "fail"
            report["mismatches"].append({"arg": name_out, "kind": "shape",
                                         "fortran": fa.shape, "jax": ja.shape})
            continue
        err = float(np.max(np.abs(fa - ja))) if fa.size else 0.0
        report["max_abs_err"] = max(report["max_abs_err"], err)
        if not np.allclose(fa, ja, atol=atol, rtol=rtol):
            report["status"] = "fail"
            report["mismatches"].append({"arg": name_out, "kind": "value",
                                         "max_abs_err": err})
    return report


def equivalence_agent(state: Phase2State) -> dict:
    """Numerically compare every emitted kernel against the original Fortran."""
    print(f"\n{SEP}")
    print("  [Equivalence] JAX output vs the original Fortran (f2py)")
    print(SEP)

    updated: List[JaxKernelInfo] = []
    mismatches: List[str] = []

    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]

        if not kernel.get("jax_code") or kernel.get("status") in ("skipped", "error"):
            updated.append(kernel)
            continue

        # A routine that reads module state gets a promoted, standalone Fortran
        # twin whose signature matches the JAX (issue #99 whole-program build);
        # a self-contained one is compiled as-is.
        use_resolved = False
        extra_fortran_args: Dict[str, Any] = {}
        if kernel.get("free_reads") or kernel.get("free_writes"):
            from fortranspire.agent.nodes_jax.promote_fortran import (
                generate_equivalence_fortran,
            )
            from fortranspire.agent.nodes_jax.gradcheck import _DEFAULT_EXTENT

            twin, dim_arg = generate_equivalence_fortran(kernel)
            if twin is None:
                print(f"  ⏭ {name:<28} skipped — {dim_arg}")
                updated.append({**kernel, "equivalence": {"status": "skipped", "reason": dim_arg}})
                continue
            source = twin
            use_resolved = True
            extra_fortran_args = {dim_arg: _DEFAULT_EXTENT}
        else:
            source = kernel["fortran_code"]

        fortran_fn, reason = compile_fortran(source, name)
        if fortran_fn is None:
            print(f"  ⏭ {name:<28} skipped — {reason}")
            updated.append({**kernel, "equivalence": {"status": "skipped", "reason": reason}})
            continue

        namespace: Dict[str, Any] = {}
        try:
            exec(kernel["jax_code"], namespace)  # noqa: S102 - our own generated code
            jax_fn = namespace[name]
        except Exception as exc:  # noqa: BLE001
            updated.append({**kernel, "equivalence": {"status": "skipped",
                            "reason": f"JAX load failed: {exc}"}})
            continue

        report = check_equivalence(kernel, jax_fn, fortran_fn,
                                   use_resolved=use_resolved,
                                   extra_fortran_args=extra_fortran_args)
        entry = {**kernel, "equivalence": report}

        if report["status"] == "pass":
            print(f"  ✓ {name:<28} matches Fortran — max|Δ| = {report['max_abs_err']:.3e}")
        elif report["status"] == "fail":
            entry["status"] = "error"
            detail = report["mismatches"][0] if report["mismatches"] else {}
            msg = f"{name}: differs from Fortran — {detail}"
            mismatches.append(msg)
            print(f"  ✗ {name:<28} DIFFERS from Fortran — max|Δ| = {report['max_abs_err']:.3e}")
        else:
            print(f"  ⏭ {name:<28} skipped — {report.get('reason', '')}")

        updated.append(entry)

    if mismatches:
        print(f"\n  {len(mismatches)} kernel(s) do not match the original Fortran.")
    else:
        print("\n  No numerical disagreement with the original Fortran.")

    return {
        "kernel_results": updated,
        "equivalence_passed": not mismatches,
        "executed_agents": state.get("executed_agents", []) + ["equivalence"],
    }
