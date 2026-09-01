"""Phase 2, node 5 — prove the emitted kernel is actually differentiable.

The Phase 2 validation that shipped before this node checked syntax,
bytecode, ``exec`` and ``make_jaxpr``. All four together prove one thing:
**the code traces**. None of them proves the gradients are right — and
differentiability is the entire reason to target JAX rather than OpenACC.
A kernel that traces but whose gradient is wrong is worse than a broken
one: it is silently wrong exactly where the caller relies on it.

This node compares ``jax.grad`` against central finite differences of the
same function. What that catches, verified against each case:

* a ``jnp.where`` whose *untaken* branch evaluates to NaN or infinity —
  forward values look perfect, the gradient is NaN. This is the classic
  one and the most common in translated Fortran, where an ``IF`` guards a
  ``sqrt`` or a division;
* a gradient that is silently detached — ``stop_gradient``, or an
  in-place update translated as mutation — wherever the true derivative
  is non-zero;
* a kernel that traces but raises under ``jax.grad``;
* a kernel that survives ``grad`` but cannot be ``jit``-ed. Outside
  ``jit``, reverse-mode traces with a tracer carrying a concrete primal,
  so a Python ``if`` on a traced value silently resolves against it and
  the gradient looks correct — then the kernel raises the first time
  anyone jits it. Checking ``grad`` alone green-lights that.

**What it cannot catch, by construction.** Finite differences see the same
function the autodiff does, so a transformation that makes the function
*locally flat* — ``floor``, ``round``, an integer cast — yields zero from
both sides and agrees. Verified: a ``floor``-based kernel passes this
check while being non-differentiable in any useful sense. Detecting that
class needs either a comparison against the original Fortran (the second
level below) or a static rule flagging the construct; do not read a pass
here as a proof of differentiability in general.

Finite differences are noisy, so two things are non-negotiable: float64
must be on (float32 noise swamps the signal entirely), and only a handful
of random entries are probed per array — a full Jacobian would cost one
evaluation per element.

Cross-checking the gradient against the *original Fortran* is stronger
still and is the natural extension; it needs the f2py build the Phase 1
equivalence harness produces, so it is wired as an optional second level
rather than assumed.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from fortranspire.agent.nodes._common import SEP
from fortranspire.agent.nodes_jax._state import JaxKernelInfo, Phase2State

# Central differences are second-order accurate: the error goes as h^2
# plus round-off/h. In float64 the sweet spot sits near 1e-5.
_DEFAULT_STEP = float(os.getenv("FORTRANSPIRE_GRADCHECK_STEP", "1e-5"))
_DEFAULT_RTOL = float(os.getenv("FORTRANSPIRE_GRADCHECK_RTOL", "1e-4"))
_DEFAULT_ATOL = float(os.getenv("FORTRANSPIRE_GRADCHECK_ATOL", "1e-6"))
# Entries probed per differentiable argument.
_DEFAULT_PROBES = int(os.getenv("FORTRANSPIRE_GRADCHECK_PROBES", "4"))
# Fallback extent for an array whose declared dimension is symbolic (`n`).
_DEFAULT_EXTENT = int(os.getenv("FORTRANSPIRE_GRADCHECK_EXTENT", "8"))


class GradcheckError(RuntimeError):
    """The emitted kernel is not differentiable, or its gradient is wrong."""


def _enable_x64() -> None:
    """float32 finite differences are pure noise — refuse to run without x64."""
    import jax

    jax.config.update("jax_enable_x64", True)


def _extent(dim: Any) -> int:
    """Turn a declared Fortran dimension into a concrete probe extent."""
    text = str(dim).strip()
    # `1:n`, `n`, `0:nx-1` … only a literal upper bound is usable directly.
    if ":" in text:
        text = text.split(":")[-1].strip()
    try:
        value = int(text)
    except ValueError:
        return _DEFAULT_EXTENT
    return max(1, min(value, 64))


def _make_inputs(kernel: JaxKernelInfo, seed: int = 0) -> Dict[str, Any]:
    """Build a random input for each argument, from the declared dimensions."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    dims = kernel.get("dimensions") or {}
    dtypes = {k.lower(): v for k, v in (kernel.get("arg_dtypes") or {}).items()}
    values: Dict[str, Any] = {}

    for arg in kernel["inputs"]:
        shape = [_extent(d) for d in dims.get(arg, [])]
        dt = dtypes.get(arg.lower(), "unknown")

        if dt == "integer":
            # Type-driven, not a name guess (the old whitelist missed
            # `numcells`, `nspecial_rxn`, … and fed them floats — issue #4).
            # A scalar integer is a bound → the probe extent; an integer array
            # is a lookup table → zeros, which are a valid in-range index for a
            # gather (index-topology kernels are caught earlier as needs_fixture).
            if shape:
                values[arg] = jnp.zeros(shape, dtype=jnp.int32)
            else:
                values[arg] = _DEFAULT_EXTENT
        elif shape:
            values[arg] = jnp.asarray(rng.standard_normal(shape))
        else:
            # real / unknown scalar → a float, the differentiable default for
            # the numeric payload. The legacy extent names stay a safety net for
            # kernels the dtype pass could not type (regex-frontend fallbacks).
            if dt == "unknown" and arg.lower() in {"n", "m", "nx", "ny", "nz", "nt", "npts"}:
                values[arg] = _DEFAULT_EXTENT
            else:
                values[arg] = float(rng.standard_normal())

    return values


def _scalarise(fn: Callable, dyn_names: List[str], static_vals: Dict[str, Any]) -> Callable:
    """Reduce the kernel's outputs to one scalar so `jax.grad` is defined.

    ``static_vals`` (integer bounds/flags) are bound as Python constants, not
    passed through ``jit`` — a size fed as a traced value breaks
    ``jnp.arange(n)`` / ``reshape`` even when it is correctly an integer, so it
    must be static, which is exactly how these kernels are called in practice.
    """
    import jax.numpy as jnp

    def scalar(**dyn):
        out = fn(**{**static_vals, **{k: dyn[k] for k in dyn_names}})
        parts = out if isinstance(out, tuple) else (out,)
        return sum(jnp.sum(jnp.asarray(p)) for p in parts)

    return scalar


def _differentiable(value: Any) -> bool:
    """Only float arrays and float scalars carry a meaningful gradient."""
    import jax.numpy as jnp

    if isinstance(value, bool) or isinstance(value, int):
        return False
    try:
        return bool(jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating))
    except Exception:  # noqa: BLE001 - anything unconvertible is not differentiable
        return False


def check_kernel(
    fn: Callable,
    kernel: JaxKernelInfo,
    *,
    step: float = _DEFAULT_STEP,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    probes: int = _DEFAULT_PROBES,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare `jax.grad` with central finite differences on `fn`.

    Returns a report dict; never raises for a numerical mismatch — the
    caller decides whether a failure is blocking.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    _enable_x64()

    names = list(kernel["inputs"])

    # Honest boundary (issue #4): an integer *index* — a lookup table like
    # IRM2, or a scalar used as a subscript — must be a valid position into
    # another array whose shape is not known from this file. A random probe
    # would index out of range. Report it as a required fixture instead of
    # crashing or fabricating a topology; resolving the shapes to lift this is
    # the LFortran/whole-program step.
    index_args = [n for n in names
                  if n.lower() in {a.lower() for a in (kernel.get("index_args") or [])}]
    if index_args:
        return {
            "status": "needs_fixture",
            "jit": False,
            "checked_args": [],
            "skipped_args": names,
            "fixture_args": index_args,
            "reason": (
                "integer index topology (" + ", ".join(index_args) + ") cannot be "
                "synthesised from this file — the emitted kernel is well-formed but "
                "differentiability needs a valid index fixture (resolve array shapes "
                "cross-module to lift this)."
            ),
            "failures": [],
        }

    inputs = _make_inputs(kernel, seed=seed)

    # Integer bounds/flags are bound statically (see _scalarise); the rest are
    # the dynamic call, and only the float ones carry a gradient.
    static_names = [n for n in names if isinstance(inputs[n], int)
                    and not isinstance(inputs[n], bool)]
    static_vals = {n: inputs[n] for n in static_names}
    dyn_names = [n for n in names if n not in static_names]
    dyn_inputs = {n: inputs[n] for n in dyn_names}
    scalar = _scalarise(fn, dyn_names, static_vals)

    diffable = [n for n in dyn_names if _differentiable(inputs[n])]
    report: Dict[str, Any] = {
        "status": "pass",
        "jit": False,
        "checked_args": diffable,
        "skipped_args": [n for n in names if n not in diffable],
        "step": step, "rtol": rtol, "atol": atol,
        "max_abs_err": 0.0,
        "failures": [],
    }

    if not diffable:
        report["status"] = "skipped"
        report["reason"] = "no floating-point input to differentiate with respect to"
        return report

    # `jit` first, and it is not a formality. Outside `jit`, reverse-mode
    # traces with a JVPTracer that carries a *concrete* primal, so a Python
    # `if` on a traced value resolves against that primal and the gradient
    # comes out fine. The same kernel raises the moment anyone jits it —
    # which is the entire reason to be in JAX. Checking grad alone would
    # green-light a kernel that cannot be used.
    try:
        jax.jit(scalar)(**dyn_inputs).block_until_ready()
        report["jit"] = True
    except Exception as exc:  # noqa: BLE001
        report["jit"] = False
        report["status"] = "fail"
        report["failures"].append({
            "kind": "jit-failed",
            "detail": f"{type(exc).__name__}: {exc}. A Python `if`/`while` on a "
                      "traced value is the usual cause — use `jnp.where` or "
                      "`lax.cond` instead.",
        })
        return report

    # Differentiate one argument at a time: clearer failures, and it avoids
    # having to reorder the call signature.
    rng = np.random.default_rng(seed + 1)

    for name in diffable:
        base = inputs[name]

        def one_arg(x, _name=name):
            return scalar(**{**dyn_inputs, _name: x})

        try:
            analytic = jax.grad(one_arg)(base)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "fail"
            report["failures"].append(
                {"arg": name, "kind": "grad-raised", "detail": f"{type(exc).__name__}: {exc}"}
            )
            continue

        analytic = np.asarray(analytic)
        if not np.all(np.isfinite(analytic)):
            report["status"] = "fail"
            report["failures"].append(
                {"arg": name, "kind": "non-finite",
                 "detail": "gradient contains NaN or inf — typically a `where` whose "
                           "untaken branch is evaluated (guarded division)"}
            )
            continue

        flat = np.asarray(base, dtype=float).ravel()
        idx = rng.choice(flat.size, size=min(probes, flat.size), replace=False)

        for i in idx:
            plus = flat.copy(); plus[i] += step
            minus = flat.copy(); minus[i] -= step
            shape = np.asarray(base).shape

            f_plus = float(one_arg(jnp.asarray(plus.reshape(shape))))
            f_minus = float(one_arg(jnp.asarray(minus.reshape(shape))))
            numeric = (f_plus - f_minus) / (2.0 * step)
            exact = float(analytic.ravel()[i])

            err = abs(numeric - exact)
            report["max_abs_err"] = max(report["max_abs_err"], err)
            if err > atol + rtol * abs(numeric):
                report["status"] = "fail"
                report["failures"].append({
                    "arg": name, "kind": "mismatch", "index": int(i),
                    "analytic": exact, "finite_difference": numeric, "abs_err": err,
                })

    return report


def gradcheck_agent(state: Phase2State) -> dict:
    """Blocking gradient validation for every emitted kernel."""
    print(f"\n{SEP}")
    print("  [Gradcheck] jax.grad vs central finite differences")
    print(SEP)

    updated: List[JaxKernelInfo] = []
    failures: List[str] = []
    unverified: List[str] = []
    logs: List[str] = []

    for kernel in state.get("kernel_results", []):
        name = kernel["routine_name"]

        if kernel.get("status") == "skipped" or not kernel.get("jax_code"):
            print(f"  ⏭ {name:<28} no emitted code — nothing to check")
            updated.append(kernel)
            continue

        namespace: Dict[str, Any] = {}
        try:
            exec(kernel["jax_code"], namespace)  # noqa: S102 - our own generated code
            fn = namespace[name]
        except Exception as exc:  # noqa: BLE001
            report = {"status": "fail", "failures": [
                {"kind": "load", "detail": f"{type(exc).__name__}: {exc}"}]}
        else:
            report = check_kernel(fn, kernel)

        entry = {**kernel, "gradcheck": report}

        if report["status"] == "fail":
            entry["status"] = "error"
            detail = report["failures"][0] if report["failures"] else {}
            msg = f"{name}: {detail.get('kind', 'fail')} — {detail.get('detail', '')}".strip()
            failures.append(msg)
            logs.append(msg)
            print(f"  ✗ {name:<28} FAIL — {detail.get('kind')}")
            for failure in report["failures"][:3]:
                print(f"      {failure}")
        elif report["status"] == "needs_fixture":
            entry["status"] = "needs_fixture"
            msg = f"{name}: {report.get('reason', '')}"
            unverified.append(msg)
            print(f"  ⚠ {name:<28} NEEDS FIXTURE — {', '.join(report.get('fixture_args', []))}")
            print(f"      {report.get('reason', '')}")
        elif report["status"] == "skipped":
            print(f"  ⏭ {name:<28} skipped — {report.get('reason', '')}")
        else:
            entry["status"] = "success"
            print(f"  ✓ {name:<28} max|Δ| = {report['max_abs_err']:.3e}")

        updated.append(entry)

    # A refuted gradient blocks; an unverified one (needs a fixture) does not —
    # the emission is well-formed, we just cannot synthesise a valid index
    # topology here. Report it honestly rather than claim "verified".
    passed = not failures
    if failures:
        print(f"\n  {len(failures)} kernel(s) failed the gradient check — blocking.")
    elif unverified:
        print(f"\n  {len(unverified)} kernel(s) emitted but UNVERIFIED — a valid "
              "index fixture is required to check the gradient.")
    else:
        print("\n  All emitted kernels are differentiable and their gradients agree.")

    return {
        "kernel_results": updated,
        "gradcheck_passed": passed,
        "gradcheck_unverified": unverified,
        "gradcheck_log": "\n".join(logs + unverified),
        "executed_agents": state.get("executed_agents", []) + ["gradcheck"],
    }
