"""GT4Py type-check — verify each emitted field operator is well-typed (#42).

The gt4py analogue of the JAX `gradcheck` node. Accessing a field
operator's ``.foast_stage`` runs gt4py.next's own frontend type checker:
it verifies the body against the annotated signature (field dimensions,
dtypes, that a scalar is not returned where a field is declared, that only
DSL-legal constructs appear) and raises a ``DSLError`` when the operator is
malformed. No execution, no offset providers, no backend compile needed.

**What this does NOT check — on purpose.** It validates the *operator*, not
the *domain*. In gt4py.next the geometry lives in the driver, not the field
operator, and it is a separate, harder problem:

* the ``domain=`` a ``program`` writes (which range of each dimension);
* the **offset providers** — a ``CartesianConnectivity`` for a structured
  shift, an ``as_connectivity`` neighbour table for an unstructured mesh;
* **halos / boundaries** — a stencil that reads ``a(Koff[1])`` at the top
  layer reads out of bounds unless the domain is restricted to the
  interior or the field carries halo points.

None of that is visible to the frontend type check. Proof from the field:
the ``vertical_avg`` example type-checks cleanly and then fails at *call*
time on the offset provider — the domain is a runtime concern. Validating
it (the way icon4py and Pace frame domain and halo correctness) is tracked
separately in issue #82; this node deliberately stops at "is it a
well-typed operator".

Two facts about gt4py.next drive the design, both learned by running it:

* **A field operator must live in a real ``.py`` file.** gt4py reads the
  operator's source with ``inspect.getsourcelines``, so ``exec``-ing the
  generated text (as the JAX path does) raises "could not get source
  code". The generated module is written to a temp file and imported.
* gt4py depends on a ``dace`` pre-release, which the project's locked
  resolver will not pull, so gt4py is **not** a declared dependency. When
  it is absent the node degrades to ``skipped`` — like the equivalence
  harness without ``gfortran`` — rather than failing.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fortranspire.agent.nodes._common import SEP


def _gt4py_available() -> bool:
    return importlib.util.find_spec("gt4py") is not None


def _load_module_from_source(code: str, name: str) -> Any:
    """Write `code` to a real .py file and import it — gt4py needs the source."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="gt4py-validate-"))
    module_path = tmp_dir / f"{name}.py"
    module_path.write_text(code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Keep the file around while the module lives — gt4py re-reads the
    # source lazily (on `.foast_stage`), so deleting it now would break the
    # type-check. It is a temp file; the OS reclaims it.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _field_operators(module: Any) -> list[tuple[str, Any]]:
    """Every gt4py field/scan operator defined at module top level."""
    found: list[tuple[str, Any]] = []
    for attr in vars(module).values():
        # A field operator carries `.foast_stage`; a scan operator too.
        if hasattr(attr, "foast_stage") and hasattr(attr, "definition_stage"):
            name = getattr(getattr(attr, "definition", None), "__name__", None) \
                or getattr(attr, "__name__", "?")
            found.append((name, attr))
    return found


def type_check_source(code: str, *, module_name: str = "gt4py_kernel") -> Dict[str, Any]:
    """Type-check every operator in `code` against gt4py's frontend.

    Returns a report; never raises for a malformed operator — the caller
    decides whether a failure is blocking.
    """
    report: Dict[str, Any] = {
        "status": "pass", "checked": [], "failures": [], "gt4py": True,
    }

    if not _gt4py_available():
        report["status"] = "skipped"
        report["gt4py"] = False
        report["reason"] = (
            "gt4py is not installed — install it (`pip install gt4py`) to "
            "type-check the emitted field operators."
        )
        return report

    try:
        module = _load_module_from_source(code, module_name)
    except Exception as exc:  # noqa: BLE001 - a broken module is a failure
        report["status"] = "fail"
        report["failures"].append(
            {"operator": "<module>", "kind": "import",
             "detail": f"{type(exc).__name__}: {exc}"}
        )
        return report

    operators = _field_operators(module)
    if not operators:
        report["status"] = "fail"
        report["failures"].append(
            {"operator": "<module>", "kind": "empty",
             "detail": "no @field_operator or @scan_operator found in the emitted module"}
        )
        return report

    for name, operator in operators:
        try:
            # Force the frontend type-check without executing anything.
            _ = operator.foast_stage
            report["checked"].append(name)
        except Exception as exc:  # noqa: BLE001 - DSLError and friends
            report["status"] = "fail"
            report["failures"].append(
                {"operator": name, "kind": "type-check",
                 "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}
            )

    return report


def type_check_agent(state) -> dict:
    """Type-check every emitted gt4py operator. Blocking when gt4py is present."""
    print(f"\n{SEP}")
    print("  [GT4Py type-check] frontend type-check (operator, not domain/halos)")
    print(SEP)

    kernels = state.get("kernel_results", [])
    updated: List[dict] = []
    failures: List[str] = []
    logs: List[str] = []
    any_checked = False

    for kernel in kernels:
        name = kernel.get("routine_name", "?")
        code = kernel.get("gt4py_code", "")

        if kernel.get("status") == "skipped" or not code:
            print(f"  ⏭ {name:<28} nothing emitted")
            updated.append(kernel)
            continue

        report = type_check_source(code, module_name=f"gt4py_{name}")
        entry = {**kernel, "domain_check": report}

        if report["status"] == "skipped":
            print(f"  ⏭ {name:<28} skipped — gt4py not installed")
            updated.append(entry)
            continue

        any_checked = True
        if report["status"] == "fail":
            entry["status"] = "error"
            detail = report["failures"][0] if report["failures"] else {}
            msg = f"{name}: {detail.get('kind')} — {detail.get('detail', '')}"
            failures.append(msg)
            logs.append(msg)
            print(f"  ✗ {name:<28} FAIL — {detail.get('kind')}")
        else:
            entry["status"] = "success"
            print(f"  ✓ {name:<28} type-checks ({len(report['checked'])} operator(s))")

        updated.append(entry)

    if not _gt4py_available():
        print("\n  gt4py not installed — operators emitted but not type-checked.")
    elif failures:
        print(f"\n  {len(failures)} operator(s) failed the gt4py type-check — blocking.")
    elif any_checked:
        print("\n  All emitted operators type-check against gt4py.next.")

    return {
        "kernel_results": updated,
        # A skip (no gt4py) is not a pass and not a failure — the caller must
        # not read absence of failures as validation.
        "type_checked": bool(any_checked) and not failures,
        "type_check_skipped": not _gt4py_available(),
        "type_check_log": "\n".join(logs),
        "executed_agents": state.get("executed_agents", []) + ["type_check"],
    }
