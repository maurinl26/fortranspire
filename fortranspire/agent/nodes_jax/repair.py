"""Self-repair for JAX emission — tolerate a small model (e.g. Mistral Large).

A capable model follows the sharpened prompt and never emits a Python `if` on
traced array data; a smaller one still slips (Mistral Large did on RBFEVAL).
Rather than depend on model capability, the pipeline checks the emitted code
against **deterministic** defects and, when it finds one, hands the model back
the *exact* error for one targeted retry. The retry is cheap (only when a defect
is present) and bounded, and it turns the lint from a diagnostic into a fix.

Only deterministic, unambiguous defects are repaired here — a Python syntax
error, or a branch on array-derived data (the `data_dependent_branches` lint).
Numerical wrongness is not in scope: that is what gradcheck and the equivalence
check are for, and it cannot be phrased as a mechanical instruction.
"""
from __future__ import annotations

import ast
from typing import Callable, List, Tuple

from fortranspire.agent.nodes_jax.lint import (
    data_dependent_branches,
    data_dependent_loops,
    disallowed_imports,
    missing_imports,
)


def emission_defects(code: str) -> List[str]:
    """Deterministic, mechanically-fixable defects in emitted JAX. Empty = clean."""
    defects: List[str] = []
    try:
        ast.parse(code)
    except SyntaxError as exc:
        # Unparseable code can't be linted further — report just this.
        return [f"Python syntax error at line {exc.lineno}: {exc.msg}"]

    for miss in missing_imports(code):
        fix = {"jax": "import jax", "jnp": "import jax.numpy as jnp",
               "np": "import numpy as np", "numpy": "import numpy",
               "lax": "from jax import lax", "scipy": "import jax.scipy"}.get(
                   miss["name"], f"import {miss['name']}")
        defects.append(
            f"uses `{miss['name']}.…` but never imports it — add `{fix}` at the top."
        )
    for imp in disallowed_imports(code):
        defects.append(
            f"line {imp['line']}: `import {imp['module']}` — this module does not "
            "exist in Python (it is a Fortran `USE`). Its symbols are already "
            "arguments of the function; remove the import and reference them "
            "directly. Import only jax / jax.numpy / fortranspire.jax_smooth."
        )
    for b in data_dependent_branches(code):
        defects.append(
            f"line {b['line']}: `if {b['snippet']}` branches on a value read from "
            "an array, which raises TracerBoolConversionError under `jit`. Rewrite "
            "it with `jnp.where` (two-way) or `jax.lax.switch` (multi-way) — never "
            "a Python `if` on array data."
        )
    for lp in data_dependent_loops(code):
        defects.append(
            f"line {lp['line']}: `for ... in {lp['snippet']}` loops over an "
            "array-derived bound, which is not a concrete trip count under `jit`. "
            "Vectorise over a fixed extent and mask the inactive entries with "
            "`jnp.where`, or use `jax.lax.fori_loop` with a static bound."
        )
    return defects


def _repair_message(defects: List[str]) -> str:
    body = "\n".join(f"- {d}" for d in defects)
    return (
        "The module you emitted has defects that make it fail under `jit`:\n"
        f"{body}\n\n"
        "Emit the corrected module — same signature, same numerics, fixing "
        "exactly these defects and nothing else. Output only the Python module, "
        "no prose."
    )


def emit_with_repair(
    llm,
    system,
    *,
    strip: Callable[[str], str],
    max_repairs: int = 2,
    human: str = "Emit the module.",
    log: Callable[[str], None] = lambda _msg: None,
) -> Tuple[str, List[str], int]:
    """Emit, then repair any deterministic defect by re-prompting with it.

    Returns ``(code, remaining_defects, n_repairs)``. ``remaining_defects`` is
    empty when the code came back clean (possibly after a repair); non-empty
    means the model could not fix it within ``max_repairs`` — the caller keeps
    the code and lets gradcheck/lint report it.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [system, HumanMessage(content=human)]
    response = llm.invoke(messages)
    code = strip(getattr(response, "content", str(response)))

    repairs = 0
    for _ in range(max(0, max_repairs)):
        defects = emission_defects(code)
        if not defects:
            break
        log(f"    ↻ repairing {len(defects)} emission defect(s) (attempt {repairs + 1})")
        messages = messages + [AIMessage(content=code),
                               HumanMessage(content=_repair_message(defects))]
        response = llm.invoke(messages)
        code = strip(getattr(response, "content", str(response)))
        repairs += 1

    return code, emission_defects(code), repairs
