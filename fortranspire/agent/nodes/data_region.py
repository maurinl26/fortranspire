"""Deterministic OpenACC/OpenMP driver data region — derived from INTENT.

The driver data region (`!$acc data copyin/copyout/copy(...)` around the time
loop) decides what crosses the CPU↔GPU boundary. A wrong clause is not a compile
error — it is **silent corruption**: a `copyin` for a result array ships stale
data back, a missing `copyout` drops the answer. The previous node asked an LLM,
with a prompt that did not even carry the INTENT of the arrays.

But it is decidable: for each array a kernel is called with inside the loop, its
role is its formal argument's INTENT. Aggregated across the calls:
  * read everywhere              → `copyin`
  * written everywhere           → `copyout`
  * read-and-written, or mixed   → `copy`      (always correct)
Scalars need no data clause. This derives the clauses — correct by construction,
and also optimal (no needless transfer), with no guess.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple


def derive_data_clauses(
    kernel_calls: List[Tuple[str, List[str]]],
    kernels: Dict[str, dict],
) -> Dict[str, List[str]]:
    """Aggregate array roles across the kernel calls into copyin/copyout/copy.

    ``kernel_calls``: ``[(kernel_name, [actual_arg, …]), …]`` in call order.
    ``kernels``: ``{name: {"intent_map": {...}, "dimensions": {...}}}``.
    """
    roles: Dict[str, set] = {}
    for kname, actuals in kernel_calls:
        k = kernels.get(kname.lower())
        if not k:
            continue
        intent_map = k.get("intent_map") or {}
        dims = k.get("dimensions") or {}
        formals = list(intent_map.keys())
        for i, actual in enumerate(actuals):
            if i >= len(formals):
                continue
            formal = formals[i]
            if not dims.get(formal):          # only arrays need a data clause
                continue
            roles.setdefault(actual, set()).add((intent_map.get(formal) or "IN").upper())

    copyin, copyout, copy = [], [], []
    for arr, intents in sorted(roles.items()):
        if intents == {"IN"}:
            copyin.append(arr)
        elif intents == {"OUT"}:
            copyout.append(arr)
        else:                                  # INOUT, or IN+OUT across kernels
            copy.append(arr)
    return {"copyin": copyin, "copyout": copyout, "copy": copy}


def render_data_pragma(clauses: Dict[str, List[str]], gpu_pragma: str = "acc") -> Tuple[str, str]:
    """The `data`/`end data` directive pair for the derived clauses."""
    if gpu_pragma == "acc":
        parts = [f"{kind}({', '.join(clauses[kind])})"
                 for kind in ("copyin", "copyout", "copy") if clauses.get(kind)]
        clause_str = (" " + " ".join(parts)) if parts else ""
        return f"!$acc data{clause_str}", "!$acc end data"

    # OpenMP target: copyin→map(to:), copyout→map(from:), copy→map(tofrom:)
    omp = {"copyin": "to", "copyout": "from", "copy": "tofrom"}
    parts = [f"map({omp[kind]}: {', '.join(clauses[kind])})"
             for kind in ("copyin", "copyout", "copy") if clauses.get(kind)]
    clause_str = (" " + " ".join(parts)) if parts else ""
    return f"!$omp target data{clause_str}", "!$omp end target data"


_CALL_RE = re.compile(r"\bCALL\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE)


def extract_kernel_calls(driver_src: str, kernel_names: set) -> List[Tuple[str, List[str]]]:
    """Ordered `(kernel, [actual args])` for every CALL to a known kernel."""
    calls: List[Tuple[str, List[str]]] = []
    for m in _CALL_RE.finditer(driver_src):
        name = m.group(1)
        if name.lower() in {n.lower() for n in kernel_names}:
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            calls.append((name, args))
    return calls


def insert_data_region(driver_src: str, open_pragma: str, close_pragma: str,
                       kernel_names: set) -> str:
    """Wrap the outermost loop that contains a kernel call with the data region."""
    lines = driver_src.splitlines()
    # Find the first `do` whose loop body contains a CALL to a kernel, and its
    # matching `end do`. Conservative and indentation-based.
    call_lines = [i for i, ln in enumerate(lines)
                  if any(re.search(rf"\bCALL\s+{re.escape(n)}\b", ln, re.IGNORECASE)
                         for n in kernel_names)]
    if not call_lines:
        return driver_src
    first_call = call_lines[0]

    do_idx = None
    for i in range(first_call, -1, -1):
        if re.match(r"\s*do\b", lines[i], re.IGNORECASE):
            do_idx = i
            break
    if do_idx is None:
        return driver_src
    indent = re.match(r"(\s*)", lines[do_idx]).group(1)

    depth = 0
    end_idx = None
    for i in range(do_idx, len(lines)):
        if re.match(r"\s*do\b", lines[i], re.IGNORECASE):
            depth += 1
        elif re.match(r"\s*end\s*do\b", lines[i], re.IGNORECASE):
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        return driver_src

    out = lines[:do_idx] + [f"{indent}{open_pragma}"] + lines[do_idx:end_idx + 1] \
        + [f"{indent}{close_pragma}"] + lines[end_idx + 1:]
    return "\n".join(out) + ("\n" if driver_src.endswith("\n") else "")
