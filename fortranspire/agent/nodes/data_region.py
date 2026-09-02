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
    order = ("copyin", "copyout", "copy", "create")
    if gpu_pragma == "acc":
        parts = [f"{kind}({', '.join(clauses[kind])})"
                 for kind in order if clauses.get(kind)]
        clause_str = (" " + " ".join(parts)) if parts else ""
        return f"!$acc data{clause_str}", "!$acc end data"

    # OpenMP target: copyin→map(to:), copyout→map(from:), copy→map(tofrom:),
    # create→map(alloc:).
    omp = {"copyin": "to", "copyout": "from", "copy": "tofrom", "create": "alloc"}
    parts = [f"map({omp[kind]}: {', '.join(clauses[kind])})"
             for kind in order if clauses.get(kind)]
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


def find_region_bounds(driver_src: str, kernel_names: set):
    """`(do_idx, end_idx)` of the outermost loop holding a kernel call, or None."""
    lines = driver_src.splitlines()
    call_lines = [i for i, ln in enumerate(lines)
                  if any(re.search(rf"\bCALL\s+{re.escape(n)}\b", ln, re.IGNORECASE)
                         for n in kernel_names)]
    if not call_lines:
        return None
    do_idx = None
    for i in range(call_lines[0], -1, -1):
        if re.match(r"\s*do\b", lines[i], re.IGNORECASE):
            do_idx = i
            break
    if do_idx is None:
        return None
    depth = 0
    for i in range(do_idx, len(lines)):
        if re.match(r"\s*do\b", lines[i], re.IGNORECASE):
            depth += 1
        elif re.match(r"\s*end\s*do\b", lines[i], re.IGNORECASE):
            depth -= 1
            if depth == 0:
                return do_idx, i
    return None


def array_actuals(kernel_calls: List[Tuple[str, List[str]]],
                  kernels: Dict[str, dict]) -> set:
    """The actual arguments that are *arrays* (need a data clause)."""
    arrays: set = set()
    for kname, actuals in kernel_calls:
        k = kernels.get(kname.lower())
        if not k:
            continue
        formals = list((k.get("intent_map") or {}).keys())
        dims = k.get("dimensions") or {}
        for i, actual in enumerate(actuals):
            if i < len(formals) and dims.get(formals[i]):
                arrays.add(actual)
    return arrays


_DECL_RE = re.compile(
    r"\s*(real|integer|complex|logical|double\s+precision|character|type\s*\(|"
    r"dimension|implicit|program|subroutine|function|end\b|use\b)",
    re.IGNORECASE,
)


def analyse_liveness(driver_src: str, arrays: set,
                     do_idx: int, end_idx: int) -> Dict[str, Tuple[bool, bool]]:
    """For each array: `(live_in, live_out)` w.r.t. the data region.

    ``live_in``  — the host references it on an executable statement *before* the
    loop (its value may be needed on the GPU → ``copyin``); ``live_out`` — *after*
    the loop (the GPU result may be needed → ``copyout``). Declarations excluded.

    Conservative: any mention outside the region counts, so a transfer is never
    dropped. The only downgrade is to ``create`` when the array appears *nowhere*
    outside the loop — provably a loop-local temporary.
    """
    lines = driver_src.splitlines()
    before = [ln for ln in lines[:do_idx] if not _DECL_RE.match(ln)]
    after = [ln for ln in lines[end_idx + 1:] if not _DECL_RE.match(ln)]
    out: Dict[str, Tuple[bool, bool]] = {}
    for a in arrays:
        pat = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
        out[a] = (any(pat.search(ln) for ln in before),
                  any(pat.search(ln) for ln in after))
    return out


def clauses_from_liveness(liveness: Dict[str, Tuple[bool, bool]]) -> Dict[str, List[str]]:
    """Turn (live_in, live_out) into the optimal clause per array."""
    clauses: Dict[str, List[str]] = {"copyin": [], "copyout": [], "copy": [], "create": []}
    for a, (li, lo) in sorted(liveness.items()):
        if li and lo:
            clauses["copy"].append(a)
        elif li:
            clauses["copyin"].append(a)
        elif lo:
            clauses["copyout"].append(a)
        else:                                   # never used on the host → GPU-only
            clauses["create"].append(a)
    return clauses


def insert_data_region(driver_src: str, open_pragma: str, close_pragma: str,
                       kernel_names: set) -> str:
    """Wrap the outermost loop that contains a kernel call with the data region."""
    bounds = find_region_bounds(driver_src, kernel_names)
    if bounds is None:
        return driver_src
    do_idx, end_idx = bounds
    lines = driver_src.splitlines()
    indent = re.match(r"(\s*)", lines[do_idx]).group(1)
    out = lines[:do_idx] + [f"{indent}{open_pragma}"] + lines[do_idx:end_idx + 1] \
        + [f"{indent}{close_pragma}"] + lines[end_idx + 1:]
    return "\n".join(out) + ("\n" if driver_src.endswith("\n") else "")
