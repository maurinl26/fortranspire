"""Generate a standalone, f2py-compilable Fortran twin of a module-state kernel.

The Phase 2 equivalence check skips a routine that reads its inputs through
`USE` — it cannot be compiled in isolation. This lifts that: it rewrites the
routine into a self-contained subroutine whose module state is passed as
explicit arguments (exactly the promotion #5 does for JAX), so its signature
lines up with the emitted JAX and the two can be run on one shared fixture.

The transform, deterministic, driven by #5 (`free_reads`/`free_writes`) and #99
resolution (dtype + rank of each promoted symbol):

* drop the `USE` lines;
* append the promoted reads (INTENT(IN)), writes (INTENT(INOUT)) and one integer
  extent `nfs` to the argument list, and declare them — arrays as assumed-shape;
* make every INTENT(OUT) array's shape explicit in `nfs` (f2py cannot infer an
  assumed-shape output; IN/INOUT get theirs from the passed array). Every axis is
  `nfs`, which matches the degenerate all-`nfs` fixture the gradcheck/equivalence
  inputs already use.

It refuses (returns a reason, never a wrong twin) when a promoted symbol is
unresolved, or the body `CALL`s an external procedure — that cannot be linked
standalone (CMAQ RBFEVAL's `SPECIAL_RATES` is this case), and is the honest
boundary of a single-file build.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_DIM_ARG = "nfs"  # integer-implicit initial letter — f2py wrappers need this

_FTYPE = {
    "integer": "INTEGER",
    "real": "REAL(KIND=8)",
    "logical": "LOGICAL",
    "complex": "COMPLEX(KIND=8)",
}


def _shape(rank: int) -> str:
    return "" if rank <= 0 else "(" + ", ".join([":"] * rank) + ")"


def generate_equivalence_fortran(kernel: dict) -> Tuple[Optional[str], str]:
    """Return ``(standalone_fortran, "")`` or ``(None, reason)``."""
    reads = list(kernel.get("free_reads") or [])
    writes = list(kernel.get("free_writes") or [])
    if not reads and not writes:
        return None, "self-contained"

    resolved = {k.lower(): v for k, v in (kernel.get("resolved") or {}).items()}
    missing = [s for s in reads + writes if s.lower() not in resolved]
    if missing:
        return None, f"unresolved shapes: {', '.join(missing)}"

    code = kernel["fortran_code"]
    name = kernel["routine_name"]

    calls = sorted({c for c in re.findall(r"\bCALL\s+(\w+)", code, re.IGNORECASE)})
    if calls:
        return None, f"calls external procedure(s): {', '.join(calls)} (not linkable standalone)"

    # 1. Drop USE lines.
    lines = [ln for ln in code.splitlines()
             if not re.match(r"\s*USE\s+\w", ln, re.IGNORECASE)]

    # 2. Make every INTENT(OUT) array explicit in nfs (assumed-shape OUT is
    #    unresolvable for f2py; IN/INOUT keep their shape from the input).
    def _explicit_out(ln: str) -> str:
        m = re.search(r"INTENT\s*\(\s*OUT\s*\)\s*::\s*(\w+)\s*\(([^)]*)\)",
                      ln, re.IGNORECASE)
        if not m:
            return ln
        rank = m.group(2).count(",") + 1
        return ln[:m.start(2)] + ", ".join([_DIM_ARG] * rank) + ln[m.end(2):]

    lines = [_explicit_out(ln) for ln in lines]
    body = "\n".join(lines)

    # 3. Append promoted symbols + nfs to the argument list.
    extra = reads + writes + [_DIM_ARG]

    def _augment(m: re.Match) -> str:
        existing = m.group(2).strip()
        joined = (existing + ", " if existing else "") + ", ".join(extra)
        return f"{m.group(1)}({joined})"

    body, n = re.subn(rf"(SUBROUTINE\s+{re.escape(name)})\s*\(([^)]*)\)",
                      _augment, body, count=1, flags=re.IGNORECASE)
    if n == 0:
        return None, "could not locate the SUBROUTINE argument list"

    # 4. Declare the promoted symbols and nfs, right after IMPLICIT NONE.
    decls = []
    for r in reads:
        rv = resolved[r.lower()]
        decls.append(f"      {_FTYPE.get(rv['dtype'], 'REAL(KIND=8)')}, "
                     f"INTENT(IN) :: {r}{_shape(rv['rank'])}")
    for w in writes:
        rv = resolved[w.lower()]
        decls.append(f"      {_FTYPE.get(rv['dtype'], 'REAL(KIND=8)')}, "
                     f"INTENT(INOUT) :: {w}{_shape(rv['rank'])}")
    decls.append(f"      INTEGER, INTENT(IN) :: {_DIM_ARG}")
    block = "\n".join(decls)

    body, n = re.subn(r"(IMPLICIT\s+NONE)", r"\1\n" + block.replace("\\", "\\\\"),
                      body, count=1, flags=re.IGNORECASE)
    if n == 0:  # no IMPLICIT NONE — put the block after the (augmented) header
        body = re.sub(rf"(SUBROUTINE\s+{re.escape(name)}\s*\([^)]*\)\s*\n)",
                      r"\1  implicit none\n" + block + "\n", body, count=1,
                      flags=re.IGNORECASE)

    return body, _DIM_ARG
