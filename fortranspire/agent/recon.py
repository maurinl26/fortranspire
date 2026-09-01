"""`recon` — arrival triage for an unfamiliar Fortran repo. No LLM, no token.

You clone a legacy codebase (CMAQ is 525 `.F` files) and the first real
question is *where do I even point the porter?* Answering it by hand — grep for
compute loops, read files, guess which routine is a kernel and which is a
driver — is exactly the manual triage this command removes.

It is a graph problem, and Loki gives the graph. For every routine we derive,
deterministically from the AST:

* **role** — a *driver* (I/O, PROGRAM, orchestration) is not a porting target;
  a pure-compute *leaf* of the call graph is the prime one;
* **portability** — the JAX purity verdict (a pure function ports to JAX) and
  the gt4py.next field-operator score, reusing the pipeline's own scorers so
  recon never disagrees with what a real port would find;
* **state cost** — how much module state must be promoted to arguments (#5),
  a direct proxy for porting effort;
* **reuse / hotness** — call-graph fan-in, an Amdahl proxy: a leaf called from
  deep in the solver matters more than one called once.

These combine into a single ranked **worklist** — start here, this target
(JAX / gt4py.next), this cost, and *why* — emitted as a table for a human and
as JSON for the chat / vibe front-end to consume.
"""
from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from fortranspire.agent.nodes._common import collect_fortran_files

# JAX portability is the purity verdict: a pure function ports, a threaded one
# needs its state carried, a blocked one cannot. Same verdict the JAX target uses.
_JAX_SCORE = {"pure": 5, "threaded": 3, "blocked": 0}


@dataclass
class RoutineTarget:
    name: str
    file: str
    role: str = "kernel"          # "driver" | "kernel"
    target: str = "none"          # "jax" | "gt4py" | "none"
    purity: str = "pure"
    jax_score: int = 0
    gt4py_score: int = 0
    has_io: bool = False
    has_save: bool = False
    is_program: bool = False
    n_loops: int = 0
    n_free_reads: int = 0
    n_free_writes: int = 0
    callees: List[str] = field(default_factory=list)
    fan_in: int = 0
    is_leaf: bool = False
    is_entry: bool = False
    rank: float = 0.0
    reason: str = ""


def _analyze_file(path: str) -> tuple[list[dict], Optional[str]]:
    """Parse one file: rich per-routine kernel dicts + CALL targets.

    Reuses ``parser_phase1`` (purity/intent/io/save/free-state, all from the
    AST) and a Loki re-walk for the callees — the same two-pass shape the
    ``graph`` command already pays.
    """
    from fortranspire.agent.nodes.parser import parser_phase1

    abspath = str(Path(path).resolve())
    state = {
        "fortran_filepath": abspath,
        "fortran_code": Path(abspath).read_text(encoding="utf-8"),
        "ast_info": {}, "kernel_results": [], "schema": {},
        "is_program": False, "executed_agents": [],
    }
    buf = io.StringIO()
    # Silence both streams: parser_phase1 prints a traceback to stderr on the
    # data-only files (a module of PARAMETERs has no routines) — expected here,
    # not an error worth surfacing per file.
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            result = parser_phase1(state)
        except Exception as exc:  # noqa: BLE001
            return [], str(exc)

    kernels = result.get("kernel_results", [])
    is_program = bool(result.get("is_program"))

    # Callees per routine (best-effort; empty on REGEX-frontend bodies).
    callee_map: dict[str, list[str]] = {}
    try:
        from loki import FindNodes, Sourcefile
        from loki.ir.nodes import CallStatement

        source = Sourcefile.from_file(abspath)
        allr = list(source.routines)
        for m in (source.modules or []):
            for r in (m.subroutines or []):
                if r not in allr:
                    allr.append(r)
        for r in allr:
            calls: list[str] = []
            try:
                for c in FindNodes(CallStatement).visit(r.body):
                    n = str(getattr(c, "name", "")).strip()
                    if n and n not in calls:
                        calls.append(n)
            except Exception:  # noqa: BLE001
                pass
            callee_map[r.name.lower()] = calls
    except Exception:  # noqa: BLE001 - keep the rich data even if calls fail
        pass

    for k in kernels:
        k["_callees"] = callee_map.get(k["routine_name"].lower(), [])
        k["_is_program"] = is_program
        k["_file"] = abspath
    return kernels, None


def _purity(kernel: dict) -> str:
    """Post-promotion purity verdict (reuses the functionalize logic + #5)."""
    from fortranspire.agent.nodes_jax.functionalize import (
        _promote_free_state, _split_by_intent, _verdict,
    )

    inputs, outputs, carried = _split_by_intent(kernel.get("intent_map") or {})
    _, outputs, _, _ = _promote_free_state(kernel, inputs, outputs, carried)
    verdict, _reason = _verdict(kernel, outputs)
    return verdict


def _gt4py_score(kernel: dict) -> int:
    from fortranspire.agent.nodes_gt4py.portability import score_routine

    try:
        return int(score_routine(kernel, kernel.get("fortran_code") or "").score)
    except Exception:  # noqa: BLE001
        return 0


def survey(paths: Iterable[str]) -> List[RoutineTarget]:
    """Analyse every routine under ``paths`` and return a ranked worklist."""
    # 1. Per-file analysis.
    files: list[str] = []
    for raw in paths:
        p = Path(raw)
        files.extend(collect_fortran_files([p]) if p.is_dir() else [str(p)])

    kernels: list[dict] = []
    seen: set[str] = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        ks, err = _analyze_file(f)
        if err:
            continue
        kernels.extend(ks)

    # 2. Cross-file graph: definitions and fan-in.
    defined = {k["routine_name"].lower() for k in kernels}
    fan_in: dict[str, int] = {}
    for k in kernels:
        for callee in k.get("_callees", []):
            fan_in[callee.lower()] = fan_in.get(callee.lower(), 0) + 1

    # 3. Per-routine target scoring.
    targets: list[RoutineTarget] = []
    for k in kernels:
        name = k["routine_name"]
        purity = _purity(k)
        jax = _JAX_SCORE.get(purity, 0)
        g4 = _gt4py_score(k)
        callees = k.get("_callees", [])
        internal_calls = [c for c in callees if c.lower() in defined]
        is_leaf = not internal_calls
        fi = fan_in.get(name.lower(), 0)
        is_entry = fi == 0
        is_program = bool(k.get("_is_program"))
        has_io = bool(k.get("has_io"))
        is_driver = has_io or is_program

        t = RoutineTarget(
            name=name, file=k.get("_file", ""),
            role="driver" if is_driver else "kernel",
            purity=purity, jax_score=jax, gt4py_score=g4,
            has_io=has_io, has_save=bool(k.get("has_save")),
            is_program=is_program,
            n_loops=len(k.get("loops") or []),
            n_free_reads=len(k.get("free_reads") or []),
            n_free_writes=len(k.get("free_writes") or []),
            callees=callees, fan_in=fi, is_leaf=is_leaf, is_entry=is_entry,
        )
        t.target, t.rank, t.reason = _rank(t)
        targets.append(t)

    targets.sort(key=lambda t: t.rank, reverse=True)
    return targets


def _rank(t: RoutineTarget) -> tuple[str, float, str]:
    """Combine the signals into (target_dsl, score, human reason)."""
    bits: list[str] = []

    if t.role == "driver":
        why = "I/O in the body" if t.has_io else "PROGRAM / top-level driver"
        return "none", 0.0, f"driver ({why}) — orchestration, not a porting target"

    portability = max(t.jax_score, t.gt4py_score)
    if portability == 0:
        return "none", 0.0, f"{t.purity} — cannot become a pure function"

    target = "jax" if t.jax_score >= t.gt4py_score else "gt4py"
    bits.append(f"{'JAX' if target == 'jax' else 'gt4py.next'} score "
                f"{max(t.jax_score, t.gt4py_score)}/5")

    score = float(portability)
    if t.is_leaf:
        score += 2.0
        bits.append("call-graph leaf (pure compute)")
    if t.fan_in:
        score += min(t.fan_in, 3)
        bits.append(f"fan-in {t.fan_in} (reused/hot)")
    if t.n_free_reads:
        # Graded by promotion effort so lighter-state kernels rank higher and
        # ties break by cost. Capped so it never outweighs portability.
        score -= min(0.1 * t.n_free_reads, 2.5)
        heavy = " (heavy)" if t.n_free_reads > 8 else ""
        bits.append(f"module state +{t.n_free_reads} args to promote{heavy}")

    return target, round(score, 2), "; ".join(bits)


# ── rendering ────────────────────────────────────────────────────────────────

def render_markdown(targets: List[RoutineTarget], top: Optional[int] = None) -> str:
    ranked = [t for t in targets if t.rank > 0]
    drivers = [t for t in targets if t.role == "driver"]
    unportable = [t for t in targets if t.rank == 0 and t.role != "driver"]
    shown = ranked[:top] if top else ranked

    out: list[str] = ["# fortranspire recon — porting worklist", ""]
    out.append(f"Scanned **{len(targets)}** routine(s): "
               f"**{len(ranked)}** portable target(s), {len(drivers)} driver(s), "
               f"{len(unportable)} unportable.")
    out.append("")
    out.append("> Deterministic (Loki AST, no LLM). Rank = portability + leaf "
               "bonus + fan-in − module-state cost.")
    out.append("")
    out.append("## Start here")
    out.append("")
    out.append("| # | Routine | Target | Rank | Purity | Why | File |")
    out.append("| - | ------- | :----: | ---: | :----: | --- | ---- |")
    for i, t in enumerate(shown, 1):
        tgt = "🟢 JAX" if t.target == "jax" else "🔵 gt4py"
        out.append(f"| {i} | `{t.name}` | {tgt} | {t.rank:.1f} | {t.purity} | "
                   f"{t.reason} | `{_rel(t.file)}` |")
    if drivers:
        out.append("")
        out.append("## Drivers (skip — orchestration)")
        out.append("")
        out.append(", ".join(f"`{t.name}`" for t in drivers[:40]))
    return "\n".join(out)


def _rel(path: str) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return Path(path).name


def to_json(targets: List[RoutineTarget]) -> str:
    return json.dumps([asdict(t) for t in targets], indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="fortranspire recon",
        description="Arrival triage for a Fortran repo — ranked porting worklist (no LLM).",
    )
    parser.add_argument("paths", nargs="+", help="Repo root(s) or file(s) to scan.")
    parser.add_argument("--json", metavar="FILE",
                        help="Write the full ranked worklist as JSON for a front-end.")
    parser.add_argument("--top", type=int, default=25,
                        help="Show only the top N targets in the table (default 25).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    targets = survey(args.paths)
    print(render_markdown(targets, top=args.top))
    if args.json:
        Path(args.json).write_text(to_json(targets))
        print(f"\nFull worklist → {args.json}")
    return 0
