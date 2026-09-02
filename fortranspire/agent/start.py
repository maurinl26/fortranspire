"""``fortranspire start`` — the didactic "I don't know where to begin" entry point.

A newcomer with a Fortran repo faces a wall: dozens of routines, no idea which are
portable, what to run, or whether their machine can even run it. Every other verb
assumes you already know your target file and target backend. This one does not:
point it at a directory (or nothing — it takes the cwd) and it

  1. triages the repo (reusing :mod:`recon`, no LLM, no tokens),
  2. checks the toolchain actually installed here (what each piece unlocks, and the
     remedy for what's missing), and
  3. explains the pipeline, ranks the kernels worth porting with the *reason*, and
     hands you the exact next command — or, with ``--run``, asks GPU vs JAX and runs it.

It teaches the map: *you are here → this is the best kernel → your machine can do
this → this is the next move.* Deterministic and free; Phase-1 GPU generation is the
recommended first step precisely because it needs no LLM, no API key and no GPU, so
a full result lands before any model or accelerator is involved.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from typing import List, Optional, Tuple

from fortranspire.agent.recon import RoutineTarget, _rel, survey

_RULE = "━" * 64


# ── Toolchain doctor ─────────────────────────────────────────────────────────

def _have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _have_mod(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _toolchain() -> List[Tuple[str, bool, str, str]]:
    """(name, present, what it unlocks, remedy) for each tool the pipeline can use."""
    return [
        ("gfortran",   _have_cmd("gfortran"),
         "CPU oracle for numerical equivalence + GPU syntax check", "install gcc/gfortran"),
        ("Cython",     _have_mod("Cython"),
         "build the Phase-1 Python wrapper", "pip install Cython"),
        ("jax",        _have_mod("jax"),
         "Phase 2 (JAX) + gradient check", "pip install 'fortranspire[jax]'"),
        ("meson",      _have_mod("mesonbuild") or _have_cmd("meson"),
         "f2py build backend (numpy 2.x equivalence)", "pip install meson"),
        ("ninja",      _have_cmd("ninja"),
         "f2py build backend", "pip install ninja"),
        ("nvfortran",  _have_cmd("nvfortran"),
         "compile the OpenACC GPU port", "NVHPC SDK — or the fortranspire-nvhpc image / RunPod"),
        ("nvidia-smi", _have_cmd("nvidia-smi"),
         "an NVIDIA GPU to RUN the port", "run on a GPU node (RunPod / sovereign)"),
        ("cupy",       _have_mod("cupy"),
         "CuPy device-array interop (zero-copy)", "pip install cupy-cudaXX"),
        ("nsys",       _have_cmd("nsys"),
         "GPU profiling in the loop", "NVHPC / Nsight Systems"),
        ("runpodctl",  _have_cmd("runpodctl"),
         "on-demand remote GPU validation", "install runpodctl + set RUNPOD_API_KEY"),
    ]


def _capabilities(tc: dict) -> List[Tuple[str, bool, str]]:
    """(capability, available, note) derived from what is installed."""
    jax_equiv = tc["gfortran"] and tc["jax"] and (tc["meson"] and tc["ninja"])
    return [
        ("Generate Phase-1 GPU (OpenACC) + Cython wrapper", True,
         "always — deterministic, no LLM, no API key, no GPU"),
        ("Translate to JAX + check gradients", tc["jax"],
         "" if tc["jax"] else "needs jax — pip install 'fortranspire[jax]'"),
        ("Numerically validate JAX vs Fortran (f2py)", jax_equiv,
         "" if jax_equiv else "needs gfortran + jax + meson + ninja"),
        ("Compile & run the OpenACC GPU port", tc["nvfortran"] and tc["nvidia-smi"],
         "" if (tc["nvfortran"] and tc["nvidia-smi"])
         else "no nvfortran/GPU here → validate on RunPod or a sovereign GPU node"),
    ]


def _render_toolchain() -> Tuple[str, dict]:
    rows = _toolchain()
    tc = {name: present for name, present, _u, _r in rows}
    out = ["Toolchain on this machine:"]
    for name, present, unlocks, remedy in rows:
        mark = "✓" if present else "✗"
        tail = f"  →  {remedy}" if not present else ""
        out.append(f"  {mark} {name:<11}{unlocks}{tail}")
    out += ["", "What you can do here:"]
    for cap, ok, note in _capabilities(tc):
        mark = "✓" if ok else "✗"
        out.append(f"  {mark} {cap}" + (f"   ({note})" if note else ""))
    return "\n".join(out), tc


# ── Guide ────────────────────────────────────────────────────────────────────

def _pipeline_blurb() -> str:
    return (
        "The pipeline, in three moves:\n"
        "  1. recon     triage — which routines are worth porting, ranked   (you are here)\n"
        "  2. gpu / jax port the top kernel — GPU (OpenACC + Cython, deterministic)\n"
        "               or JAX (differentiable, Phase 2)\n"
        "  3. validate  check it — gradients + numerical equivalence vs the original Fortran"
    )


_VERB = {"gpu": "gpu", "jax": "translate"}


def _default_target(t: RoutineTarget) -> str:
    """Phase-1 GPU is the didactic default (no LLM/key/GPU needed to generate)."""
    return "gpu"


def _guide(targets: List[RoutineTarget], path: str, top: int) -> Tuple[str, Optional[RoutineTarget], dict]:
    ranked = [t for t in targets if t.rank > 0]
    drivers = [t for t in targets if t.role == "driver"]
    n_files = len({t.file for t in targets})
    tc_text, tc = _render_toolchain()

    out: List[str] = [
        _RULE, "  fortranspire start — where do I begin?", _RULE, "",
        f"Scanned {path}: {n_files} Fortran file(s), {len(targets)} routine(s).",
        "", tc_text, "", _pipeline_blurb(), "",
    ]

    if not ranked:
        out += [
            "No directly portable kernel found here (routines are drivers, or carry "
            "I/O / state / derived types).", "",
            "Try:",
            f"  fortranspire recon {path}      # the full ranked worklist, with reasons",
            f"  fortranspire explain {path}    # cost + risk estimate (no tokens)",
        ]
        return "\n".join(out) + "\n", None, tc

    shown = ranked[:top]
    out += [f"Top {len(shown)} target(s), best first:", ""]
    for i, t in enumerate(shown, 1):
        out.append(f"  #{i}  {t.name:<26} {_rel(t.file)}")
        out.append(f"      why  : {t.reason}")
        out.append(f"      next : fortranspire {_VERB[_default_target(t)]} {_rel(t.file)}")
        out.append("")

    if drivers:
        names = ", ".join(f"`{d.name}`" for d in drivers[:6])
        out += [f"Drivers (orchestrate, not ported directly): {names}", ""]

    best = shown[0]
    jax_note = "" if tc["jax"] else "   # needs jax — pip install 'fortranspire[jax]'"
    out += [
        _RULE, "Next step — start with the deterministic GPU port "
        "(no LLM, no key, no GPU to generate):", "",
        f"  fortranspire gpu {_rel(best.file)}",
        f"  fortranspire translate {_rel(best.file)}{jax_note or '   # Phase 2 — JAX (needs a model; sovereign on vibe)'}",
        "", "Re-run with --run to execute the recommended step (it will ask GPU vs JAX).",
    ]
    return "\n".join(out) + "\n", best, tc


def _choose_target(default: str) -> str:
    """Ask GPU vs JAX when interactive; fall back to the default otherwise."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default
    try:
        ans = input(f"\nWhich port? [g]pu (OpenACC+Cython) / [j]ax  "
                    f"(default {default}): ").strip().lower()
    except EOFError:
        return default
    if ans.startswith("j"):
        return "jax"
    if ans.startswith("g"):
        return "gpu"
    return default


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fortranspire start",
        description="Didactic entry point: triage a repo, check the toolchain, guide the first port.",
    )
    ap.add_argument("path", nargs="?", default=".",
                    help="repo directory or a single file (default: current directory)")
    ap.add_argument("--top", type=int, default=5, help="how many ranked targets to show")
    ap.add_argument("--run", action="store_true",
                    help="execute the recommended first step on the top target")
    ap.add_argument("--target", choices=("gpu", "jax"),
                    help="pick the backend for --run (skips the interactive question)")
    args = ap.parse_args(argv if argv is not None else None)

    import os
    if not os.path.exists(args.path):
        print(f"fortranspire start: path not found: {args.path}\n"
              f"Point me at a Fortran repo directory (or run me from inside one).",
              file=sys.stderr)
        return 2

    targets = survey([args.path])
    text, best, tc = _guide(targets, args.path, args.top)
    print(text)

    if not args.run:
        return 0
    if best is None:
        print("Nothing to run — no portable kernel found.", file=sys.stderr)
        return 1

    target = args.target or _choose_target(_default_target(best))
    if target == "jax" and not tc["jax"]:
        print("JAX is not installed here (pip install 'fortranspire[jax]'). "
              "Falling back to the GPU port.", file=sys.stderr)
        target = "gpu"
    verb = _VERB[target]

    print(f"\n{_RULE}\n  Running: fortranspire {verb} {_rel(best.file)}\n{_RULE}\n")
    from fortranspire.cli import main as cli_main
    return cli_main([verb, best.file])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
