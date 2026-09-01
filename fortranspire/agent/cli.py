import argparse
import sys

# `translation_graph` and `translation_graph_phase1` pull in the LangChain
# stack at import time, which is only present when the [gpu] extra is
# installed. Importing them eagerly at module top would break
# `agent-analyze`, `agent-doc`, `agent-format`, `agent-explain` and
# `agent-port-batch` on the core-only install. Pulled into the functions
# that actually need them.


def _read_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Cannot read {filepath}: {e}")
        sys.exit(1)


def _sibling_output(filepath: str, tail: str) -> str:
    """An output path beside the input, **never equal to it**.

    The old ``filepath.replace(".f90", ...)`` silently returned the *input*
    path for any other suffix (``.F``, ``.f``, ``.for``): the replace did not
    match, so an empty or blocked result was written straight over the source
    file — destroying it (observed on CMAQ ``rbfeval.F``). Deriving from the
    stem makes the output distinct for every suffix, and the equality guard is
    a last-resort safety net.
    """
    from pathlib import Path

    p = Path(filepath)
    out = p.with_name(p.stem + tail)
    if out.resolve() == p.resolve():  # never overwrite the input
        out = p.with_name(p.stem + "_out" + tail)
    return str(out)


def _write_output(output_path: str, content: str, kind: str) -> None:
    """Write emitted code, but never replace an existing file with nothing.

    A blocked run emits an empty module; writing that would clobber whatever
    was at ``output_path``. Skip it and say so instead.
    """
    from pathlib import Path

    if not content.strip():
        print(f"\n   No {kind} emitted (nothing to write) — {output_path} left untouched.")
        return
    with open(output_path, "w") as f:
        f.write(content)
    print(f"\n   Written → {output_path}")


def translate_file(filepath: str, *, smoothing: str = "none",
                   module_path: list[str] | None = None) -> int:
    """Phase 2 — Fortran → functional refactoring → JAX (issue #73).

    Runs ``translation_app_phase2``: the graph that derives the functional
    interface from the INTENT map before emitting, and verifies the
    gradients afterwards.

    Returns a non-zero code when the gradient check fails. That check is
    deliberately blocking: differentiability is the whole reason to target
    JAX, and a kernel that traces with a wrong gradient is silently wrong
    exactly where the caller relies on it.

    The previous graph (``translation_graph.translation_app``) is still
    importable. It carries the Phase 3-5 experiments — halo exchange,
    surrogate models — which this pipeline does not run.
    """
    from fortranspire.agent.translation_graph_phase2 import translation_app_phase2

    print("\n🔬 Phase 2 — Fortran → functional → JAX")
    print(f"   Input     : {filepath}")
    print(f"   Smoothing : {smoothing}")
    if smoothing == "smooth":
        print("   Relaxations change the forward values — this is a modelling choice.")
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "kernel_results": [],
        "is_program": False,
        "smoothing": smoothing,
        "module_search_dirs": module_path or [],
        "executed_agents": [],
    }
    final_state = translation_app_phase2.invoke(initial_state)

    output_path = _sibling_output(filepath, "_jax.py")
    _write_output(output_path, final_state.get("jax_module", ""), "JAX module")

    blocked = [k for k in final_state.get("kernel_results", [])
               if k.get("purity") == "blocked"]
    if blocked:
        print(f"   {len(blocked)} routine(s) could not become pure functions:")
        for kernel in blocked:
            print(f"     - {kernel['routine_name']}: {kernel.get('purity_reason', '')}")

    if not final_state.get("gradcheck_passed", False):
        print("\n❌ Gradient check failed — the emitted kernels are not usable.")
        print(final_state.get("gradcheck_log", ""))
        return 1

    # Numerical equivalence against the original Fortran — the correctness gate.
    # A mismatch means the kernel is differentiable but computes the wrong thing.
    if not final_state.get("equivalence_passed", True):
        print("\n❌ Numerical equivalence failed — the JAX does not match the "
              "original Fortran.")
        for k in final_state.get("kernel_results", []):
            eq = k.get("equivalence") or {}
            if eq.get("status") == "fail":
                print(f"   - {k['routine_name']}: max|Δ| = "
                      f"{eq.get('max_abs_err', '?')}, {eq.get('mismatches')}")
        return 1

    checked = [k for k in final_state.get("kernel_results", [])
               if (k.get("equivalence") or {}).get("status") == "pass"]

    unverified = final_state.get("gradcheck_unverified", [])
    if unverified:
        print(f"\n⚠️  Translation complete — {len(unverified)} kernel(s) emitted but "
              "NOT verified (a valid index fixture is required):")
        for msg in unverified:
            print(f"   - {msg}")
        return 0

    if checked:
        print(f"\n✅ Translation complete — gradients verified and {len(checked)} "
              "kernel(s) match the original Fortran numerically.")
    else:
        print("\n✅ Translation complete, gradients verified. "
              "(Numerical equivalence skipped — no gfortran/meson, or module state.)")
    return 0


def gt4py_file(filepath: str) -> int:
    """GT4Py — Fortran → gt4py.next field operators (issue #42).

    Runs the GT4Py graph: the Phase 2 functional analysis (shared
    `functionalize` node) followed by field-operator emission and a
    type-check against gt4py.next's own frontend.

    Returns non-zero when an emitted operator fails the type-check.
    A skip (gt4py not installed) is reported but does not fail — the
    operators are still written; they are just not type-checked.
    """
    from fortranspire.agent.translation_graph_gt4py import translation_app_gt4py

    print("\n🌐 GT4Py — Fortran → gt4py.next field operators")
    print(f"   Input : {filepath}")
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "kernel_results": [],
        "is_program": False,
        "executed_agents": [],
    }
    final_state = translation_app_gt4py.invoke(initial_state)

    output_path = _sibling_output(filepath, "_gt4py.py")
    _write_output(output_path, final_state.get("gt4py_module", ""), "gt4py module")

    blocked = [k for k in final_state.get("kernel_results", [])
               if k.get("purity") == "blocked"]
    if blocked:
        print(f"   {len(blocked)} routine(s) cannot be field operators:")
        for kernel in blocked:
            print(f"     - {kernel['routine_name']}: {kernel.get('purity_reason', '')}")

    if final_state.get("type_check_skipped"):
        print("\n⚠️  gt4py not installed — operators emitted but NOT type-checked.")
        print("   `pip install gt4py` to validate them against gt4py.next.")
        return 0

    if not final_state.get("type_checked", False):
        print("\n❌ A field operator failed the gt4py type-check.")
        print(final_state.get("type_check_log", ""))
        return 1

    # Domain / halo: the driver was generated and statically checked (#82).
    problems = final_state.get("domain_problems") or []
    if problems:
        print("\n❌ Domain/halo check failed:")
        for prob in problems:
            print(f"   - {prob}")
        return 1

    print("\n✅ Field operators type-checked; drivers generated with the "
          "interior domain and offset providers.")
    print("   Static domain/halo check passed (no execution — running a "
          "Cartesian shift needs a gtfn toolchain, #82).")
    return 0


def translate_file_gpu(filepath: str, *, gpu_pragma: str = "acc"):
    """Phase 1 — pipeline Fortran GPU + Cython.

    ``gpu_pragma`` selects the directive family for the openacc/openmp
    node (issue #18): ``"acc"`` (default, OpenACC + nvfortran) or
    ``"omp"`` (OpenMP target, multi-vendor: gfortran 13+, nvfortran -mp=gpu,
    ifx -fiopenmp).
    """
    pragma_label = "OpenACC" if gpu_pragma == "acc" else "OpenMP target"
    print(f"\n{'═' * 60}")
    print(f"  🚀 Fortran → GPU Pipeline (Phase 1) — pragma: {pragma_label}")
    print(f"{'═' * 60}")
    print(f"  📂 Input  : {filepath}")
    print(f"  🤖 Model  : Mistral-Large (endpoint souverain)")
    print(f"  📦 Output : output/fortran_gpu/  +  output/cython/")
    print(f"{'─' * 60}\n")

    from fortranspire.agent.translation_graph_phase1 import translation_app_phase1
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "module_fortran": "",
        "driver_fortran": "",
        "kernel_names": [],
        "pure_elemental_fortran": "",
        "openacc_fortran": "",
        "cython_pyx": "",
        "cython_header": "",
        "cython_setup": "",
        "validation_passed": False,
        "validation_log": "",
        "gpu_pragma": gpu_pragma,
        "executed_agents": [],
    }
    final_state = translation_app_phase1.invoke(initial_state)

    passed = final_state.get("validation_passed", False)
    status_icon = "✅" if passed else "⚠️"
    status_text = "PASSED" if passed else "FAILED (see validation.log)"

    print(f"\n{'═' * 60}")
    print(f"  {status_icon} Phase 1 Complete")
    print(f"{'═' * 60}")
    print(f"  Validation : {status_text}")
    print(f"  Output     : output/fortran_gpu/  +  output/cython/")
    print(f"")
    print(f"  Next steps:")
    if not passed:
        print(f"    🔧 Check output/fortran_gpu/validation.log for errors")
        print(f"    📋 gfortran -O2 -fsyntax-only output/fortran_gpu/module_kernels_gpu.f90")
    print(f"    🖥️  GPU compile: rsync -a output/ user@<gpu-node>:~/k/ && ssh user@<gpu-node> 'cd ~/k && bash compile_gpu.sh'")
    print(f"    📊 Bench       : fortranspire bench output/")
    print(f"    🐍 Cython      : cd output && python setup.py build_ext --inplace")
    print(f"{'═' * 60}\n")

    if final_state.get("validation_log"):
        print("📋 Validation log:")
        print(final_state["validation_log"])


def profile_file(filepath: str):
    from fortranspire.agent.translation_graph import performance_agent
    print(f"\n📊 Performance Profile")
    print(f"   Input : {filepath}\n")
    state = {"fortran_filepath": filepath, "performance_metrics": {}}
    state = performance_agent(state)  # type: ignore
    print("\n📈 Performance Results:")
    for k, v in state.get("performance_metrics", {}).items():
        print(f"   {k}: {v}")


# ── UV Entry Points ────────────────────────────────────────────────────────────


def _deprecation_notice(legacy: str, new: str) -> None:
    """Print a one-line stderr warning so legacy `agent-*` callers know
    the unified `fortranspire <verb>` CLI is the way forward.

    Removal is scheduled for 0.3 — keep the version pinned in the message
    so users can plan their migration without guessing.
    """
    print(f"warning: `{legacy}` is deprecated; use `{new}` instead. "
          f"Will be removed in fortranspire 0.3.", file=sys.stderr)


def run_analyze():
    """agent-analyze — Loki-only static analysis, no LLM, SARIF/JSON/text."""
    _deprecation_notice("agent-analyze", "fortranspire analyze")
    from fortranspire.agent.analyze import main as _analyze_main
    sys.exit(_analyze_main())


def run_doc():
    """agent-doc — LLM-driven Fortran documentation (inline + Sphinx site)."""
    _deprecation_notice("agent-doc", "fortranspire doc")
    from fortranspire.agent.document import main as _doc_main
    sys.exit(_doc_main())


def run_explain():
    """agent-explain — pre-flight cost + risk estimate (no LLM, no tokens)."""
    _deprecation_notice("agent-explain", "fortranspire explain")
    from fortranspire.agent.explain import main as _explain_main
    sys.exit(_explain_main())


def run_format():
    """agent-format — fprettify-based Fortran source formatter."""
    _deprecation_notice("agent-format", "fortranspire format")
    from fortranspire.agent.format import main as _format_main
    sys.exit(_format_main())


def run_port_batch():
    """agent-port-batch — parallel Fortran → GPU port across many files."""
    _deprecation_notice("agent-port-batch", "fortranspire port-batch")
    from fortranspire.agent.batch import main as _batch_main
    sys.exit(_batch_main())


# ── Phase 2 / Phase 1 / Profile — pure entry points ──────────────────────
# Each `_*_main` holds the actual logic and is called by both:
#   - the unified `fortranspire <verb>` dispatcher (no deprecation noise), and
#   - the legacy `run_*` wrapper (prints the one-line deprecation notice).
# The split fixes a bug where calling `fortranspire gpu/translate/profile`
# through the unified CLI was triggering the deprecation message intended
# only for the legacy `agent-*` aliases.


def _domain_main():
    """Interactive domain agent — geometry catalogue + decomposition proposer.

    Geometry cannot be read from a kernel (it is a modelling choice), so the
    resolution is given by the user; the stencil **halo** is read from the
    Fortran when a kernel is supplied (via the typed domain model).
    """
    from fortranspire.agent.geometry import (
        catalogue_table,
        identify,
        propose_decomposition,
    )

    parser = argparse.ArgumentParser(
        description="🌍 Domain geometry catalogue + software-decomposition proposer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fortranspire domain --list\n"
            "  fortranspire domain O1280 --ranks 1024\n"
            "  fortranspire domain nside=1024 --ranks 512 --kernel src/stencil.f90\n"
        ),
    )
    parser.add_argument("resolution", nargs="?",
                        help="Grid resolution, e.g. O1280 / nside=1024 / C768 / R2B9")
    parser.add_argument("--list", action="store_true", help="List the geometry catalogue")
    parser.add_argument("--ranks", type=int, default=1024, help="MPI ranks (default 1024)")
    parser.add_argument("--levels", type=int, default=137, help="Vertical levels (default 137)")
    parser.add_argument("--fields", type=int, default=10, help="3-D fields for the memory estimate")
    parser.add_argument("--kernel", help="Fortran kernel — its stencil halo feeds the decomposition")
    parser.add_argument("--halo", type=int, default=None, help="Stencil halo (overrides --kernel)")
    args = parser.parse_args()

    if args.list or not args.resolution:
        print("# Geometry catalogue\n")
        print(catalogue_table())
        if not args.resolution:
            print("\nGive a resolution to propose a decomposition, "
                  "e.g. `fortranspire domain O1280 --ranks 1024`.")
        return 0

    if identify(args.resolution) is None:
        print(f"Unknown resolution {args.resolution!r}. Known families:", file=sys.stderr)
        print(catalogue_table(), file=sys.stderr)
        return 2

    # Halo: explicit flag, or read from the kernel's stencil, or 0.
    halo = args.halo
    if halo is None and args.kernel:
        from fortranspire.agent.domain_model import build_domain_model

        model = build_domain_model(_read_file(args.kernel))
        halo = model.max_halo
        print(f"# stencil halo {halo} read from {args.kernel}\n")
    halo = halo or 0

    decomp = propose_decomposition(
        args.resolution, n_ranks=args.ranks, halo=halo,
        levels=args.levels, fields=args.fields,
    )
    print(decomp.render())
    return 0


def _gt4py_main():
    """Fortran → gt4py.next field operators (unified `fortranspire gt4py`)."""
    parser = argparse.ArgumentParser(
        description="🌐 Fortran → gt4py.next field operators (experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  fortranspire gt4py /path/to/kernel.f90\n",
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    args = parser.parse_args()
    return gt4py_file(args.filepath)


def _translate_main():
    """Phase 2 — Fortran → JAX (shared by unified + legacy entries)."""
    parser = argparse.ArgumentParser(
        description="🔬 Fortran → JAX Translation (Phase 2 — experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  fortranspire translate /path/to/kernel.f90\n"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    parser.add_argument(
        "--smoothing", choices=("none", "guarded", "smooth"), default="none",
        help=(
            "How far to go for gradients. 'none' (default): faithful translation; "
            "gradients may be zero or one-sided where the Fortran was non-smooth. "
            "'guarded': NaN and infinite-derivative guards only — forward values "
            "unchanged. 'smooth': also relax MAX/MIN/ABS/thresholds — this CHANGES "
            "what the code computes and is a modelling decision, so applied "
            "relaxations are named in the emitted code."
        ),
    )
    parser.add_argument(
        "--module-path", action="append", metavar="DIR", default=[],
        help=(
            "Directory holding the modules a routine USEs, so their symbols' "
            "types/shapes resolve (issue #99). The routine's own directory is "
            "always searched; add a mechanism dir here to resolve promoted state "
            "and lift a gradcheck 'needs fixture'. Repeatable; "
            "FORTRANSPIRE_MODULE_PATH also works."
        ),
    )
    args = parser.parse_args()
    # Propagate the gradient-check verdict: a failed check must fail the
    # command, not just print a warning (issue #73).
    return translate_file(args.filepath, smoothing=args.smoothing,
                          module_path=args.module_path)


def run_translate():
    """agent-translate — Phase 2 : Fortran → JAX (legacy alias)."""
    _deprecation_notice("agent-translate", "fortranspire translate")
    _translate_main()


def _translate_gpu_main():
    """Phase 1 — Fortran → GPU + Cython (shared by unified + legacy entries)."""
    parser = argparse.ArgumentParser(
        description=(
            "🚀 Fortran → Fortran GPU + Cython (Phase 1)\n\n"
            "Transforms scientific Fortran (COMMON blocks, SAVE, implicit INTENT)\n"
            "into GPU-ready Fortran (OpenACC) + a Python/Cython wrapper.\n\n"
            "Fortran patterns handled:\n"
            "  COMMON BLOCKS → explicit MODULE arguments\n"
            "  SAVE variables → INTENT(INOUT) args (no hidden state)\n"
            "  POINTER       → allocatable or direct argument\n"
            "  INTENT gaps   → inferred by Loki AST + LLM\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fortranspire gpu /path/to/seismic_CPML_2D.f90\n"
            "  fortranspire gpu --gpu-pragma omp /path/to/kernel.f90\n\n"
            "After the pipeline:\n"
            "  # Syntax check (no GPU needed)\n"
            "  gfortran -O2 -fsyntax-only output/fortran_gpu/module_kernels_gpu.f90\n\n"
            "  # GPU compile: copy output/ to a GPU node and run `bash compile_gpu.sh`\n"
            "  # Benchmark    : `fortranspire bench output/` (regression detector)\n"
        ),
    )
    parser.add_argument("args", nargs="+", help="[translate] <filepath.f90>")
    parser.add_argument("--gpu-pragma", choices=("acc", "omp"), default="acc",
                        help="GPU directive family (default: acc). "
                             "'omp' = OpenMP target offload")
    parsed = parser.parse_args()
    # Strip optional subcommand for parity with legacy agent-pipeline syntax
    parts = [p for p in parsed.args if p not in {"translate", "profile"}]
    if not parts:
        parser.error("❌ filepath is required")
    translate_file_gpu(parts[0], gpu_pragma=parsed.gpu_pragma)


def run_translate_gpu():
    """agent-gpu — Phase 1 : Fortran → Fortran GPU + Cython (legacy alias)."""
    _deprecation_notice("agent-gpu", "fortranspire gpu")
    _translate_gpu_main()


def _profile_main():
    """Performance benchmarking (shared by unified + legacy entries)."""
    parser = argparse.ArgumentParser(
        description="📊 Fortran Performance Profile Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  fortranspire profile /path/to/kernel.f90\n"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    args = parser.parse_args()
    profile_file(args.filepath)


def run_profile():
    """agent-profile — Performance benchmarking (legacy alias)."""
    _deprecation_notice("agent-profile", "fortranspire profile")
    _profile_main()


def main():
    """agent-pipeline — Master dispatcher"""
    parser = argparse.ArgumentParser(
        description=(
            "🚀 Fortran Agent Pipeline — Fortran → GPU + JAX\n\n"
            "Transforms scientific Fortran into GPU-ready code (OpenACC)\n"
            "and optionally into JAX for differentiable ML integration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  agent-pipeline translate-gpu kernel.f90   # → GPU (Phase 1, recommended)\n"
            "  agent-pipeline translate     kernel.f90   # → JAX  (Phase 2, experimental)\n"
            "  agent-pipeline profile       kernel.f90   # → performance report\n"
        ),
    )
    parser.add_argument(
        "action",
        choices=["translate", "translate-gpu", "profile"],
        help=(
            "translate-gpu  🚀 Fortran → Fortran GPU (OpenACC) + Cython  [Phase 1]\n"
            "translate      🔬 Fortran → JAX                              [Phase 2, experimental]\n"
            "profile        📊 Performance benchmark\n"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    args = parser.parse_args()

    if args.action == "translate":
        translate_file(args.filepath)
    elif args.action == "translate-gpu":
        translate_file_gpu(args.filepath)
    elif args.action == "profile":
        profile_file(args.filepath)


if __name__ == "__main__":
    main()
