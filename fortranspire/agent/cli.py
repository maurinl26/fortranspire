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


def translate_file(filepath: str):
    """Phase 2 — pipeline JAX (legacy)."""
    from fortranspire.agent.translation_graph import translation_app
    print(f"\n🔬 JAX Translation Pipeline")
    print(f"   Input : {filepath}")
    print(f"   Model : Mistral-Large (endpoint souverain)\n")
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "isolated_kernel": "",
        "jax_code": "",
        "compilation_error": "",
        "test_results": {},
        "performance_metrics": {},
    }
    final_state = translation_app.invoke(initial_state)
    output_path = filepath.replace(".f90", "_jax.py").replace(".F90", "_jax.py")
    with open(output_path, "w") as f:
        f.write(final_state.get("jax_code", ""))
    print(f"\n✅ Translation complete → {output_path}")


def translate_file_gpu(filepath: str):
    """Phase 1 — pipeline Fortran GPU + Cython."""
    print(f"\n{'═' * 60}")
    print(f"  🚀 Fortran → GPU Pipeline (Phase 1)")
    print(f"{'═' * 60}")
    print(f"  📂 Input  : {filepath}")
    print(f"  🤖 Model  : Mistral-Large (endpoint souverain)")
    print(f"  📦 Output : output/fortran_gpu/  +  output/cython/")
    print(f"{'─' * 60}")
    print(f"  Steps:")
    print(f"    🔍 parser         → Loki AST analysis")
    print(f"    🔧 extractor      → MODULE extraction (COMMON/SAVE/INTENT)")
    print(f"    ✨ pure_elemental  → Semantic purity annotation")
    print(f"    🚀 openacc        → !$acc parallel loop + !$acc data")
    print(f"    🐍 cython_wrapper → NumPy memoryview wrapper")
    print(f"    ✅ validation      → gfortran syntax check × 2 + nvfortran")
    print(f"{'═' * 60}\n")

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
    print(f"    🖥️  GPU : AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh")
    print(f"    📊 Bench: AZURE_GPU_HOST=<ip> bash scripts/bench_gpu.sh {filepath}")
    print(f"    🐍 Cython: cd output && python setup.py build_ext --inplace")
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


def run_translate():
    """agent-translate — Phase 2 : Fortran → JAX"""
    _deprecation_notice("agent-translate", "fortranspire translate")
    parser = argparse.ArgumentParser(
        description="🔬 Fortran → JAX Translation (Phase 2 — experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run agent-translate /path/to/kernel.f90\n"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    args = parser.parse_args()
    translate_file(args.filepath)


def run_translate_gpu():
    """agent-gpu — Phase 1 : Fortran → Fortran GPU + Cython

    Accepts both:
      uv run agent-gpu /path/to/kernel.f90
      uv run agent-gpu translate /path/to/kernel.f90
    """
    _deprecation_notice("agent-gpu", "fortranspire gpu")
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
            "  uv run agent-gpu /path/to/seismic_CPML_2D.f90\n"
            "  uv run agent-gpu translate /path/to/kernel.f90\n\n"
            "After the pipeline:\n"
            "  # Syntax check (no GPU needed)\n"
            "  gfortran -O2 -fsyntax-only output/fortran_gpu/module_kernels_gpu.f90\n\n"
            "  # Deploy and compile on GPU node\n"
            "  AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh\n\n"
            "  # CPU vs GPU speedup benchmark\n"
            "  AZURE_GPU_HOST=<ip> bash scripts/bench_gpu.sh /path/to/original.f90\n"
        ),
    )
    parser.add_argument("args", nargs="+", help="[translate] <filepath.f90>")
    parsed = parser.parse_args()
    # Strip optional subcommand for parity with agent-pipeline syntax
    parts = [p for p in parsed.args if p not in {"translate", "profile"}]
    if not parts:
        parser.error("❌ filepath is required")
    translate_file_gpu(parts[0])


def run_profile():
    """agent-profile — Performance benchmarking"""
    _deprecation_notice("agent-profile", "fortranspire profile")
    parser = argparse.ArgumentParser(
        description="📊 Fortran Performance Profile Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run agent-profile /path/to/kernel.f90\n"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 Fortran file")
    args = parser.parse_args()
    profile_file(args.filepath)


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
