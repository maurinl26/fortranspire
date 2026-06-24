"""Serveur MCP (FastMCP) exposant le CodeAgent et les pipelines Fortran via HTTP/SSE."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from fortranspire.config import config


def _jail(user_path: str) -> Path:
    """Resolve ``user_path`` and refuse anything that escapes the workspace.

    The MCP tools accept file/directory paths from network-reachable
    clients. Without a check, a malicious or careless tool argument
    (``../../etc/passwd``, a symlink farm, ``/etc/shadow``) would be
    happily read or written. We resolve the path, then assert that the
    real path is contained under ``config.workspace_dir`` — or under any
    of the comma-separated extra roots in
    ``FORTRANSPIRE_WORKSPACE_EXTRA_ROOTS``. Symlinks are followed by
    ``resolve(strict=True)``, so a symlink jail-break still resolves to
    its actual target before the check.

    Set ``FORTRANSPIRE_DISABLE_JAIL=1`` to bypass — only safe when the
    server runs in stdio mode under a trusted IDE process.
    """
    if os.getenv("FORTRANSPIRE_DISABLE_JAIL") == "1":
        return Path(user_path).expanduser()

    resolved = Path(user_path).expanduser().resolve(strict=False)
    roots = [Path(config.workspace_dir).resolve()]
    extra = os.getenv("FORTRANSPIRE_WORKSPACE_EXTRA_ROOTS", "")
    roots.extend(Path(p).expanduser().resolve() for p in extra.split(":") if p)

    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    allowed = ", ".join(str(r) for r in roots)
    raise PermissionError(
        f"path '{user_path}' (-> {resolved}) escapes the configured workspace "
        f"roots [{allowed}]. Set FORTRANSPIRE_WORKSPACE_EXTRA_ROOTS to widen "
        "the allow-list, or FORTRANSPIRE_DISABLE_JAIL=1 to disable (stdio-only)."
    )

# CodeAgent is imported lazily inside ``_get_agent`` so the module can be
# loaded by clients that only call the analyze / explain / doc / graph
# tools (no LLM stack installed). The agent is the only consumer of
# langchain; the other tools talk to deterministic Loki code paths.
_agent: Any | None = None


def _get_agent() -> Any:
    global _agent
    if _agent is None:
        from fortranspire.agent.code_agent import CodeAgent  # local import
        _agent = CodeAgent()
    return _agent


mcp = FastMCP(
    name="fortranspire",
    instructions=(
        "Pipeline de transformation Fortran 90 legacy.\n"
        "\n"
        "Analyse offline (no-LLM): analyze_kernels, explain_port_cost, "
        "build_call_graph, generate_docs.\n"
        "Phase 1 (LLM): translate_kernel_gpu — Fortran → Fortran GPU "
        "(OpenACC) + Cython wrapper.\n"
        "Phase 2 (LLM): translate_kernel — Fortran → JAX (expérimental).\n"
        "Profilage: profile_kernels.\n"
        "Question libre: ask_agent."
    ),
)


@mcp.tool()
def ask_agent(query: str) -> str:
    """Envoie une requête en langage naturel à l'agent de code et retourne sa réponse.

    Args:
        query: La question ou instruction en langage naturel.
    """
    return _get_agent().run(query)


@mcp.tool()
def agent_status() -> str:
    """Retourne la configuration du serveur MCP."""
    from fortranspire.config import config
    return (
        f"Serveur MCP actif\n"
        f"  Modèle    : {config.model_name}\n"
        f"  Workspace : {config.workspace_dir}\n"
        f"  Max iter  : {config.max_iterations}\n"
    )


@mcp.tool()
def translate_kernel_gpu(filepath: str) -> str:
    """Phase 1 — Transforme un fichier Fortran en Fortran GPU + wrapper Cython.

    Pipeline : parser → PURE/ELEMENTAL → OpenACC → Cython wrapper → validation
    Compilateur : nvfortran -acc -gpu=cc80

    Args:
        filepath: Chemin absolu vers le fichier .f90.
    """
    from fortranspire.agent.translation_graph_phase1 import translation_app_phase1

    try:
        safe_path = _jail(filepath)
    except PermissionError as e:
        return f"Erreur d'accès : {e}"

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return f"Erreur de lecture : {e}"

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

    final = translation_app_phase1.invoke(initial_state)
    status = "PASSED" if final.get("validation_passed") else "FAILED"

    return (
        f"=== Phase 1 — Fortran GPU + Cython ===\n"
        f"Fichier    : {filepath}\n"
        f"Validation : {status}\n\n"
        f"Sorties :\n"
        f"  output/fortran_gpu/kernel_pure.f90  — PURE/ELEMENTAL annotated\n"
        f"  output/fortran_gpu/kernel_gpu.f90   — OpenACC pragmas\n"
        f"  output/cython/*.pyx                 — Cython wrapper\n"
        f"  output/cython/kernel_c.h            — C header (iso_c_binding)\n\n"
        f"Log validation :\n{final.get('validation_log', '')}"
    )


@mcp.tool()
def translate_kernel(filepath: str) -> str:
    """Phase 2 — Traduit un kernel Fortran en JAX (expérimental).

    Args:
        filepath: Chemin absolu vers le fichier .f90.
    """
    from fortranspire.agent.translation_graph import translation_app

    try:
        safe_path = _jail(filepath)
    except PermissionError as e:
        return f"Erreur d'accès : {e}"

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return f"Erreur de lecture : {e}"

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

    final = translation_app.invoke(initial_state)
    return (
        f"=== Phase 2 — JAX Translation ===\n"
        f"Fichier : {filepath}\n\n"
        f"Code JAX généré :\n```python\n{final.get('jax_code', '')}\n```\n\n"
        f"Reproductibilité : {final.get('test_results', {})}\n"
        f"Performances     : {final.get('performance_metrics', {})}"
    )


@mcp.tool()
def profile_kernels(filepath: str) -> str:
    """Compare les performances entre le Fortran original et sa traduction existante.

    Args:
        filepath: Chemin absolu vers le fichier .f90 original.
    """
    from fortranspire.agent.translation_graph import performance_agent

    try:
        safe_path = _jail(filepath)
    except PermissionError as e:
        return f"Erreur d'accès : {e}"

    state = {"fortran_filepath": str(safe_path), "performance_metrics": {}}  # type: ignore
    result = performance_agent(state)
    return str(result["performance_metrics"])


def _capture_main(main_fn, argv: list[str]) -> tuple[int, str]:
    """Run an argparse-based ``main(argv) -> int`` and capture its stdout.

    Wraps SystemExit (raised on ``--help`` or arg errors) so MCP tools
    never crash the server. Returns ``(rc, captured_text)``.
    """
    import sys
    from io import StringIO

    buf = StringIO()
    old_stdout, sys.stdout = sys.stdout, buf
    try:
        try:
            rc = main_fn(argv)
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 0
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()


@mcp.tool()
def analyze_kernels(
    path: str,
    sarif_out: str | None = None,
    no_toolchain_check: bool = False,
) -> str:
    """Static Loki-based analysis of one Fortran file or a directory tree.

    No LLM call, deterministic. Suitable as a CI gate or pre-flight check.

    Args:
        path: Absolute path to a `.f90` file or a directory.
        sarif_out: If set, writes a SARIF 2.1.0 report to this path
            (GitHub Code Scanning compatible) using ``--format sarif``.
        no_toolchain_check: Skip the gfortran / nvfortran probe (useful
            on hosts where compilers are absent).
    """
    from fortranspire.agent.analyze import main as analyze_main

    try:
        safe_path = _jail(path)
        safe_sarif = _jail(sarif_out) if sarif_out else None
    except PermissionError as e:
        return f"analyze rc=2\nErreur d'accès : {e}"

    argv: list[str] = []
    if safe_sarif:
        argv += ["--format", "sarif", "-o", str(safe_sarif)]
    if no_toolchain_check:
        argv.append("--no-toolchain-check")
    argv.append(str(safe_path))

    rc, text = _capture_main(analyze_main, argv)
    head = f"analyze rc={rc}\n"
    if sarif_out:
        head += f"SARIF report → {sarif_out}\n"
    return head + text


@mcp.tool()
def explain_port_cost(filepath: str) -> str:
    """Pre-flight cost + risk estimate. No LLM, no tokens consumed.

    Reports routine count, control-flow complexity, GPU portability
    score, estimated token spend, and walltime for `translate_kernel_gpu`
    and `translate_kernel`. Always call this *before* a Phase-1 / Phase-2
    port to know whether the file is portable and what it would cost.

    Args:
        filepath: Absolute path to the `.f90` file.
    """
    from fortranspire.agent.explain import main as explain_main

    try:
        safe_path = _jail(filepath)
    except PermissionError as e:
        return f"explain rc=2\nErreur d'accès : {e}"

    rc, text = _capture_main(explain_main, [str(safe_path)])
    return f"explain rc={rc}\n{text}"


@mcp.tool()
def build_call_graph(path: str, out: str | None = None) -> str:
    """Mermaid call-graph for a Fortran module or directory.

    No LLM. Outputs a `flowchart LR` Mermaid block (renders in GitHub,
    mdBook, sphinxcontrib-mermaid).

    Args:
        path: Absolute path to a `.f90` file or directory.
        out: Optional file path to write the Markdown output to. When
            omitted, the Markdown is returned inline.
    """
    from fortranspire.agent.call_graph import main as graph_main

    try:
        safe_path = _jail(path)
        safe_out = _jail(out) if out else None
    except PermissionError as e:
        return f"graph rc=2\nErreur d'accès : {e}"

    argv: list[str] = []
    if safe_out:
        argv += ["-o", str(safe_out)]
    argv.append(str(safe_path))

    rc, text = _capture_main(graph_main, argv)
    return f"graph rc={rc}\n{text}"


@mcp.tool()
def generate_docs(
    path: str,
    with_llm: bool = False,
    dry_run: bool = False,
    sphinx: bool = False,
    output_dir: str | None = None,
) -> str:
    """Generate inline `!>` docstrings for a Fortran file or directory.

    With ``with_llm=False`` (default), inject placeholders deterministically
    — safe to call without any API key. With ``with_llm=True``, fill the
    placeholders by calling the configured LLM endpoint.

    Args:
        path: Absolute path to a `.f90` file or directory.
        with_llm: Call the LLM to write the docstring content. When
            False, ``--no-llm`` is passed to the underlying command.
        dry_run: Print what would change without writing files.
        sphinx: When True, also scaffold a Sphinx site under
            ``output_dir`` (or ``./documentation`` by default).
        output_dir: Override the Sphinx output directory.
    """
    from fortranspire.agent.document import main as doc_main

    try:
        safe_path = _jail(path)
        safe_outdir = _jail(output_dir) if output_dir else None
    except PermissionError as e:
        return f"doc rc=2\nErreur d'accès : {e}"

    argv: list[str] = []
    if not with_llm:
        argv.append("--no-llm")
    if dry_run:
        argv.append("--dry-run")
    if sphinx:
        argv.append("--sphinx")
    if safe_outdir:
        argv += ["--output", str(safe_outdir)]
    argv.append(str(safe_path))

    rc, text = _capture_main(doc_main, argv)
    return f"doc rc={rc}\n{text}"


def _install_auth(mcp_instance) -> None:
    """Attach the auth + rate-limit + audit middleware when configured.

    Backwards-compatible: when no registry file and no `API_KEY` env are
    set, the server runs unauthenticated (legacy behavior). When `API_KEY`
    is set, it's promoted to a single-token registry entry. When
    `FORTRANSPIRE_TOKENS_FILE` is set, the JSON registry takes precedence.
    """
    from fortranspire.security.auth import RateLimiter, TokenRegistry, build_middleware

    registry = TokenRegistry.from_env()
    if not registry:
        print("[Sécurité] Pas de token configuré — serveur public (API_KEY ou "
              "FORTRANSPIRE_TOKENS_FILE pour activer l'auth).")
        return

    print(f"[Sécurité] Auth activée — {len(registry)} token(s), "
          f"audit log → {os.getenv('FORTRANSPIRE_AUDIT_PATH', 'output/audit.jsonl')}")

    middleware_cls = build_middleware(registry, RateLimiter())
    if hasattr(mcp_instance, "_app"):
        mcp_instance._app.add_middleware(middleware_cls)
    else:
        print("[Erreur] Impossible de sécuriser l'API : app interne non accessible.")


def main() -> None:
    """Console entry point — ``fortranspire mcp``.

    Default transport is SSE on ``$MCP_HOST:$MCP_PORT`` (network-exposed,
    auth-able). Pass ``--stdio`` to speak the stdio JSON-RPC framing
    instead — the mode mistral-vibe and Claude Code Desktop spawn locally
    when they own the server lifecycle. stdio is local-by-construction
    (stdin/stdout only), so the HTTP auth middleware is skipped.
    """
    import sys

    argv = sys.argv[1:]
    if "--stdio" in argv or os.getenv("FORTRANSPIRE_MCP_STDIO") == "1":
        # stdio: stay silent on stdout (the client speaks JSON-RPC there).
        # FastMCP's banner is gated by FASTMCP_SHOW_SERVER_BANNER.
        os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "0")
        # stdio mode: the IDE owns the subprocess lifecycle and the trust
        # boundary is the local user. Disable the workspace jail by default
        # so cross-project analysis (e.g. PHYEX under ~/PHYEX from a server
        # installed elsewhere) works out of the box. Set
        # FORTRANSPIRE_DISABLE_JAIL=0 explicitly to keep it on.
        os.environ.setdefault("FORTRANSPIRE_DISABLE_JAIL", "1")
        mcp.run(transport="stdio")
        return

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    _install_auth(mcp)
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
