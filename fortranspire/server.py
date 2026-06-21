"""Serveur MCP (FastMCP) exposant le CodeAgent et les pipelines Fortran via HTTP/SSE."""
import os
from fastmcp import FastMCP
from fortranspire.agent.code_agent import CodeAgent

_agent: CodeAgent | None = None


def _get_agent() -> CodeAgent:
    global _agent
    if _agent is None:
        _agent = CodeAgent()
    return _agent


mcp = FastMCP(
    name="fortran-gpu-agent",
    instructions=(
        "Agent de transformation Fortran scientifique vers GPU (OpenACC) + Cython.\n"
        "Phase 1 : translate_kernel_gpu — Fortran → Fortran GPU + Cython wrapper.\n"
        "Phase 2 : translate_kernel    — Fortran → JAX (expérimental).\n"
        "Outil général : ask_agent pour questions en langage naturel sur le code."
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
        with open(filepath, 'r', encoding='utf-8') as f:
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
        with open(filepath, 'r', encoding='utf-8') as f:
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

    state = {"fortran_filepath": filepath, "performance_metrics": {}}  # type: ignore
    result = performance_agent(state)
    return str(result["performance_metrics"])


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
    """Console entry point — `run-mcp`."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    _install_auth(mcp)
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
