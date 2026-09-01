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


# ── Health probe ──────────────────────────────────────────────────────────
# Declared as a public endpoint by `integration/le-chat-connector.json`
# (`health_check: /health`) and listed in `_OPEN_PATHS` of the auth
# middleware, so it answers before the bearer check. Deployment targets
# (European Weather Cloud, Le Chat connector directory, any load balancer)
# probe this to decide whether the instance is live — it must never
# require a token and must never touch the LLM stack.
@mcp.custom_route("/health", methods=["GET"])
async def health(_request):  # pragma: no cover - exercised via HTTP in tests
    from starlette.responses import JSONResponse

    from fortranspire import __version__

    return JSONResponse(
        {
            "status": "ok",
            "service": "fortranspire",
            "version": __version__,
            "transport": "sse",
            "tools": len(_TOOL_NAMES),
        }
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
def translate_kernel_gpu(path: str) -> str:
    """Phase 1 — Transforme un fichier Fortran en Fortran GPU + wrapper Cython.

    Pipeline : parser → PURE/ELEMENTAL → OpenACC → Cython wrapper → validation
    Compilateur : nvfortran -acc -gpu=cc80

    Args:
        path: Chemin absolu vers le fichier .f90.
    """
    from fortranspire.agent.translation_graph_phase1 import translation_app_phase1

    try:
        safe_path = _jail(path)
    except PermissionError as e:
        return f"Erreur d'accès : {e}"

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return f"Erreur de lecture : {e}"

    initial_state = {
        "fortran_filepath": str(safe_path),
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
        f"Fichier    : {safe_path}\n"
        f"Validation : {status}\n\n"
        f"Sorties :\n"
        f"  output/fortran_gpu/kernel_pure.f90  — PURE/ELEMENTAL annotated\n"
        f"  output/fortran_gpu/kernel_gpu.f90   — OpenACC pragmas\n"
        f"  output/cython/*.pyx                 — Cython wrapper\n"
        f"  output/cython/kernel_c.h            — C header (iso_c_binding)\n\n"
        f"Log validation :\n{final.get('validation_log', '')}"
    )


@mcp.tool()
def translate_kernel(path: str) -> str:
    """Phase 2 — Traduit un kernel Fortran en JAX (expérimental).

    Args:
        path: Chemin absolu vers le fichier .f90.
    """
    from fortranspire.agent.translation_graph import translation_app

    try:
        safe_path = _jail(path)
    except PermissionError as e:
        return f"Erreur d'accès : {e}"

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return f"Erreur de lecture : {e}"

    initial_state = {
        "fortran_filepath": str(safe_path),
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
        f"Fichier : {safe_path}\n\n"
        f"Code JAX généré :\n```python\n{final.get('jax_code', '')}\n```\n\n"
        f"Reproductibilité : {final.get('test_results', {})}\n"
        f"Performances     : {final.get('performance_metrics', {})}"
    )


@mcp.tool()
def profile_kernels(path: str) -> str:
    """Compare les performances entre le Fortran original et sa traduction existante.

    Args:
        path: Chemin absolu vers le fichier .f90 original.
    """
    from fortranspire.agent.translation_graph import performance_agent

    try:
        safe_path = _jail(path)
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
def explain_port_cost(path: str) -> str:
    """Pre-flight cost + risk estimate. No LLM, no tokens consumed.

    Reports routine count, control-flow complexity, GPU portability
    score, estimated token spend, and walltime for `translate_kernel_gpu`
    and `translate_kernel`. Always call this *before* a Phase-1 / Phase-2
    port to know whether the file is portable and what it would cost.

    Args:
        path: Absolute path to the `.f90` file or directory.
    """
    from fortranspire.agent.explain import main as explain_main

    try:
        safe_path = _jail(path)
    except PermissionError as e:
        return f"explain rc=2\nErreur d'accès : {e}"

    rc, text = _capture_main(explain_main, [str(safe_path)])
    return f"explain rc={rc}\n{text}"


# ── Inline-source tools (hosted / Le Chat) ─────────────────────────────────
# The path-taking tools above assume the client and server share a
# filesystem, which is true for a local stdio IDE and false for a hosted
# SSE endpoint: a Le Chat user's Fortran file is on their machine, not on
# ours. These variants take the source text itself, so the hosted
# connector is actually useful rather than merely reachable. They are the
# only tools that should face a public directory — deterministic, no LLM,
# no token, and nothing written outside a temp file that is deleted before
# returning.

import contextlib as _contextlib
import tempfile as _tempfile


@_contextlib.contextmanager
def _source_as_file(source: str, filename: str | None):
    """Materialise inline source as a temp file under the workspace.

    The deterministic analysers take a path; the jail requires it to sit
    under the workspace root. So the source is written to a private temp
    directory inside the workspace, handed to the existing code path, and
    removed on exit — no second implementation, and nothing persists.

    ``filename`` only shapes the suffix, so the frontend picks fixed vs
    free form and the preprocessor triggers on an uppercase suffix. Any
    directory component is stripped: a hosted caller must not steer where
    the file lands.
    """
    from pathlib import Path as _Path

    suffix = ".f90"
    if filename:
        stem_suffix = _Path(filename).suffix
        if stem_suffix:
            suffix = stem_suffix

    root = _Path(config.workspace_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = _tempfile.mkdtemp(prefix="lechat-", dir=str(root))
    tmp_file = _Path(tmp_dir) / f"kernel{suffix}"
    try:
        tmp_file.write_text(source, encoding="utf-8")
        yield tmp_file
    finally:
        import shutil as _shutil

        _shutil.rmtree(tmp_dir, ignore_errors=True)


def _reject_oversize(source: str) -> str | None:
    """Guard against a caller pasting a whole codebase into one request."""
    limit = int(os.getenv("FORTRANSPIRE_MAX_SOURCE_BYTES", "1000000"))
    size = len(source.encode("utf-8"))
    if size > limit:
        return (
            f"source is {size} bytes, over the {limit}-byte limit for an "
            "inline request. Split it or run the server locally over stdio."
        )
    return None


@mcp.tool()
def analyze_source(source: str, filename: str | None = None) -> str:
    """Static Loki analysis of Fortran source passed inline. No LLM, no token.

    Use this when the Fortran is not a file on the server — e.g. from Le
    Chat or any hosted client. Reports GPU-portability findings the same
    way ``analyze_kernels`` does for a path.

    Args:
        source: The Fortran source text.
        filename: Optional name, used only to pick the dialect from the
            suffix (``.f`` fixed form, ``.F90`` triggers preprocessing).
    """
    from fortranspire.agent.analyze import main as analyze_main

    too_big = _reject_oversize(source)
    if too_big:
        return f"analyze rc=2\n{too_big}"
    with _source_as_file(source, filename) as f:
        rc, text = _capture_main(analyze_main, ["--no-toolchain-check", str(f)])
    return f"analyze rc={rc}\n{text}"


@mcp.tool()
def explain_source(source: str, filename: str | None = None) -> str:
    """Pre-flight cost + risk estimate for inline Fortran source. No LLM.

    The hosted counterpart of ``explain_port_cost``: same port-cost report,
    for source that is not a file on the server.

    Args:
        source: The Fortran source text.
        filename: Optional name, used only to pick the dialect.
    """
    from fortranspire.agent.explain import main as explain_main

    too_big = _reject_oversize(source)
    if too_big:
        return f"explain rc=2\n{too_big}"
    with _source_as_file(source, filename) as f:
        rc, text = _capture_main(explain_main, [str(f)])
    return f"explain rc={rc}\n{text}"


@mcp.tool()
def build_call_graph_source(source: str, filename: str | None = None) -> str:
    """Mermaid call-graph for inline Fortran source. No LLM.

    Args:
        source: The Fortran source text.
        filename: Optional name, used only to pick the dialect.
    """
    from fortranspire.agent.call_graph import main as graph_main

    too_big = _reject_oversize(source)
    if too_big:
        return f"graph rc=2\n{too_big}"
    with _source_as_file(source, filename) as f:
        rc, text = _capture_main(graph_main, [str(f)])
    return f"graph rc={rc}\n{text}"


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


# Canonical MCP tool surface. Kept as an explicit tuple (rather than
# introspected) so a rename shows up as a diff here, in the docs, and in
# `integration/le-chat-connector.json` at the same time — the three drifted
# apart once already (README advertised `fortranspire_*`-prefixed names
# that never existed on the server).
_TOOL_NAMES: tuple[str, ...] = (
    "ask_agent",
    "agent_status",
    "translate_kernel_gpu",
    "translate_kernel",
    "profile_kernels",
    "analyze_kernels",
    "explain_port_cost",
    "build_call_graph",
    "generate_docs",
    # Inline-source variants for hosted / Le Chat use (no shared filesystem).
    "analyze_source",
    "explain_source",
    "build_call_graph_source",
)


class AuthNotInstallable(RuntimeError):
    """Raised when tokens are configured but the middleware cannot be attached.

    Fail **closed**: a server that was asked to authenticate and cannot must
    refuse to start rather than silently serve every tool to the open
    internet. The previous implementation reached into ``mcp._app``, which
    FastMCP 3.x no longer exposes — it printed an error and started anyway,
    so `API_KEY=... fortranspire mcp` served unauthenticated traffic.
    """


def _build_auth_middleware() -> list:
    """Return the ASGI middleware stack for the HTTP transports.

    Empty list when no token is configured (legacy unauthenticated mode,
    kept for OSS users running on localhost). When a token *is* configured
    and the stack cannot be built, raises ``AuthNotInstallable`` so the
    caller aborts instead of falling back to an open server.
    """
    from fortranspire.security.auth import RateLimiter, TokenRegistry, build_middleware

    registry = TokenRegistry.from_env()
    if not registry:
        print("[Sécurité] Pas de token configuré — serveur public (API_KEY ou "
              "FORTRANSPIRE_TOKENS_FILE pour activer l'auth).")
        return []

    try:
        from starlette.middleware import Middleware

        stack = [Middleware(build_middleware(registry, RateLimiter()))]
    except Exception as exc:  # noqa: BLE001 - re-raised as a fatal start error
        raise AuthNotInstallable(
            f"tokens are configured but the auth middleware could not be built "
            f"({type(exc).__name__}: {exc}). Refusing to start an unauthenticated "
            f"network server."
        ) from exc

    print(f"[Sécurité] Auth activée — {len(registry)} token(s), "
          f"audit log → {os.getenv('FORTRANSPIRE_AUDIT_PATH', 'output/audit.jsonl')}")
    return stack


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
    try:
        middleware = _build_auth_middleware()
    except AuthNotInstallable as exc:
        print(f"[Erreur] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    mcp.run(transport="sse", host=host, port=port, middleware=middleware)


if __name__ == "__main__":
    main()
