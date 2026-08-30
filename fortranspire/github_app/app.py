"""Webhook receiver for the fortranspire GitHub App (issue #50).

Runs on Starlette — already a dependency through FastMCP, so the App adds
no web framework of its own.

The request path is deliberately short: verify the signature, decide
whether the comment is for us, authorise, and hand the job to a worker
thread. GitHub retries a delivery it does not get an answer to within ten
seconds, and a Phase-1 port takes minutes, so the webhook always answers
immediately and reports the outcome as a comment.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from fortranspire import __version__
from fortranspire.github_app import github
from fortranspire.github_app.auth import ConfigError, TokenCache, verify_signature
from fortranspire.github_app.commands import Command, CommandError, mentions_command, parse
from fortranspire.github_app.policy import Policy, actor_permission, authorize
from fortranspire.github_app.runner import JobError, run

log = logging.getLogger("fortranspire.github_app")

# Bounded so a burst of comments cannot spawn unbounded ports; each job
# holds a checkout and possibly an LLM session.
_MAX_WORKERS = int(os.getenv("FORTRANSPIRE_APP_WORKERS", "2"))

_TOKENS = TokenCache()
_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="fortranspire-job")


def _webhook_secret() -> str:
    secret = os.getenv("GITHUB_APP_WEBHOOK_SECRET", "")
    if not secret:
        raise ConfigError(
            "GITHUB_APP_WEBHOOK_SECRET is not set — refusing to accept deliveries"
        )
    return secret


def _branch_name(command: Command) -> str:
    stem = Path(command.path).stem.replace("_", "-")[:40] or "kernel"
    return f"fortranspire/{command.verb}-{stem}-{int(time.time())}"


def _execute(
    *,
    command: Command,
    repo_full_name: str,
    issue_number: int,
    ref: str,
    token: str,
) -> None:
    """Clone, run the verb, publish the result. Runs on a worker thread."""
    workspace = Path(tempfile.mkdtemp(prefix="fortranspire-job-"))
    checkout_dir = workspace / "repo"
    try:
        checkout = github.clone(
            repo_full_name=repo_full_name, ref=ref, token=token,
            destination=checkout_dir,
        )
        result = run(command, checkout.path)
        body = result.as_comment()

        # `port` and `doc` rewrite files; publish them as a branch and a PR
        # so a human reviews the diff before anything lands.
        if command.needs_llm and result.artifacts:
            branch = _branch_name(command)
            pushed = github.push_branch(
                checkout=checkout, repo_full_name=repo_full_name, branch=branch,
                message=f"fortranspire {command.verb}: {command.path}",
                token=token,
            )
            if pushed:
                url = github.open_pull_request(
                    repo_full_name=repo_full_name, head=branch, base=checkout.base_ref,
                    title=f"fortranspire {command.verb}: `{command.path}`",
                    body=body, token=token,
                )
                body += f"\n\nOpened {url} with the result."
            else:
                body += "\n\nThe run produced no file changes, so no PR was opened."

        github.post_comment(
            repo_full_name=repo_full_name, issue_number=issue_number,
            body=body, token=token,
        )
    except (JobError, github.GitHubError) as exc:
        github.post_comment(
            repo_full_name=repo_full_name, issue_number=issue_number,
            body=f"`/fortranspire {command.verb}` could not run: {exc}", token=token,
        )
    except Exception:  # noqa: BLE001 - a worker must never die silently
        log.exception("job failed for %s on %s", command.verb, repo_full_name)
        github.post_comment(
            repo_full_name=repo_full_name, issue_number=issue_number,
            body=(f"`/fortranspire {command.verb}` failed unexpectedly. "
                  "The operator has the details in the server log."),
            token=token,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def health(_request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "fortranspire-github-app", "version": __version__}
    )


async def webhook(request: Request) -> JSONResponse:
    body = await request.body()

    try:
        secret = _webhook_secret()
    except ConfigError as exc:
        log.error("%s", exc)
        return JSONResponse({"detail": "server misconfigured"}, status_code=500)

    if not verify_signature(body, request.headers.get("X-Hub-Signature-256"), secret):
        return JSONResponse({"detail": "bad signature"}, status_code=401)

    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return JSONResponse({"detail": "pong"})
    if event not in {"issue_comment", "pull_request_review_comment"}:
        return JSONResponse({"detail": "ignored"})

    payload = await request.json()
    if payload.get("action") != "created":
        return JSONResponse({"detail": "ignored"})

    comment = payload.get("comment") or {}
    if (comment.get("user") or {}).get("type") == "Bot":
        # Our own summary comments would otherwise re-trigger the App.
        return JSONResponse({"detail": "ignored"})

    comment_body = comment.get("body") or ""
    if not mentions_command(comment_body):
        return JSONResponse({"detail": "ignored"})

    repo_full_name = (payload.get("repository") or {}).get("full_name", "")
    installation_id = (payload.get("installation") or {}).get("id")
    actor = ((comment.get("user") or {}).get("login")) or ""
    issue = payload.get("issue") or payload.get("pull_request") or {}
    issue_number = issue.get("number")

    if not (repo_full_name and installation_id and issue_number):
        return JSONResponse({"detail": "incomplete payload"}, status_code=400)

    try:
        token = _TOKENS.get(int(installation_id))
    except ConfigError as exc:
        log.error("token minting failed: %s", exc)
        return JSONResponse({"detail": "server misconfigured"}, status_code=500)

    # Authorise before parsing so a stranger cannot probe the grammar, and
    # before any clone so an unapproved installation costs nothing.
    try:
        command = parse(comment_body)
    except CommandError as exc:
        command = None
        parse_error: str | None = str(exc)
    else:
        parse_error = None

    with httpx.Client(timeout=30) as client:
        permission = actor_permission(
            repo_full_name=repo_full_name, actor=actor, token=token, client=client
        )
    decision = authorize(
        policy=Policy.from_env(),
        installation_id=int(installation_id),
        repo_full_name=repo_full_name,
        actor=actor,
        needs_llm=bool(command and command.needs_llm),
        actor_permission=permission,
    )
    if not decision.allowed:
        log.info("refused %s on %s: %s", actor, repo_full_name, decision.reason)
        if not decision.silent:
            github.post_comment(
                repo_full_name=repo_full_name, issue_number=issue_number,
                body=decision.reason, token=token,
            )
        return JSONResponse({"detail": "refused"}, status_code=202)

    if parse_error is not None:
        github.post_comment(
            repo_full_name=repo_full_name, issue_number=issue_number,
            body=f"{parse_error}\n\nUsage: `/fortranspire <explain|analyze|graph|port|doc> <path>`",
            token=token,
        )
        return JSONResponse({"detail": "bad command"}, status_code=202)

    assert command is not None  # parse_error is None => command parsed

    # A comment on a PR runs against that PR's head; on a plain issue it
    # runs against the default branch.
    ref = (payload.get("repository") or {}).get("default_branch", "main")
    if "pull_request" in issue or payload.get("pull_request"):
        try:
            pull = github.get_pull_request(
                repo_full_name=repo_full_name, number=issue_number, token=token
            )
            ref = pull["head"]["ref"]
        except github.GitHubError:
            log.warning("could not resolve PR head; falling back to %s", ref)

    _POOL.submit(
        _execute,
        command=command, repo_full_name=repo_full_name,
        issue_number=issue_number, ref=ref, token=token,
    )
    return JSONResponse({"detail": "accepted", "verb": command.verb}, status_code=202)


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/webhooks/github", webhook, methods=["POST"]),
    ]
)


def main() -> None:
    """Console entry point — `fortranspire github-app`."""
    import uvicorn

    logging.basicConfig(level=os.getenv("FORTRANSPIRE_LOG_LEVEL", "INFO"))
    uvicorn.run(
        app,
        host=os.getenv("FORTRANSPIRE_APP_HOST", "0.0.0.0"),
        port=int(os.getenv("FORTRANSPIRE_APP_PORT", "8080")),
    )
