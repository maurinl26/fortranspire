"""Executes one `/fortranspire` command inside an isolated job workspace.

The App never reimplements a transformation: every verb shells out to the
same `fortranspire <verb>` console script the CLI and the MCP server use,
so there is exactly one code path to keep correct (issue #50, "réutiliser
le MCP/CLI — aucun fork de logique").

Isolation matters twice over. Each job gets its own temporary directory,
so two concurrent installations cannot see or clobber each other's
`output/`; and the command's path is resolved against that directory and
rejected if it escapes, so a crafted comment cannot read outside the
checkout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fortranspire.github_app.commands import Command

# Wall-clock ceiling per job. A runaway LLM port must not pin a worker.
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("FORTRANSPIRE_APP_JOB_TIMEOUT", "1800"))

# Environment variables that must never reach the subprocess for a
# deterministic verb: those runs are advertised as costing nothing, so the
# key simply should not be reachable from them.
_LLM_ENV_KEYS = (
    "MISTRAL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "API_KEY",
)


class JobError(RuntimeError):
    """The job could not run. The message is shown to the commenter."""


@dataclass
class JobResult:
    """What a finished job produced."""

    command: Command
    ok: bool
    report: str
    # Files the port produced, relative to the workspace, for the commit.
    artifacts: list[Path] = field(default_factory=list)
    metrics: str = ""

    def as_comment(self) -> str:
        """Render the GitHub comment body."""
        header = f"### `fortranspire {self.command.verb}` — `{self.command.path}`"
        status = "" if self.ok else "\n> [!WARNING]\n> The run did not complete cleanly.\n"
        body = self.report.strip()
        # GitHub rejects comments over 65 536 characters; keep well under and
        # say so rather than letting the API reject the whole reply.
        limit = 60_000
        if len(body) > limit:
            body = body[:limit] + "\n\n_(report truncated — see the run artifacts)_"
        parts = [header, status, body]
        if self.metrics:
            parts.append(f"\n<details>\n<summary>Structural metrics</summary>\n\n{self.metrics}\n\n</details>")
        parts.append(
            "\n<sub>Posted by the fortranspire GitHub App. "
            "`explain`, `analyze` and `graph` never call an LLM.</sub>"
        )
        return "\n\n".join(p for p in parts if p)


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Resolve `relative` under `workspace`, refusing anything that escapes.

    Mirrors the MCP server's `_jail()`: symlinks are followed before the
    containment check, so a symlink planted in the repository cannot be used
    to read outside the checkout.
    """
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise JobError(f"`{relative}` escapes the repository checkout") from exc
    if not candidate.exists():
        raise JobError(f"`{relative}` does not exist in this repository")
    return candidate


def _subprocess_env(*, needs_llm: bool) -> dict[str, str]:
    """Build the child environment, withholding LLM keys from free verbs."""
    env = dict(os.environ)
    if not needs_llm:
        for key in _LLM_ENV_KEYS:
            env.pop(key, None)
    # The transformation writes under the job workspace, never the server's.
    env.pop("WORKSPACE_DIR", None)
    return env


def _run(argv: list[str], *, cwd: Path, needs_llm: bool, timeout: int) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=_subprocess_env(needs_llm=needs_llm),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JobError(f"the run exceeded {timeout}s and was stopped") from exc
    except FileNotFoundError as exc:
        raise JobError("the `fortranspire` console script is not on PATH") from exc
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def run(
    command: Command,
    workspace: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    executable: str = "fortranspire",
) -> JobResult:
    """Run one command against a checkout already present at `workspace`."""
    target = resolve_in_workspace(workspace, command.path)
    relative = str(target.relative_to(workspace.resolve()))
    reports = Path(tempfile.mkdtemp(prefix="fortranspire-report-"))

    try:
        if command.verb == "explain":
            out = reports / "port-cost.md"
            rc, log = _run(
                [executable, "explain", "--output", str(out), relative],
                cwd=workspace, needs_llm=False, timeout=timeout,
            )
            return JobResult(command, rc == 0, _read(out) or log)

        if command.verb == "analyze":
            rc, log = _run(
                [executable, "analyze", "--no-color", "--no-toolchain-check",
                 "--fail-on", command.fail_on, relative],
                cwd=workspace, needs_llm=False, timeout=timeout,
            )
            return JobResult(command, rc == 0, f"```\n{log.strip()}\n```")

        if command.verb == "graph":
            out = reports / "call-graph.md"
            rc, log = _run(
                [executable, "graph", "--output", str(out), relative],
                cwd=workspace, needs_llm=False, timeout=timeout,
            )
            return JobResult(command, rc == 0, _read(out) or log)

        if command.verb == "doc":
            rc, log = _run(
                [executable, "doc", "--with-llm", relative],
                cwd=workspace, needs_llm=True, timeout=timeout,
            )
            return JobResult(
                command, rc == 0, f"```\n{log.strip()}\n```",
                artifacts=_changed_files(workspace),
            )

        if command.verb == "port":
            verb_args = (
                [executable, "gpu", relative]
                if command.target == "gpu"
                else [executable, "translate", relative]
            )
            rc, log = _run(verb_args, cwd=workspace, needs_llm=True, timeout=timeout)
            metrics = ""
            output_root = workspace / "output"
            if output_root.is_dir():
                # `bench` reports structural metrics (routine, pragma and
                # file counts, generated bytes, LLM cost) — not wall-clock
                # speedup, which would need a GPU on the runner.
                _, metrics_log = _run(
                    [executable, "bench", "--format", "text", str(output_root)],
                    cwd=workspace, needs_llm=False, timeout=timeout,
                )
                metrics = f"```\n{metrics_log.strip()}\n```"
            return JobResult(
                command, rc == 0, f"```\n{log.strip()}\n```",
                artifacts=_changed_files(workspace), metrics=metrics,
            )

        raise JobError(f"verb `{command.verb}` has no runner")
    finally:
        shutil.rmtree(reports, ignore_errors=True)


def _changed_files(workspace: Path) -> list[Path]:
    """List files the run created or modified, relative to the workspace."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace, capture_output=True, text=True, check=False,
    )
    changed: list[Path] = []
    for line in completed.stdout.splitlines():
        # Porcelain v1: two status characters, a space, then the path.
        path = line[3:].strip()
        if path:
            changed.append(Path(path))
    return changed
