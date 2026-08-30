"""Contract tests for the three agent-host integrations (#48, #49, #50).

Each of these files is configuration that only executes on someone else's
infrastructure — a GitHub runner, a Copilot cloud agent — so a mistake in
one is invisible locally until it fails in production. The checks here pin
the things that would silently break: the tool names the server actually
registers, the job name GitHub matches on, and the environment variables
that keep the workspace jail on.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def _registered_tools() -> set[str]:
    from fortranspire.server import mcp

    return {t.name for t in asyncio.run(mcp.list_tools())}


# ── #48 — Claude Code GitHub Action ───────────────────────────────────────

@pytest.fixture(scope="module")
def claude_workflow() -> dict:
    return yaml.safe_load(
        (ROOT / ".github/workflows/claude-port.yml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def claude_step(claude_workflow) -> dict:
    return next(
        s for s in claude_workflow["jobs"]["port"]["steps"]
        if "claude-code-action" in str(s.get("uses", ""))
    )


def test_claude_action_is_pinned_to_a_major_tag(claude_step):
    assert claude_step["uses"] == "anthropics/claude-code-action@v1"


def test_claude_mcp_config_is_valid_json(claude_step):
    """It is embedded in a shell string inside YAML — three chances to break."""
    args = claude_step["with"]["claude_args"]
    raw = args.split("--mcp-config '", 1)[1].rsplit("'", 1)[0]
    config = json.loads(raw)
    server = config["mcpServers"]["fortranspire"]
    assert server["args"] == ["mcp", "--stdio"]


def test_claude_workflow_keeps_the_workspace_jail_on(claude_step):
    """`fortranspire mcp --stdio` disables the jail by default.

    That default is right for a local IDE, where the user owns the trust
    boundary. Here the agent acts on a comment a third party wrote, so the
    jail must be switched back on — both for the action's own process and
    inside the MCP server's environment.
    """
    assert claude_step["env"]["FORTRANSPIRE_DISABLE_JAIL"] == "0"
    assert "AGENT_WORKSPACE" in claude_step["env"]

    raw = claude_step["with"]["claude_args"].split("--mcp-config '", 1)[1].rsplit("'", 1)[0]
    server_env = json.loads(raw)["mcpServers"]["fortranspire"]["env"]
    assert server_env["FORTRANSPIRE_DISABLE_JAIL"] == "0"
    assert "AGENT_WORKSPACE" in server_env


def test_claude_workflow_does_not_inline_a_secret(claude_step):
    """Secrets reach the MCP server through the process env, not the argv."""
    args = claude_step["with"]["claude_args"]
    assert "secrets." not in args, "a secret in claude_args lands in the process list"
    assert claude_step["env"]["MISTRAL_API_KEY"].startswith("${{ secrets.")


def test_claude_workflow_can_push_and_comment(claude_workflow):
    perms = claude_workflow["permissions"]
    assert perms["contents"] == "write"
    assert perms["pull-requests"] == "write"


def test_claude_workflow_serialises_runs_per_thread(claude_workflow):
    """Two concurrent ports would race on the branch and on `output/`."""
    group = claude_workflow["concurrency"]["group"]
    assert "issue.number" in group or "pull_request.number" in group
    assert claude_workflow["concurrency"]["cancel-in-progress"] is False


# ── #49 — GitHub Copilot cloud agent ──────────────────────────────────────

def test_copilot_setup_job_has_the_name_github_matches_on():
    """GitHub only picks up a job literally called `copilot-setup-steps`."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/copilot-setup-steps.yml").read_text(encoding="utf-8")
    )
    assert list(workflow["jobs"]) == ["copilot-setup-steps"]
    assert workflow["jobs"]["copilot-setup-steps"]["timeout-minutes"] <= 59


@pytest.fixture(scope="module")
def copilot_agent() -> dict:
    text = (ROOT / ".github/agents/fortran-porter.agent.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1])


def test_copilot_agent_declares_a_local_stdio_server(copilot_agent):
    server = copilot_agent["mcp-servers"]["fortranspire"]
    # Copilot maps the `stdio` spelling onto `local`; `local` is canonical.
    assert server["type"] == "local"
    assert server["args"] == ["mcp", "--stdio"]


def test_copilot_agent_allow_lists_only_tools_the_server_registers(copilot_agent):
    declared = set(copilot_agent["mcp-servers"]["fortranspire"]["tools"])
    unknown = declared - _registered_tools()
    assert not unknown, f"agent profile allow-lists unknown tools: {sorted(unknown)}"


def test_copilot_agent_exposes_no_token_spending_tool(copilot_agent):
    """Copilot invokes MCP tools with no approval prompt.

    Anything listed here runs unattended, so the LLM verbs stay out until a
    maintainer opts in deliberately (issue #49).
    """
    spends_tokens = {"translate_kernel_gpu", "translate_kernel", "ask_agent", "generate_docs"}
    declared = set(copilot_agent["mcp-servers"]["fortranspire"]["tools"])
    assert not (declared & spends_tokens), (
        f"unattended token-spending tools exposed: {sorted(declared & spends_tokens)}"
    )


def test_copilot_agent_command_matches_where_setup_installs_it(copilot_agent):
    """The profile names an absolute path; the workflow must create it.

    The config cannot rely on PATH (whether setup-step PATH changes reach
    the process that launches MCP servers is undocumented), so the two
    files agree on a literal path or the server never starts.
    """
    command = copilot_agent["mcp-servers"]["fortranspire"]["command"]
    workflow = (ROOT / ".github/workflows/copilot-setup-steps.yml").read_text(
        encoding="utf-8"
    )
    assert command in workflow, (
        f"agent profile launches {command!r}, which copilot-setup-steps.yml "
        "never installs"
    )


def test_copilot_setup_does_not_install_into_the_system_python(copilot_agent):
    """Installing into the runner's Debian Python fails on distro packages.

    `sudo pip install --break-system-packages` cannot uninstall a
    distro-owned dependency ("Cannot uninstall typing_extensions ... RECORD
    file not found") and the whole setup job exits 1.
    """
    workflow = (ROOT / ".github/workflows/copilot-setup-steps.yml").read_text(
        encoding="utf-8"
    )
    assert "--break-system-packages" not in workflow
    assert "python3 -m venv" in workflow


def test_copilot_agent_does_not_grant_edit_or_execute(copilot_agent):
    """The profile is analysis-only; it must not be able to rewrite Fortran."""
    granted = set(copilot_agent["tools"])
    assert "edit" not in granted and "execute" not in granted


# ── #50 — GitHub App ──────────────────────────────────────────────────────

def test_github_app_is_reachable_from_the_cli():
    from fortranspire.cli import _DISPATCH

    assert _DISPATCH["github-app"] == ("fortranspire.github_app.app", "main")


def test_github_app_extra_is_declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "github-app = [" in text
    assert "PyJWT" in text


def test_github_app_documents_every_verb_the_parser_accepts():
    """A verb the parser takes but the docs never mention is a trap."""
    from fortranspire.github_app.commands import VERBS

    doc = (ROOT / "docs/integrations/github-app.md")
    if not doc.exists():
        pytest.skip("docs page not written yet")
    text = doc.read_text(encoding="utf-8")
    missing = [v for v in sorted(VERBS) if f"`{v}`" not in text and f"/fortranspire {v}" not in text]
    assert not missing, f"undocumented verbs: {missing}"
