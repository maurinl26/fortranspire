"""Contract tests for the published composite GitHub Action (issue #47).

`action.yml` is a public interface: once `maurinl26/fortranspire@v1` is
referenced from someone else's workflow, renaming an input silently breaks
their CI. These tests pin the contract and keep the README table honest —
the same doc/code drift that made the README advertise MCP tool names the
server never registered.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "action.yml"


@pytest.fixture(scope="module")
def action() -> dict:
    return yaml.safe_load(ACTION.read_text(encoding="utf-8"))


def test_action_is_a_well_formed_composite(action):
    assert action["runs"]["using"] == "composite"
    assert action["name"] and action["description"]
    # Marketplace requires branding on a root action.
    assert action["branding"]["icon"] and action["branding"]["color"]
    for step in action["runs"]["steps"]:
        assert "uses" in step or "shell" in step, f"step {step.get('name')!r} has no runner"


def test_every_step_that_runs_shell_declares_bash(action):
    for step in action["runs"]["steps"]:
        if "run" in step:
            assert step.get("shell") == "bash", f"step {step.get('name')!r}"


def test_declared_outputs_reference_existing_step_ids(action):
    step_ids = {s["id"] for s in action["runs"]["steps"] if "id" in s}
    for name, spec in action["outputs"].items():
        referenced = set(re.findall(r"steps\.([A-Za-z0-9_-]+)\.outputs", spec["value"]))
        missing = referenced - step_ids
        assert not missing, f"output {name!r} reads unknown step(s) {sorted(missing)}"


def test_severity_gate_is_the_last_step(action):
    """Reports must be published before the build is allowed to fail.

    If the gate moved earlier, a failing analysis would abort the action
    before the SARIF upload, the job summary and the PR comment — exactly
    the feedback the user needs in order to act on the failure.
    """
    assert action["runs"]["steps"][-1]["name"] == "Enforce severity gate"


def test_no_llm_extra_is_installed(action):
    """The action's whole value proposition is zero-token, zero-secret CI."""
    install = next(
        s for s in action["runs"]["steps"] if s.get("name", "").startswith("Install")
    )
    assert "fortranspire[" not in install["run"], "must not pull an LLM extra"
    joined = " ".join(
        str(s.get("run", "")) + str(s.get("with", "")) for s in action["runs"]["steps"]
    )
    for secret_ish in ("MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert secret_ish not in joined, f"action must not reference {secret_ish}"


def test_readme_documents_every_input_and_output(action):
    """The README table is the contract users read before adopting."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 🤖 GitHub Action")[1].split("\n## ")[0]
    for name in action["inputs"]:
        assert f"`{name}`" in section, f"input {name!r} is undocumented in the README"
    for name in action["outputs"]:
        assert f"`{name}`" in section or name in section, (
            f"output {name!r} is undocumented in the README"
        )


def test_repo_dogfoods_the_action():
    """Acceptance criterion of #47: our own analyze.yml consumes the action."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/analyze.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["analyze"]["steps"]
    uses = [s.get("uses") for s in steps]
    assert "./" in uses, f"analyze.yml should consume the local action, got {uses}"

    action_step = next(s for s in steps if s.get("uses") == "./")
    assert action_step["with"]["version"] == "local", (
        "dogfooding must analyze the working tree, not the last published wheel"
    )


def test_workflows_that_consume_the_action_grant_the_needed_permissions():
    """`upload-sarif` and `comment` each require an explicit permission."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/analyze.yml").read_text(encoding="utf-8")
    )
    perms = workflow["permissions"]
    assert perms.get("security-events") == "write"
    action_step = next(
        s for s in workflow["jobs"]["analyze"]["steps"] if s.get("uses") == "./"
    )
    if action_step["with"].get("comment") is True:
        assert perms.get("pull-requests") == "write"
