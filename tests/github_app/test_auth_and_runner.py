"""Credential handling and job isolation (issue #50)."""
from __future__ import annotations

import hashlib
import hmac
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from fortranspire.github_app.auth import (
    ConfigError,
    InstallationToken,
    verify_signature,
)
from fortranspire.github_app.commands import parse
from fortranspire.github_app.runner import JobError, resolve_in_workspace, run

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "doc_kernel.f90"


class TestWebhookSignature:
    SECRET = "s3cr3t"
    BODY = b'{"action":"created"}'

    def _valid(self) -> str:
        return "sha256=" + hmac.new(
            self.SECRET.encode(), self.BODY, hashlib.sha256
        ).hexdigest()

    def test_valid_signature(self):
        assert verify_signature(self.BODY, self._valid(), self.SECRET) is True

    @pytest.mark.parametrize(
        "header",
        [None, "", "sha256=", "sha1=abc", "deadbeef", "sha256=" + "0" * 64],
    )
    def test_invalid_headers(self, header):
        assert verify_signature(self.BODY, header, self.SECRET) is False

    def test_body_tampering(self):
        assert verify_signature(self.BODY + b" ", self._valid(), self.SECRET) is False

    def test_wrong_secret(self):
        assert verify_signature(self.BODY, self._valid(), "other") is False

    def test_empty_server_secret_raises_rather_than_accepting(self):
        """An empty secret must never be read as 'no check needed'."""
        with pytest.raises(ConfigError):
            verify_signature(self.BODY, self._valid(), "")


class TestInstallationToken:
    def test_freshness_accounts_for_the_refresh_margin(self):
        assert InstallationToken("t", time.time() + 3600).is_fresh is True
        # Inside the refresh margin: treated as stale so it is re-minted.
        assert InstallationToken("t", time.time() + 60).is_fresh is False
        assert InstallationToken("t", time.time() - 1).is_fresh is False

    def test_repr_never_leaks_the_token(self):
        """Tokens end up in tracebacks and logs unless this holds."""
        assert "super-secret" not in repr(InstallationToken("super-secret", 0))


@pytest.fixture
def checkout(tmp_path):
    """A git working tree containing one clean Fortran kernel."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / "src").mkdir()
    shutil.copy(FIXTURE, workspace / "src" / "kernel.f90")
    return workspace


class TestWorkspaceJail:
    @pytest.mark.parametrize(
        "relative", ["../outside.f90", "../../etc/passwd", "/etc/passwd"]
    )
    def test_paths_escaping_the_checkout_are_refused(self, checkout, relative):
        with pytest.raises(JobError, match="escapes"):
            resolve_in_workspace(checkout, relative)

    def test_symlink_out_of_the_checkout_is_refused(self, checkout, tmp_path):
        """A symlink committed to the repo must not become a read primitive."""
        secret = tmp_path / "secret.txt"
        secret.write_text("private")
        (checkout / "link.f90").symlink_to(secret)
        with pytest.raises(JobError, match="escapes"):
            resolve_in_workspace(checkout, "link.f90")

    def test_missing_path_is_reported_clearly(self, checkout):
        with pytest.raises(JobError, match="does not exist"):
            resolve_in_workspace(checkout, "src/absent.f90")

    def test_legitimate_path_resolves(self, checkout):
        resolved = resolve_in_workspace(checkout, "src/kernel.f90")
        assert resolved.name == "kernel.f90"


@pytest.mark.slow
class TestDeterministicVerbs:
    """These shell out to the real `fortranspire` CLI — no LLM involved."""

    def test_explain_returns_the_port_cost_report(self, checkout):
        result = run(parse("/fortranspire explain src/kernel.f90"), checkout)
        assert result.ok is True
        assert "port-cost estimate" in result.report

    def test_analyze_returns_findings(self, checkout):
        result = run(parse("/fortranspire analyze src/kernel.f90"), checkout)
        assert result.ok is True
        assert "0 error(s)" in result.report

    def test_graph_returns_mermaid(self, checkout):
        result = run(parse("/fortranspire graph src/kernel.f90"), checkout)
        assert result.ok is True
        assert "mermaid" in result.report

    def test_comment_body_is_bounded(self, checkout):
        """GitHub rejects comments over 65 536 characters."""
        result = run(parse("/fortranspire explain src/kernel.f90"), checkout)
        assert len(result.as_comment()) < 65_536


class TestSecretWithholding:
    def test_llm_keys_are_removed_for_deterministic_verbs(self, monkeypatch):
        """`explain` is advertised as free — the key should not be reachable."""
        from fortranspire.github_app.runner import _subprocess_env

        monkeypatch.setenv("MISTRAL_API_KEY", "should-not-be-visible")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-visible")

        free = _subprocess_env(needs_llm=False)
        assert "MISTRAL_API_KEY" not in free
        assert "ANTHROPIC_API_KEY" not in free

        paid = _subprocess_env(needs_llm=True)
        assert paid["MISTRAL_API_KEY"] == "should-not-be-visible"
