"""Authorisation tests (issues #48, #50).

The App exposes a public webhook. Anyone who can comment on a public pull
request can reach the parser; only these gates stand between that and a
token-spending run that writes code. Each test below is one way in that
must stay shut.
"""
from __future__ import annotations

import json

import pytest

from fortranspire.github_app.policy import Installation, Policy, authorize


def _policy(**overrides) -> Policy:
    installation = Installation(
        installation_id=overrides.pop("installation_id", 42),
        repositories=overrides.pop("repositories", ("acme/phyex",)),
        allow_llm=overrides.pop("allow_llm", True),
        extra_actors=overrides.pop("extra_actors", ()),
    )
    return Policy(installations={installation.installation_id: installation}, configured=True)


def _authorize(policy: Policy, **overrides):
    kwargs = {
        "policy": policy,
        "installation_id": 42,
        "repo_full_name": "acme/phyex",
        "actor": "maintainer",
        "needs_llm": False,
        "actor_permission": "write",
    }
    kwargs.update(overrides)
    return authorize(**kwargs)


class TestFailsClosed:
    def test_no_allow_list_configured_refuses_everything(self):
        """An operator who has not configured the App must not run jobs."""
        decision = _authorize(Policy())
        assert decision.allowed is False
        assert decision.silent is True

    def test_unknown_installation_is_refused_silently(self):
        """Silence, so the App cannot be used to enumerate approved repos."""
        decision = _authorize(_policy(), installation_id=999)
        assert decision.allowed is False
        assert decision.silent is True

    def test_installation_not_approved_for_this_repository(self):
        decision = _authorize(_policy(repositories=("acme/other",)))
        assert decision.allowed is False
        assert decision.silent is True


class TestActorGating:
    @pytest.mark.parametrize("permission", ["read", "triage", "none", None])
    def test_non_maintainers_are_refused(self, permission):
        """A drive-by commenter on a public PR must not trigger a run."""
        decision = _authorize(_policy(), actor_permission=permission)
        assert decision.allowed is False
        assert decision.silent is False  # tell them why
        assert "write access" in decision.reason

    @pytest.mark.parametrize("permission", ["write", "admin", "maintain"])
    def test_maintainers_are_allowed(self, permission):
        assert _authorize(_policy(), actor_permission=permission).allowed is True

    def test_unresolvable_permission_is_treated_as_none(self):
        """A failed API lookup must not be read as consent."""
        assert _authorize(_policy(), actor_permission=None).allowed is False

    def test_extra_actor_allow_list(self):
        policy = _policy(extra_actors=("release-bot",))
        decision = _authorize(policy, actor="release-bot", actor_permission="read")
        assert decision.allowed is True


class TestLLMGate:
    def test_llm_verbs_refused_when_installation_disallows_them(self):
        decision = _authorize(_policy(allow_llm=False), needs_llm=True)
        assert decision.allowed is False
        # The refusal must point at the free alternative, not just say no.
        assert "explain" in decision.reason

    def test_free_verbs_still_run_when_llm_is_disabled(self):
        assert _authorize(_policy(allow_llm=False), needs_llm=False).allowed is True

    def test_llm_verbs_run_when_enabled(self):
        assert _authorize(_policy(allow_llm=True), needs_llm=True).allowed is True


class TestPolicyLoading:
    def test_from_env_reads_the_registry(self, tmp_path, monkeypatch):
        registry = tmp_path / "installations.json"
        registry.write_text(json.dumps([
            {"installation_id": 7, "repositories": ["acme/*"], "allow_llm": True},
        ]))
        monkeypatch.setenv("FORTRANSPIRE_APP_INSTALLATIONS", str(registry))

        policy = Policy.from_env()
        assert policy.configured is True
        assert policy.installation_for(7).covers("acme/phyex") is True
        assert policy.installation_for(7).covers("other/repo") is False

    def test_from_env_without_the_variable_is_unconfigured(self, monkeypatch):
        monkeypatch.delenv("FORTRANSPIRE_APP_INSTALLATIONS", raising=False)
        assert Policy.from_env().configured is False

    def test_allow_llm_defaults_to_false(self, tmp_path, monkeypatch):
        """Spending tokens must be opt-in per installation."""
        registry = tmp_path / "installations.json"
        registry.write_text(json.dumps([{"installation_id": 7}]))
        monkeypatch.setenv("FORTRANSPIRE_APP_INSTALLATIONS", str(registry))
        assert Policy.from_env().installation_for(7).allow_llm is False
