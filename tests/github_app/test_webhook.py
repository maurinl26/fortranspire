"""Webhook-transport tests for the GitHub App (issue #50).

Drives the real Starlette app through a test client. The signature check
is the App's outermost defence — a forged delivery must never reach the
parser, let alone the runner — so it gets the most attention here.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient  # noqa: E402

from fortranspire.github_app import app as app_module  # noqa: E402

SECRET = "webhook-secret-for-tests"


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_WEBHOOK_SECRET", SECRET)


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _comment_payload(body: str = "/fortranspire explain src/kernel.f90", **overrides):
    payload = {
        "action": "created",
        "comment": {"body": body, "user": {"login": "maintainer", "type": "User"}},
        "issue": {"number": 12},
        "repository": {"full_name": "acme/phyex", "default_branch": "main"},
        "installation": {"id": 42},
    }
    payload.update(overrides)
    return payload


def _post(client, payload, *, event="issue_comment", signature=None):
    body = json.dumps(payload).encode()
    headers = {
        "X-GitHub-Event": event,
        "X-Hub-Signature-256": signature if signature is not None else _sign(body),
        "Content-Type": "application/json",
    }
    return client.post("/webhooks/github", content=body, headers=headers)


class TestSignature:
    def test_health_needs_no_signature(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["service"] == "fortranspire-github-app"

    def test_unsigned_delivery_is_rejected(self, client):
        """No `X-Hub-Signature-256` at all — the public-URL attack."""
        response = _post(client, _comment_payload(), signature="")
        assert response.status_code == 401

    def test_wrong_signature_is_rejected(self, client):
        response = _post(client, _comment_payload(), signature="sha256=" + "0" * 64)
        assert response.status_code == 401

    def test_tampered_body_is_rejected(self, client):
        """A signature valid for a *different* body must not be accepted."""
        signature = _sign(json.dumps(_comment_payload()).encode())
        response = _post(
            client,
            _comment_payload("/fortranspire port src/kernel.f90"),
            signature=signature,
        )
        assert response.status_code == 401

    def test_missing_server_secret_is_a_server_error_not_an_open_door(
        self, client, monkeypatch
    ):
        monkeypatch.delenv("GITHUB_APP_WEBHOOK_SECRET", raising=False)
        response = _post(client, _comment_payload(), signature="sha256=" + "0" * 64)
        assert response.status_code == 500


class TestEventFiltering:
    """Cheap rejections that must happen before any GitHub API call."""

    def test_ping_is_answered(self, client):
        assert _post(client, {}, event="ping").status_code == 200

    def test_unrelated_event_is_ignored(self, client):
        response = _post(client, _comment_payload(), event="push")
        assert response.json()["detail"] == "ignored"

    def test_edited_comment_is_ignored(self, client):
        response = _post(client, _comment_payload(**{"action": "edited"}))
        assert response.json()["detail"] == "ignored"

    def test_comment_without_a_command_is_ignored(self, client):
        response = _post(client, _comment_payload("nice work everyone"))
        assert response.json()["detail"] == "ignored"

    def test_bot_comments_are_ignored(self, client):
        """Our own summary comments must not re-trigger the App."""
        payload = _comment_payload()
        payload["comment"]["user"]["type"] = "Bot"
        assert _post(client, payload).json()["detail"] == "ignored"


class TestAuthorisationIsReachedBeforeWork:
    def test_unapproved_installation_never_runs_a_job(self, client, monkeypatch):
        """No allow-list configured => refused, and nothing is submitted."""
        monkeypatch.delenv("FORTRANSPIRE_APP_INSTALLATIONS", raising=False)
        monkeypatch.setattr(app_module._TOKENS, "get", lambda *_a, **_k: "tok")

        submitted = []
        monkeypatch.setattr(
            app_module._POOL, "submit", lambda *a, **k: submitted.append(k)
        )
        monkeypatch.setattr(
            app_module, "actor_permission", lambda **_kwargs: "admin"
        )

        response = _post(client, _comment_payload())
        assert response.status_code == 202
        assert response.json()["detail"] == "refused"
        assert submitted == []

    def test_approved_installation_submits_the_job(self, client, monkeypatch, tmp_path):
        registry = tmp_path / "installations.json"
        registry.write_text(json.dumps([
            {"installation_id": 42, "repositories": ["acme/*"], "allow_llm": False},
        ]))
        monkeypatch.setenv("FORTRANSPIRE_APP_INSTALLATIONS", str(registry))
        monkeypatch.setattr(app_module._TOKENS, "get", lambda *_a, **_k: "tok")
        monkeypatch.setattr(app_module, "actor_permission", lambda **_kwargs: "write")

        submitted = []
        monkeypatch.setattr(
            app_module._POOL, "submit", lambda *a, **k: submitted.append(k)
        )

        response = _post(client, _comment_payload())
        assert response.status_code == 202
        assert response.json()["verb"] == "explain"
        assert len(submitted) == 1
        assert submitted[0]["command"].verb == "explain"

    def test_llm_verb_refused_when_installation_disallows_it(
        self, client, monkeypatch, tmp_path
    ):
        registry = tmp_path / "installations.json"
        registry.write_text(json.dumps([
            {"installation_id": 42, "repositories": ["acme/*"], "allow_llm": False},
        ]))
        monkeypatch.setenv("FORTRANSPIRE_APP_INSTALLATIONS", str(registry))
        monkeypatch.setattr(app_module._TOKENS, "get", lambda *_a, **_k: "tok")
        monkeypatch.setattr(app_module, "actor_permission", lambda **_kwargs: "write")

        posted = []
        monkeypatch.setattr(
            app_module.github, "post_comment",
            lambda **kwargs: posted.append(kwargs["body"]),
        )
        submitted = []
        monkeypatch.setattr(
            app_module._POOL, "submit", lambda *a, **k: submitted.append(k)
        )

        response = _post(client, _comment_payload("/fortranspire port src/k.f90"))
        assert response.json()["detail"] == "refused"
        assert submitted == []
        assert posted and "explain" in posted[0]

    def test_malformed_command_is_explained_not_run(
        self, client, monkeypatch, tmp_path
    ):
        registry = tmp_path / "installations.json"
        registry.write_text(json.dumps([
            {"installation_id": 42, "repositories": ["acme/*"], "allow_llm": True},
        ]))
        monkeypatch.setenv("FORTRANSPIRE_APP_INSTALLATIONS", str(registry))
        monkeypatch.setattr(app_module._TOKENS, "get", lambda *_a, **_k: "tok")
        monkeypatch.setattr(app_module, "actor_permission", lambda **_kwargs: "write")

        posted = []
        monkeypatch.setattr(
            app_module.github, "post_comment",
            lambda **kwargs: posted.append(kwargs["body"]),
        )
        submitted = []
        monkeypatch.setattr(
            app_module._POOL, "submit", lambda *a, **k: submitted.append(k)
        )

        response = _post(client, _comment_payload("/fortranspire port ../../etc/passwd"))
        assert response.json()["detail"] == "bad command"
        assert submitted == []
        assert posted and "escapes the repository root" in posted[0]
