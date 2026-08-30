"""GitHub App credentials: webhook signatures and installation tokens.

Three distinct secrets live here and must not be confused:

* the **webhook secret**, shared with GitHub, proves a delivery really came
  from GitHub and was not forged by anyone who found the public URL;
* the **App private key**, which signs a short-lived JWT proving we are the
  App;
* the **installation token**, minted per installation from that JWT, which
  is what actually talks to a repository. It expires in an hour and is
  scoped to one installation — it is never logged and never returned to a
  caller.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

GITHUB_API = os.getenv("GITHUB_API_URL", "https://api.github.com")

# GitHub rejects a JWT whose `exp` is more than 10 minutes out. Stay well
# inside that so clock skew on the host cannot invalidate every request.
_JWT_TTL_SECONDS = 480
# Refresh an installation token this long before it actually expires.
_TOKEN_REFRESH_MARGIN = 300


class ConfigError(RuntimeError):
    """The App is not configured well enough to run safely."""


def verify_signature(body: bytes, signature_header: str | None, secret: str) -> bool:
    """Constant-time check of the `X-Hub-Signature-256` header.

    Returns False for a missing, malformed or wrong signature. The caller
    must treat False as "drop the delivery" — an unsigned webhook endpoint
    is a remote trigger for anyone who learns the URL.
    """
    if not secret:
        raise ConfigError(
            "webhook secret is empty — refusing to accept unverified deliveries"
        )
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


def _load_private_key() -> str:
    """Read the App private key from a path or an inline PEM env var."""
    inline = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if inline:
        # Env vars round-trip newlines as literal `\n` through most secret
        # stores and container runtimes; PEM parsing needs the real thing.
        return inline.replace("\\n", "\n")
    path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
    if not path:
        raise ConfigError(
            "set GITHUB_APP_PRIVATE_KEY (inline PEM) or GITHUB_APP_PRIVATE_KEY_PATH"
        )
    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read the App private key at {path}: {exc}") from exc


def app_jwt(app_id: str | None = None, *, now: int | None = None) -> str:
    """Mint the short-lived JWT that identifies the App itself."""
    import jwt  # PyJWT — only needed on the App path, imported lazily

    app_id = app_id or os.getenv("GITHUB_APP_ID", "")
    if not app_id:
        raise ConfigError("GITHUB_APP_ID is not set")

    issued = int(now if now is not None else time.time())
    payload = {
        # Backdate by a minute: GitHub rejects a JWT issued in its future,
        # and the host clock is routinely a few seconds ahead.
        "iat": issued - 60,
        "exp": issued + _JWT_TTL_SECONDS,
        "iss": app_id,
    }
    return jwt.encode(payload, _load_private_key(), algorithm="RS256")


@dataclass
class InstallationToken:
    """A minted installation token and the moment it stops being usable."""

    token: str
    expires_at: float

    @property
    def is_fresh(self) -> bool:
        return time.time() < self.expires_at - _TOKEN_REFRESH_MARGIN

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never let a token reach a log line or a traceback.
        return f"InstallationToken(token='***', expires_at={self.expires_at})"


class TokenCache:
    """Mints and reuses installation tokens.

    GitHub rate-limits token minting, and a busy repository produces many
    webhook deliveries in a burst, so tokens are cached per installation
    until shortly before they expire.
    """

    def __init__(self) -> None:
        self._tokens: dict[int, InstallationToken] = {}

    def get(self, installation_id: int, *, client: httpx.Client | None = None) -> str:
        cached = self._tokens.get(installation_id)
        if cached and cached.is_fresh:
            return cached.token

        owns_client = client is None
        client = client or httpx.Client(timeout=30)
        try:
            response = client.post(
                f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt()}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        finally:
            if owns_client:
                client.close()

        if response.status_code != 201:
            # The body can echo request details; report the status only.
            raise ConfigError(
                f"could not mint an installation token for {installation_id} "
                f"(HTTP {response.status_code})"
            )

        payload = response.json()
        minted = InstallationToken(
            token=payload["token"],
            # `expires_at` is ISO-8601; fall back to the documented 1 hour
            # rather than depending on parsing it.
            expires_at=time.time() + 3600,
        )
        self._tokens[installation_id] = minted
        return minted.token
