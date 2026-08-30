"""Who may run what, on which repository (issues #48 and #50).

Two independent gates, both of which must open:

1. **Installation allow-list** — the App may be installed on any repository
   by anyone who finds it. An installation the operator has not approved is
   answered with a polite refusal, never with a run. This is what keeps a
   BYO-key or hosted deployment from becoming an open compute faucet.
2. **Actor permission** — inside an approved installation, only users with
   `write` or `admin` on that repository may trigger anything. A drive-by
   comment on a public PR must not be able to spend tokens or write code.

On top of both, LLM verbs (`port`, `doc`) are gated separately from the
deterministic ones: an operator can enable free `explain`/`analyze` for a
repository while keeping token-spending runs on a shorter leash.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from fortranspire.github_app.auth import GITHUB_API

# Repository roles that may trigger the App. `triage` and `read` may not:
# they are routinely granted to people outside the maintainer team.
_WRITE_ROLES: frozenset[str] = frozenset({"write", "admin", "maintain"})


@dataclass(frozen=True)
class Installation:
    """Operator-approved configuration for one App installation."""

    installation_id: int
    # Repository full names (`owner/repo`), fnmatch patterns allowed.
    repositories: tuple[str, ...] = ("*",)
    # Whether token-spending verbs are permitted for this installation.
    allow_llm: bool = False
    # Extra GitHub logins allowed to trigger even without write access.
    extra_actors: tuple[str, ...] = ()

    def covers(self, repo_full_name: str) -> bool:
        return any(fnmatch.fnmatch(repo_full_name, pat) for pat in self.repositories)


@dataclass
class Policy:
    """The operator's allow-list, loaded from the environment."""

    installations: dict[int, Installation] = field(default_factory=dict)
    # When no allow-list is configured at all the App refuses everything —
    # failing closed is the only safe default for a public webhook.
    configured: bool = False

    @classmethod
    def from_env(cls) -> "Policy":
        """Load from `FORTRANSPIRE_APP_INSTALLATIONS` (a JSON file path).

        Schema::

            [
              {"installation_id": 12345,
               "repositories": ["myorg/*"],
               "allow_llm": true,
               "extra_actors": ["a-bot"]}
            ]
        """
        path = os.getenv("FORTRANSPIRE_APP_INSTALLATIONS")
        if not path:
            return cls()
        entries = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        policy = cls(configured=True)
        for entry in entries:
            installation = Installation(
                installation_id=int(entry["installation_id"]),
                repositories=tuple(entry.get("repositories", ("*",))),
                allow_llm=bool(entry.get("allow_llm", False)),
                extra_actors=tuple(entry.get("extra_actors", ())),
            )
            policy.installations[installation.installation_id] = installation
        return policy

    def installation_for(self, installation_id: int) -> Installation | None:
        return self.installations.get(installation_id)


@dataclass(frozen=True)
class Decision:
    """The outcome of the authorisation check."""

    allowed: bool
    # Message posted back to the commenter when refused. Empty when the
    # right response is silence (an unapproved installation gets no reply,
    # so the App cannot be used to probe which repos an operator approved).
    reason: str = ""
    silent: bool = False


def authorize(
    *,
    policy: Policy,
    installation_id: int,
    repo_full_name: str,
    actor: str,
    needs_llm: bool,
    actor_permission: str | None,
) -> Decision:
    """Decide whether `actor` may run this command on this repository.

    `actor_permission` is the role reported by the GitHub API
    (`admin`/`write`/`maintain`/`triage`/`read`/`none`), or None when it
    could not be determined — which is treated as no permission.
    """
    if not policy.configured:
        return Decision(
            False,
            "the fortranspire App has no installation allow-list configured; "
            "the operator must approve this installation before it can run.",
            silent=True,
        )

    installation = policy.installation_for(installation_id)
    if installation is None or not installation.covers(repo_full_name):
        return Decision(
            False,
            f"installation {installation_id} is not approved for {repo_full_name}.",
            silent=True,
        )

    is_maintainer = (actor_permission or "none") in _WRITE_ROLES
    if not (is_maintainer or actor in installation.extra_actors):
        return Decision(
            False,
            f"@{actor} needs write access on this repository to run fortranspire.",
        )

    if needs_llm and not installation.allow_llm:
        return Decision(
            False,
            "token-spending verbs (`port`, `doc`) are disabled for this "
            "installation. `/fortranspire explain <path>` and "
            "`/fortranspire analyze <path>` run for free and need no key.",
        )

    return Decision(True)


def actor_permission(
    *, repo_full_name: str, actor: str, token: str, client: httpx.Client | None = None
) -> str | None:
    """Ask GitHub what role `actor` holds on `repo_full_name`.

    Returns None when the answer cannot be established — the caller treats
    that as "no permission" rather than assuming the benign case.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.get(
            f"{GITHUB_API}/repos/{repo_full_name}/collaborators/{actor}/permission",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
    except httpx.HTTPError:
        return None
    finally:
        if owns_client:
            client.close()

    if response.status_code != 200:
        return None
    return response.json().get("permission")
