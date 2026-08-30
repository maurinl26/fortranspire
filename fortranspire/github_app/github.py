"""Minimal GitHub REST + git operations used by the App.

Deliberately small: only the calls the command flow needs, so there is no
third-party GitHub SDK to keep current. Every function takes an explicit
installation token — nothing reads credentials from module state, which
keeps two installations from ever borrowing each other's authority.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from fortranspire.github_app.auth import GITHUB_API

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubError(RuntimeError):
    """A GitHub API or git call failed."""


def _headers(token: str) -> dict[str, str]:
    return {**_HEADERS, "Authorization": f"Bearer {token}"}


def post_comment(
    *, repo_full_name: str, issue_number: int, body: str, token: str,
    client: httpx.Client | None = None,
) -> None:
    """Comment on an issue or pull request."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.post(
            f"{GITHUB_API}/repos/{repo_full_name}/issues/{issue_number}/comments",
            headers=_headers(token),
            json={"body": body},
        )
        if response.status_code >= 300:
            raise GitHubError(f"could not post a comment (HTTP {response.status_code})")
    finally:
        if owns:
            client.close()


def get_pull_request(
    *, repo_full_name: str, number: int, token: str, client: httpx.Client | None = None
) -> dict:
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.get(
            f"{GITHUB_API}/repos/{repo_full_name}/pulls/{number}",
            headers=_headers(token),
        )
        if response.status_code != 200:
            raise GitHubError(f"could not read PR #{number} (HTTP {response.status_code})")
        return response.json()
    finally:
        if owns:
            client.close()


def open_pull_request(
    *, repo_full_name: str, head: str, base: str, title: str, body: str, token: str,
    client: httpx.Client | None = None,
) -> str:
    """Open a PR and return its HTML URL."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.post(
            f"{GITHUB_API}/repos/{repo_full_name}/pulls",
            headers=_headers(token),
            json={"head": head, "base": base, "title": title, "body": body},
        )
        if response.status_code >= 300:
            raise GitHubError(f"could not open a pull request (HTTP {response.status_code})")
        return response.json()["html_url"]
    finally:
        if owns:
            client.close()


@dataclass(frozen=True)
class Checkout:
    """A cloned working tree plus the ref it came from."""

    path: Path
    base_ref: str


def _git(args: list[str], *, cwd: Path | None = None) -> str:
    """Run git, raising with the stderr but never echoing the argv.

    The clone URL carries the installation token, so a failure must not
    print the command line it failed on.
    """
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise GitHubError(f"git {args[0]} failed: {completed.stderr.strip()[:300]}")
    return completed.stdout


def clone(
    *, repo_full_name: str, ref: str, token: str, destination: Path, depth: int = 1
) -> Checkout:
    """Shallow-clone `ref` into `destination` using the installation token."""
    url = f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
    _git(["clone", "--depth", str(depth), "--branch", ref, url, str(destination)])
    # Drop the tokenised remote immediately: the URL would otherwise sit in
    # `.git/config` inside the job workspace for the life of the job.
    _git(["remote", "set-url", "origin",
          f"https://github.com/{repo_full_name}.git"], cwd=destination)
    return Checkout(path=destination, base_ref=ref)


def push_branch(
    *, checkout: Checkout, repo_full_name: str, branch: str, message: str,
    token: str, paths: list[Path] | None = None,
) -> bool:
    """Commit the working tree onto `branch` and push it. False if nothing changed."""
    _git(["checkout", "-b", branch], cwd=checkout.path)
    _git(["add", "--", *(str(p) for p in paths)] if paths else ["add", "-A"],
         cwd=checkout.path)

    staged = _git(["diff", "--cached", "--name-only"], cwd=checkout.path).strip()
    if not staged:
        return False

    _git(["-c", "user.name=fortranspire[bot]",
          "-c", "user.email=fortranspire[bot]@users.noreply.github.com",
          "commit", "-m", message], cwd=checkout.path)
    url = f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
    _git(["push", url, f"HEAD:refs/heads/{branch}"], cwd=checkout.path)
    return True
