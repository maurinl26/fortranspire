"""GitHub App surface for fortranspire (issue #50).

Turns `/fortranspire <verb> <path>` comments on issues and pull requests
into pipeline runs, and publishes the result as a comment — plus a branch
and a pull request for the verbs that rewrite code.

Layering, from untrusted input inwards:

* :mod:`.commands`  — parses the comment; refuses anything ambiguous
* :mod:`.auth`      — webhook signature, App JWT, installation tokens
* :mod:`.policy`    — installation allow-list and actor permission
* :mod:`.runner`    — runs the verb via the `fortranspire` CLI, jailed
* :mod:`.github`    — the REST and git calls the flow needs
* :mod:`.app`       — the Starlette webhook receiver that wires it together

Every transformation goes through the same console script the CLI and the
MCP server use, so the App adds a trigger and an authorisation layer, not
a second implementation.
"""
from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Console entry point — `fortranspire github-app`."""
    from fortranspire.github_app.app import main as _main

    _main()
