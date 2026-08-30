"""Parser for the `/fortranspire …` slash commands (issue #50).

The App reacts to comments written by humans on issues and pull requests,
so the parser has to be strict in both directions: it must find the command
inside a comment that may also contain prose, quoted text and code fences,
and it must refuse anything it does not fully understand rather than guess.
A mis-parse here either runs the wrong verb on the wrong path or, worse,
runs a token-spending LLM port when the user asked for a free estimate.

Grammar::

    /fortranspire <verb> <path> [options]

    verb    := explain | analyze | port | doc | graph
    path    := repository-relative path, no `..`, no leading `/`
    options := --target gpu|jax        (port only)
               --fail-on error|warning|note   (analyze only)
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Literal

COMMAND_PREFIX = "/fortranspire"

# Verbs that never call an LLM and never spend a token. They are safe to
# offer more widely than `port`, which writes code and costs money.
DETERMINISTIC_VERBS: frozenset[str] = frozenset({"explain", "analyze", "graph"})
LLM_VERBS: frozenset[str] = frozenset({"port", "doc"})
VERBS: frozenset[str] = DETERMINISTIC_VERBS | LLM_VERBS

TARGETS: frozenset[str] = frozenset({"gpu", "jax"})
SEVERITIES: frozenset[str] = frozenset({"error", "warning", "note"})

# A fenced code block may legitimately contain a line that looks like a
# command (a doc example, a quoted comment). Those must not trigger a run.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


class CommandError(ValueError):
    """The comment addressed the App but the command was not usable.

    Carries a message written for the person who typed the comment — it is
    posted back to them, so it names the mistake and the fix.
    """


@dataclass(frozen=True)
class Command:
    """One parsed, validated invocation."""

    verb: Literal["explain", "analyze", "port", "doc", "graph"]
    path: str
    target: Literal["gpu", "jax"] = "gpu"
    fail_on: Literal["error", "warning", "note"] = "error"
    raw: str = field(default="", repr=False)

    @property
    def needs_llm(self) -> bool:
        """Whether running this command spends tokens and writes code."""
        return self.verb in LLM_VERBS


def _strip_noise(body: str) -> str:
    """Remove fenced code and quoted (`>`) lines before looking for a command."""
    without_fences = _FENCE_RE.sub("", body)
    return "\n".join(
        line for line in without_fences.splitlines() if not line.lstrip().startswith(">")
    )


def mentions_command(body: str | None) -> bool:
    """Cheap pre-filter: does this comment address the App at all?

    Used to drop the overwhelming majority of webhook deliveries without
    doing any work, before any authorisation lookup or API call.
    """
    if not body:
        return False
    return any(
        line.strip().startswith(COMMAND_PREFIX)
        for line in _strip_noise(body).splitlines()
    )


def _validate_path(raw_path: str) -> str:
    """Reject anything that is not a plain repository-relative path.

    The runner also jails the resolved path against the checkout root, so
    this is defence in depth — but rejecting here gives the commenter a
    clear error instead of an opaque permission failure.
    """
    path = raw_path.strip()
    if not path:
        raise CommandError("a path is required: `/fortranspire explain src/kernel.f90`")
    if path.startswith("-"):
        raise CommandError(f"expected a path but got the option `{path}`")
    if path.startswith(("/", "~")):
        raise CommandError(
            f"`{path}` must be relative to the repository root (drop the leading `/`)"
        )
    if ".." in path.split("/"):
        raise CommandError(f"`{path}` escapes the repository root")
    if "\x00" in path:
        raise CommandError("path contains a null byte")
    return path


def _take_option(tokens: list[str], index: int, name: str, allowed: frozenset[str]) -> str:
    """Read `--name value`, validating the value against `allowed`."""
    if index >= len(tokens):
        raise CommandError(f"`{name}` needs a value ({'|'.join(sorted(allowed))})")
    value = tokens[index]
    if value not in allowed:
        raise CommandError(
            f"unknown {name} `{value}` — expected one of {', '.join(sorted(allowed))}"
        )
    return value


def parse(body: str) -> Command | None:
    """Extract the command from a comment body.

    Returns ``None`` when the comment does not address the App at all, and
    raises :class:`CommandError` when it does but is malformed — the caller
    distinguishes "not for us, stay silent" from "for us, explain the
    mistake".
    """
    if not body:
        return None

    line = next(
        (
            ln.strip()
            for ln in _strip_noise(body).splitlines()
            if ln.strip().startswith(COMMAND_PREFIX)
        ),
        None,
    )
    if line is None:
        return None

    try:
        tokens = shlex.split(line)
    except ValueError as exc:  # unbalanced quote
        raise CommandError(f"could not parse the command ({exc})") from exc

    tokens = tokens[1:]  # drop the `/fortranspire` prefix itself
    if not tokens:
        raise CommandError(
            "missing a verb — try `/fortranspire explain <path>` "
            f"(verbs: {', '.join(sorted(VERBS))})"
        )

    verb = tokens[0]
    if verb not in VERBS:
        raise CommandError(
            f"unknown verb `{verb}` — expected one of {', '.join(sorted(VERBS))}"
        )

    path: str | None = None
    target = "gpu"
    fail_on = "error"
    # Track whether the option was *given*, not whether it differs from the
    # default: `--target gpu` on `explain` is still a misuse worth naming,
    # and comparing against the default silently accepted it.
    saw_target = False
    saw_fail_on = False

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--target":
            target = _take_option(tokens, i + 1, "--target", TARGETS)
            saw_target = True
            i += 2
        elif token == "--fail-on":
            fail_on = _take_option(tokens, i + 1, "--fail-on", SEVERITIES)
            saw_fail_on = True
            i += 2
        elif token.startswith("--"):
            raise CommandError(f"unknown option `{token}`")
        elif path is None:
            path = _validate_path(token)
            i += 1
        else:
            raise CommandError(
                f"unexpected extra argument `{token}` — one path per command"
            )

    if path is None:
        raise CommandError(f"`{verb}` needs a path: `/fortranspire {verb} src/kernel.f90`")

    if saw_target and verb != "port":
        raise CommandError("`--target` only applies to `port`")
    if saw_fail_on and verb != "analyze":
        raise CommandError("`--fail-on` only applies to `analyze`")

    return Command(verb=verb, path=path, target=target, fail_on=fail_on, raw=line)
