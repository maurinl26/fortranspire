"""Parser tests for the `/fortranspire` slash command (issue #50).

The parser sits directly on untrusted input — anyone who can comment on a
public pull request can reach it — so the cases that matter most are the
refusals.
"""
from __future__ import annotations

import pytest

from fortranspire.github_app.commands import (
    Command,
    CommandError,
    mentions_command,
    parse,
)


class TestAccepts:
    def test_minimal_command(self):
        assert parse("/fortranspire explain src/kernel.f90") == Command(
            verb="explain", path="src/kernel.f90", raw="/fortranspire explain src/kernel.f90"
        )

    def test_command_embedded_in_prose(self):
        parsed = parse("Hi team, could you run this?\n\n/fortranspire port src/a.f90\n\nThanks!")
        assert parsed is not None and parsed.verb == "port"

    def test_target_and_fail_on(self):
        assert parse("/fortranspire port a.f90 --target jax").target == "jax"
        assert parse("/fortranspire analyze src/ --fail-on warning").fail_on == "warning"

    def test_quoted_path_with_spaces(self):
        assert parse('/fortranspire explain "src/my kernel.f90"').path == "src/my kernel.f90"

    def test_directory_path(self):
        assert parse("/fortranspire analyze src/").path == "src/"


class TestIgnores:
    """Comments that do not address the App produce None, not an error."""

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "just a normal comment",
            "we should use /fortranspire someday",  # not at line start
            "```\n/fortranspire port x.f90\n```",  # inside a code fence
            "> /fortranspire port x.f90",  # quoting someone else
        ],
    )
    def test_not_a_command(self, body):
        assert parse(body) is None
        assert mentions_command(body) is False


class TestRefuses:
    """A comment that addresses the App but is malformed must be named as such."""

    @pytest.mark.parametrize(
        "body,expected",
        [
            ("/fortranspire", "missing a verb"),
            ("/fortranspire frobnicate x.f90", "unknown verb"),
            ("/fortranspire explain", "needs a path"),
            ("/fortranspire port a.f90 b.f90", "one path per command"),
            ("/fortranspire port a.f90 --target cuda", "unknown --target"),
            ("/fortranspire port a.f90 --nope", "unknown option"),
            ("/fortranspire explain a.f90 --target gpu", "only applies to `port`"),
            ("/fortranspire port a.f90 --fail-on error", "only applies to `analyze`"),
            ('/fortranspire explain "unbalanced', "could not parse"),
        ],
    )
    def test_malformed(self, body, expected):
        with pytest.raises(CommandError, match=expected):
            parse(body)

    @pytest.mark.parametrize(
        "path",
        [
            "../../etc/passwd",
            "../secrets.f90",
            "/etc/passwd",
            "~/.ssh/id_rsa",
        ],
    )
    def test_path_escape_is_refused(self, path):
        """Defence in depth — the runner jails too, but say why here."""
        with pytest.raises(CommandError):
            parse(f"/fortranspire explain {path}")

    def test_single_dash_option_where_a_path_belongs(self):
        """A `-x` is not a known option and must not be taken for a path."""
        with pytest.raises(CommandError, match="expected a path"):
            parse("/fortranspire explain -x")

    def test_options_without_a_path_still_demand_one(self):
        with pytest.raises(CommandError, match="needs a path"):
            parse("/fortranspire explain --target gpu")


class TestCostClassification:
    """Whether a verb spends tokens decides which gate it must pass."""

    @pytest.mark.parametrize("verb", ["explain", "analyze", "graph"])
    def test_deterministic_verbs_are_free(self, verb):
        assert parse(f"/fortranspire {verb} src/").needs_llm is False

    @pytest.mark.parametrize("verb", ["port", "doc"])
    def test_llm_verbs_are_flagged(self, verb):
        assert parse(f"/fortranspire {verb} src/a.f90").needs_llm is True
