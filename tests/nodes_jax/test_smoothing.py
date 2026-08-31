"""Tests for the smoothing modes and non-smooth detection (issue #73).

The distinction these pin is the one that matters: a **guard** repairs a
translation and leaves the forward values alone; a **relaxation** changes
what the code computes and is therefore a modelling decision. Conflating
them — or letting a relaxation happen by default — is exactly the silent
wrongness the whole Phase 2 rework exists to prevent.
"""
from __future__ import annotations

import re

import pytest

from fortranspire.jax_smooth import (
    GUARDS,
    NON_SMOOTH,
    RELAXATIONS,
    catalogue_for_prompt,
)


class TestCatalogue:
    def test_every_entry_has_a_pattern_that_compiles(self):
        for entry in NON_SMOOTH:
            re.compile(entry.pattern, re.IGNORECASE)

    def test_every_replacement_names_a_real_function(self):
        """A catalogue pointing at a function that does not exist is worse
        than no catalogue: the model imports it and the kernel fails to load."""
        import fortranspire.jax_smooth as module

        for entry in NON_SMOOTH:
            if entry.replacement:
                assert hasattr(module, entry.replacement), entry.key

    def test_constructs_without_a_replacement_say_what_to_do_instead(self):
        """`FLOOR` and `DO WHILE` have no relaxation — silence would read
        as "unhandled" rather than "genuinely discrete"."""
        for entry in NON_SMOOTH:
            if entry.replacement is None:
                assert len(entry.limit) > 30, entry.key

    def test_truncation_notes_that_gradcheck_cannot_see_it(self):
        """The one class finite differences provably miss, so the static
        rule is the only way to catch it. That has to be written down."""
        entry = next(e for e in NON_SMOOTH if e.key == "truncation")
        assert "gradcheck" in entry.why

    def test_while_loop_notes_the_reverse_mode_restriction(self):
        entry = next(e for e in NON_SMOOTH if e.key == "while")
        assert "reverse-mode" in entry.why or "reverse" in entry.why

    def test_prompt_rendering_covers_every_entry(self):
        rendered = catalogue_for_prompt()
        for entry in NON_SMOOTH:
            assert entry.key.upper() in rendered


class TestDetection:
    FORTRAN = """
      SUBROUTINE limiter(phi, r, n)
        INTEGER, INTENT(IN) :: n
        REAL(KIND=8), INTENT(IN) :: r(n)
        REAL(KIND=8), INTENT(INOUT) :: phi(n)
        INTEGER :: i
        DO i = 1, n
          phi(i) = MAX(0.0d0, MIN(1.0d0, r(i))) * ABS(r(i))
        END DO
      END SUBROUTINE limiter
    """

    def _keys(self, source: str) -> set[str]:
        from fortranspire.agent.analyze import _non_smooth_constructs

        return {key for key, _, _ in _non_smooth_constructs(source)}

    def test_finds_the_constructs_present(self):
        assert {"max", "min", "abs"} <= self._keys(self.FORTRAN)

    def test_reports_each_construct_once(self):
        """A kernel using MAX forty times needs one finding, not forty."""
        from fortranspire.agent.analyze import _non_smooth_constructs

        source = "x = MAX(a,b)\ny = MAX(c,d)\nz = MAX(e,f)\n"
        assert len([k for k, _, _ in _non_smooth_constructs(source) if k == "max"]) == 1

    def test_comments_do_not_produce_findings(self):
        """`! a minmod limiter uses MAX` is prose, not code."""
        assert self._keys("! this limiter uses MAX and ABS\nx = y + 1\n") == set()

    def test_identifiers_containing_a_keyword_are_not_matched(self):
        """`IMAX` is a loop bound, not a MAX call."""
        assert "max" not in self._keys("DO i = 1, IMAX\n  x = y\nEND DO\n")

    def test_do_while_is_detected(self):
        assert "while" in self._keys("      DO WHILE (err > tol)\n        err = err / 2\n      END DO\n")

    def test_clean_source_yields_nothing(self):
        assert self._keys("x = a * b + c\n") == set()

    def test_rule_is_registered_and_labelled(self):
        from fortranspire.agent.analyze import RULES
        from fortranspire.agent.explain import _RISK_LABELS

        assert RULES["FORT031"]["severity"] == "note"
        assert "FORT031" in _RISK_LABELS


class TestSmoothingModes:
    def test_default_is_faithful(self):
        """A translation that silently changes the model is worse than one
        that is honestly non-differentiable."""
        import inspect

        from fortranspire.agent.cli import translate_file

        assert 'smoothing: str = "none"' in inspect.signature(translate_file).__str__() \
            or inspect.signature(translate_file).parameters["smoothing"].default == "none"

    @pytest.mark.parametrize("mode", ["none", "guarded", "smooth"])
    def test_every_mode_has_an_instruction(self, mode):
        from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS, SMOOTHING_MODES

        assert mode in SMOOTHING_MODES
        assert len(_INSTRUCTIONS[mode]) > 80

    def test_none_forbids_relaxation_explicitly(self):
        from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS

        assert "not" in _INSTRUCTIONS["none"].lower()

    def test_guarded_forbids_relaxation_but_allows_guards(self):
        from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS

        text = _INSTRUCTIONS["guarded"]
        assert "safe_sqrt" in text
        assert "smooth_max" not in text
        assert "identical" in text or "unchanged" in text

    def test_smooth_requires_naming_each_relaxation(self):
        """The report is what makes the modelling decision visible."""
        from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS

        assert "relaxed:" in _INSTRUCTIONS["smooth"]

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_prompt_carries_mode_and_catalogue(self, lang):
        from fortranspire.prompts.loader import load_prompt
        from fortranspire.agent.nodes_jax.jax_kernel import _INSTRUCTIONS

        text = load_prompt(
            "jax_kernel", version="v1", lang=lang,
            signature="s", hints="h", fortran_code="f",
            smoothing_mode="smooth",
            smoothing_instruction=_INSTRUCTIONS["smooth"],
            smoothing_catalogue=catalogue_for_prompt(),
        )
        assert "smooth_max" in text
        assert "{" not in text.split("```")[0] or "{smoothing" not in text

    def test_guards_and_relaxations_are_disjoint(self):
        """Naming a relaxation as a guard would let it apply by default."""
        assert not (set(GUARDS) & set(RELAXATIONS))
