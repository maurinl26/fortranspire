"""Tests for the guards and relaxations library (issue #73).

Two properties carry the whole module, and neither is obvious enough to
take on trust:

* a **relaxation converges** to the hard form as its parameter tightens —
  otherwise it is not a relaxation of anything, just a different function;
* a **guard leaves forward values alone** where the original was defined —
  otherwise it is a relaxation wearing a guard's name, and it changes the
  model without saying so.

The stability claims matter too: the textbook softmax overflows, which is
the reason this lives in a library instead of being re-derived by a model
on every run.
"""
from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fortranspire.jax_smooth import (  # noqa: E402
    interp_table,
    safe_divide,
    safe_log,
    safe_sqrt,
    smooth_abs,
    smooth_argmax,
    smooth_clamp,
    smooth_max,
    smooth_min,
    smooth_sign,
    smooth_step,
    smooth_where,
)


def grad_is_finite(fn, x) -> bool:
    return bool(np.isfinite(np.asarray(jax.grad(fn)(x))).all())


# ── Relaxations converge to the hard form ──────────────────────────────────

class TestConvergence:
    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.0, 0.5), (7.0, 7.0), (0.0, -0.0)])
    def test_smooth_max_tends_to_max(self, a, b):
        errors = [abs(float(smooth_max(a, b, beta)) - max(a, b))
                  for beta in (10.0, 100.0, 1000.0)]
        # Non-increasing, not strictly decreasing: for well-separated
        # arguments the bias decays like exp(-beta*|a-b|) and reaches
        # exactly zero in float64 well before beta = 1000, so the last two
        # entries are legitimately equal.
        assert errors[0] >= errors[1] >= errors[2]
        assert errors[0] > errors[-1] or errors[0] == 0.0
        assert errors[-1] < 1e-2

    def test_smooth_max_bias_at_equality_is_log2_over_beta(self):
        """The worst case, and it is exact — worth pinning as a number."""
        for beta in (10.0, 50.0, 500.0):
            bias = float(smooth_max(3.0, 3.0, beta)) - 3.0
            assert bias == pytest.approx(math.log(2.0) / beta, rel=1e-9)

    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.0, 0.5), (7.0, 7.0)])
    def test_smooth_min_tends_to_min(self, a, b):
        assert float(smooth_min(a, b, 1000.0)) == pytest.approx(min(a, b), abs=1e-2)

    @pytest.mark.parametrize("x", [-2.0, -0.1, 0.0, 0.1, 2.0])
    def test_smooth_abs_tends_to_abs(self, x):
        assert float(smooth_abs(x, 1e-8)) == pytest.approx(abs(x), abs=1e-6)

    def test_smooth_abs_bias_at_zero_is_eps(self):
        for eps in (1e-2, 1e-4):
            assert float(smooth_abs(0.0, eps)) == pytest.approx(eps, rel=1e-9)

    @pytest.mark.parametrize("x", [-5.0, -0.5, 0.5, 5.0])
    def test_smooth_sign_tends_to_sign(self, x):
        assert float(smooth_sign(x, 1e-6)) == pytest.approx(math.copysign(1.0, x), abs=1e-6)

    def test_smooth_sign_is_zero_at_the_origin(self):
        assert float(smooth_sign(0.0, 1e-3)) == 0.0

    @pytest.mark.parametrize("x,expected", [(-1.0, 0.0), (1.0, 1.0)])
    def test_smooth_step_tends_to_heaviside(self, x, expected):
        assert float(smooth_step(x, 0.0, 1e-3)) == pytest.approx(expected, abs=1e-6)

    def test_smooth_step_is_one_half_at_the_threshold(self):
        assert float(smooth_step(2.0, 2.0, 1e-3)) == pytest.approx(0.5)

    @pytest.mark.parametrize("x,expected", [(-5.0, -1.0), (0.5, 0.5), (5.0, 1.0)])
    def test_smooth_clamp_tends_to_hard_clamp(self, x, expected):
        assert float(smooth_clamp(x, -1.0, 1.0, 5000.0)) == pytest.approx(expected, abs=1e-3)

    def test_smooth_where_tends_to_the_branch_selection(self):
        assert float(smooth_where(1.0, 0.0, 10.0, -10.0, 1e-4)) == pytest.approx(10.0, abs=1e-3)
        assert float(smooth_where(-1.0, 0.0, 10.0, -10.0, 1e-4)) == pytest.approx(-10.0, abs=1e-3)

    def test_smooth_argmax_tends_to_argmax(self):
        values = jnp.asarray([0.1, 0.4, 3.0, 0.2])
        assert float(smooth_argmax(values, 1000.0)) == pytest.approx(2.0, abs=1e-3)

    def test_smooth_argmax_on_a_tie_returns_the_mean_position(self):
        """The sensible continuous answer, and worth stating."""
        assert float(smooth_argmax(jnp.asarray([5.0, 5.0]), 100.0)) == pytest.approx(0.5)


# ── They are actually differentiable ───────────────────────────────────────

class TestDifferentiability:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: smooth_max(x, 1.0, 20.0),
            lambda x: smooth_min(x, 1.0, 20.0),
            lambda x: smooth_abs(x, 1e-3),
            lambda x: smooth_sign(x, 1e-2),
            lambda x: smooth_step(x, 0.0, 1e-2),
            lambda x: smooth_clamp(x, -1.0, 1.0, 20.0),
        ],
    )
    def test_gradient_is_finite_at_the_kink(self, fn):
        """The point of the exercise: hard forms are non-smooth exactly here."""
        assert grad_is_finite(fn, 0.0)
        assert grad_is_finite(fn, 1.0)

    def test_smooth_abs_gradient_is_continuous_through_zero(self):
        """`abs` jumps from -1 to +1; the relaxation must not."""
        d = jax.grad(lambda x: smooth_abs(x, 1e-2))
        left, right = float(d(-1e-4)), float(d(1e-4))
        assert abs(right - left) < 0.1

    def test_hard_max_gradient_is_one_sided(self):
        """Why a relaxation is sometimes needed at all.

        `jnp.maximum` is differentiable — it simply routes the whole
        gradient to one argument and zero to the other. That is fine for a
        forward solve and wrong for a sensitivity study.
        """
        d = jax.grad(lambda a: jnp.maximum(a, 1.0))
        assert float(d(0.5)) == 0.0
        assert float(jax.grad(lambda a: smooth_max(a, 1.0, 5.0))(0.5)) > 0.0


# ── Numerical stability: the reason this is a library ──────────────────────

class TestStability:
    @pytest.mark.parametrize("value", [1e2, 1e3, 1e4])
    def test_smooth_max_survives_arguments_that_overflow_the_naive_form(self, value):
        """This is the reason the relaxations live in a library.

        The textbook softmax is ``log(exp(beta*a) + exp(beta*b)) / beta``.
        With ``beta = 50`` and ``a = 100`` the inner exponential is
        ``exp(5000)``, far past the float64 ceiling near ``exp(709)``, so
        the naive form returns infinity. A model re-deriving this
        expression on each run writes the textbook version.
        """
        beta = 50.0
        with np.errstate(over="ignore"):  # the overflow is the point
            assert np.isinf(np.exp(np.float64(beta * value)))

        result = float(smooth_max(value, value / 2.0, beta))
        assert math.isfinite(result)
        assert result == pytest.approx(value, rel=1e-6)

    def test_smooth_max_is_finite_for_large_negative_arguments(self):
        assert math.isfinite(float(smooth_max(-1e4, -2e4, 50.0)))


# ── Guards leave forward values alone ──────────────────────────────────────

class TestGuards:
    @pytest.mark.parametrize("x", [1.0, 4.0, 100.0])
    def test_safe_sqrt_matches_sqrt_where_defined(self, x):
        assert float(safe_sqrt(x, 1e-12)) == pytest.approx(math.sqrt(x))

    def test_safe_sqrt_has_a_finite_gradient_at_zero(self):
        """`d/dx sqrt(x)` diverges at 0 — the guard bounds it."""
        assert not math.isfinite(float(jax.grad(jnp.sqrt)(0.0)))
        assert grad_is_finite(lambda x: safe_sqrt(x, 1e-6), 0.0)

    @pytest.mark.parametrize("num,den", [(1.0, 2.0), (-3.0, 4.0), (5.0, -2.0)])
    def test_safe_divide_matches_division_where_defined(self, num, den):
        assert float(safe_divide(num, den, 1e-12)) == pytest.approx(num / den)

    def test_safe_divide_preserves_the_denominator_sign(self):
        """Flooring the magnitude must not flip the result's sign."""
        assert float(safe_divide(1.0, -1e-12, 1e-6)) < 0
        assert float(safe_divide(1.0, 1e-12, 1e-6)) > 0

    def test_safe_divide_gradient_is_finite_at_zero_denominator(self):
        assert grad_is_finite(lambda d: safe_divide(1.0, d, 1e-3), 0.0)

    def test_safe_log_matches_log_where_defined(self):
        assert float(safe_log(2.5, 1e-12)) == pytest.approx(math.log(2.5))

    def test_safe_log_gradient_is_finite_at_zero(self):
        assert grad_is_finite(lambda x: safe_log(x, 1e-4), 0.0)


class TestTableLookup:
    def test_interpolation_is_exact_at_the_nodes(self):
        nodes = [0.0, 1.0, 2.0]
        values = [10.0, 20.0, 40.0]
        for node, value in zip(nodes, values):
            assert float(interp_table(node, nodes, values)) == pytest.approx(value)

    def test_interpolation_has_a_non_zero_gradient_between_nodes(self):
        """A nearest-neighbour lookup is differentiable in name only."""
        d = jax.grad(lambda x: interp_table(x, [0.0, 1.0, 2.0], [10.0, 20.0, 40.0]))
        assert float(d(0.5)) == pytest.approx(10.0)
        assert float(d(1.5)) == pytest.approx(20.0)
